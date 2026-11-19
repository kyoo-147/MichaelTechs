---
title: "The Economics of Long Context"
description: "Attention costs grow with the square of the window while the cache grows linearly, and that gap decides what long context is worth."
pubDatetime: "2025-04-19T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Long Context
  - Attention
categories:
  - Math
---

A team builds a document assistant on a model with a million-token window. The whole contract folder goes in on every request. The demo is convincing. Then concurrency arrives: a few users hold long sessions open at once, and both response time and the bill move in directions the pricing sheet did not predict. Nothing is misconfigured. The prompt fits, which is the only question the vendor answered.

The questions that decide whether the product survives are different. How much arithmetic does the window require, and how much memory does it hold while a user waits. Reading costs time quadratic in length. Holding costs memory linear in length. The compute bill lands once per request. The capacity bill lands for as long as the session is open, and it is paid in concurrency.

## Two curves, and only one of them is priced

Take one layer, $d$ model dimensions, $N$ tokens in the window, and a prefill pass. Forming the scores costs $2N^2d$ multiply-adds, and multiplying by the values costs the same again, so across $L$ layers

$$F_{\text{prefill}} = 4LN^2d \quad \text{FLOPs}.$$

Decode is the other curve. Each new token attends to $N$ cached tokens: $4LNd$ FLOPs against $2LNd_{kv}b$ bytes read from the cache, where $d_{kv}$ is the per-head key and value width and $b$ is the bytes per element. Its arithmetic intensity is

$$\frac{4LNd}{2LNd_{kv}b} = \frac{2d}{d_{kv}b}.$$

For $d = d_{kv}$ and fp16 that is 1 FLOP per byte. An A100 asks for roughly 150. Decode attention sits two orders of magnitude below the machine balance, so the kernel waits on memory, and the memory it waits on grows with the context.

Prefill is quadratic and compute bound; decode is linear per token and bandwidth bound. Over an $N$-token window with $M$ output tokens the attention cost is $O(N^2 + NM)$.

| Context $N$ | KV cache, one sequence | Score entries per layer-head | Sequences on an 80 GiB device |
| --- | --- | --- | --- |
| 32,768 | 10 GiB | $1.1 \times 10^{9}$ | 8 |
| 131,072 | 40 GiB | $1.7 \times 10^{10}$ | 2 |
| 1,048,576 | 320 GiB | $1.1 \times 10^{12}$ | 0 |

Those figures come from a 70B-shaped configuration: 80 layers, 8 key and value heads, head dimension 128, fp16. That gives 320 KiB per cached token. The last column is the commercial one. Capacity is sold as a capability and consumed as concurrency. At 128k tokens an 80 GiB device holds two sessions and spends the rest on weights. At a million tokens a single session does not fit; the serving system moves state to host memory and the queue stops being about compute.

## Fitting is a capacity test. Using is a signal test.

Capacity tests pass by construction. When a vendor states a window size, the test behind the number is whether the API accepts the tokens and returns something. That is a memory test: the cache fits or it does not.

Signal has no hardware answer. Softmax normalises over $N$ positions, so if the logits are close to uniform each token carries weight near $1/N$ in the readout, and a single relevant fact arrives attenuated by the length of the window. To hold its influence constant as $N$ grows, the model must sharpen its logits, which means being confident about which positions matter before it reads them. That is the mechanism behind the pattern people report: single-fact lookup survives, while composition over several distant facts degrades. A needle in a haystack is a point lookup. Real work needs two or three facts that sit far apart to be combined, and each one arrives quieter as the prompt grows.

The operational definition follows. A context is used when changing a fact at position 500 and changing a fact at position 400,000 both change the answer. Swap one fact near the front and one near the back of your eval set, and count how often the output follows. If only the front swap moves the answer, the window is a place where tokens go, not a memory the model reads.

A window that fits and is not used is the most expensive retrieval system available. You paid attention prices for a lookup an index would have done before the request started.

## The contrarian case: retrieval and long context are different products

The standard framing says a large window makes retrieval unnecessary. I think that is backwards on the economics, and the two are not substitutes even when they produce the same answer. The difference is who pays for the noise.

A retriever does discrimination offline. The index is built once, the lookup is cheap and roughly independent of corpus size, and the ranking is an artefact you can inspect, log and fix. When it fails you can see the wrong passage.

Long context does the same discrimination inside the attention, on the request path, at quadratic price, and the result is not inspectable. The prompt is the ranking. When it fails you have no object to fix, only a model that was distracted by position 900,000 in a document you cannot audit.

So the products differ in the shape of their cost curve, not only their quality. Retrieved evidence is cheap per query and the cost is flat. Long context costs in proportion to what you include, plus capacity proportional to how many users wait. One architecture scales by keeping the window small and the index good. The other by buying memory and hoping the queue drains.

AlayaDB marks where this is heading. The authors decouple the KV cache and attention from inference, put them in a vector database, and treat cache management and attention as query processing with its own optimiser, comparing explicitly against KV cache disaggregation and retrieval-based sparse attention. That is the two categories converging at the systems layer while staying separate in the accounting. My reading is that selection does not disappear as the boundary moves. It relocates, from the retriever to whoever writes the prompt to the attention pattern the model was trained with, and it becomes unpriced and unaudited rather than unnecessary.

## The latency curve a user feels

Two numbers are felt, and they behave differently.

Time to first token is the prefill. It is $4LN^2d$ FLOPs against whatever the batch achieves, so it grows quadratically with prompt length and falls with batching. It is also mostly hideable: prefill chunks and streams, and a user who sees the first words is more patient than one who sees nothing.

The inter-token interval is the decode. Per token the model reads the whole cache, so

$$t_{\text{token}} \ge \frac{2LNd_{kv}b}{B},$$

where $B$ is achieved bandwidth. At 128k tokens in the configuration above, every token re-reads 40 GiB. On a device with 2 TB/s that is about 21 ms of pure cache traffic, which caps one sequence near 47 tokens per second before weights are touched. At a million tokens the same arithmetic gives roughly 5.8 tokens per second. Speculative decoding spreads one cache read over several accepted tokens, which is why it is popular in long-context serving.

Batching is where the curve turns. Weights are shared across sequences, so a large batch amortises weight reads and short-context serving gets cheaper per token. Caches are not shared. Each sequence contributes its own $2LNd_{kv}b$ bytes, so cache traffic scales with batch size and long-context serving gains far less from batching than people expect.

The curve is a step, not a slope. While the cache fits in device memory the interval is roughly flat in $N$. When it stops fitting and starts living in host memory or on disk, the interval jumps by the ratio between link and device bandwidth, which is a factor and not a percentage. Users do not perceive the average. They perceive the week the sessions got longer and the product became slow.

## What the engineering changes, and what it cannot

Everything useful in long-context systems changes a constant in $2LNd_{kv}b$ or a factor in $4LN^2d$. Nothing moves the exponents.

Compressing the cache is the clearest case. The xKV authors show that the dominant singular vectors of the KV cache are aligned across layers, and use that to factorise grouped layers into a shared low-rank subspace after training. They report up to 8x KV cache compression with accuracy preserved on long-context tasks, and up to a 4.23x end-to-end speedup when combined with selective reconstruction at decode time. At 8x, the 128k case drops from about 21 ms of cache traffic to under 3 ms, and concurrency changes with it. The sequence still needs 5 GiB, and a second sequence needs another 5.

Reducing what attention reads is the other lever, and it changes the effective $N$ in the prefill term. Pruning and sparse attention are how most million-token claims are served. The bill arrives as a selection decision: something must choose which positions attend to which, and it chooses before the model knows what the question needs. The error is quiet, which is worse than a loud one.

There is a training-side version. The long context tuning work on video expands full attention from a single shot to every shot in a scene, then fine-tunes with context-causal attention so autoregressive generation can run on an efficient KV cache. Train with the quadratic form, deploy with the cached form. The model learns what to attend to while attention is dense, and the serving path reads a subset of what it learned to value.

What cannot change is the shape. Exact attention over $N$ tokens touches an $N \times N$ score matrix, and exact caching stores $N$ states. Any method that preserves both pays the exponents. Methods that avoid them are deciding which tokens matter, and that decision is the product.

## A decision rule that survives contact with a bill

Three numbers decide the architecture, and all three are arithmetic.

```python
LAYERS, KV_HEADS, HEAD_DIM, BYTES = 80, 8, 128, 2
BANDWIDTH, PEAK_FLOPS = 2.0e12, 3.1e14

def budget(n_ctx, n_out):
    cache = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES * n_ctx
    prefill_s = 4 * LAYERS * HEAD_DIM * n_ctx**2 / PEAK_FLOPS
    token_s = cache / BANDWIDTH
    return cache, prefill_s, token_s, n_out * token_s
```

Run it for the window you want and the window you need, then read the last value against your latency target. Peak FLOPs is generous, so the prefill figure is a floor.

Then the part arithmetic cannot answer. Measure what the extra tokens buy: run your eval set with the full window, then with the top twenty passages your retriever would have returned, and compare. If the answers agree, the window is decoration and you are buying capacity to impress a benchmark. If they disagree, you have found the cases where the long window is the product, and those are worth paying for. That is the only measurement that separates fitting from using for a specific application, and it costs an afternoon.

Per-token pricing is usually linear in the prompt while prefill is quadratic, and someone absorbs the difference. My opinion is that you should not build an architecture whose margin depends on the seller continuing to absorb it. Design so that a change in the per-token price changes your bill and not your design.

Long context is a capacity product sold as a capability. Some questions cannot be answered from twenty passages, and for those a window is the only tool. But capacity is what you pay for, per token, per user, per second of waiting, and it is worth buying only the part you can demonstrate that you use.

## Sources

- Yangshen Deng et al., [AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference](https://arxiv.org/abs/2504.10326) (2025).
- Chi-Chih Chang et al., [xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](https://arxiv.org/abs/2503.18893) (2025).
- Xiaoran Liu et al., [Thus Spake Long-Context Large Language Model](https://arxiv.org/abs/2502.17129) (2025).
