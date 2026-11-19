---
title: "The KV Cache Is a Memory Budget"
description: "The KV cache costs a fixed number of bytes per token per layer, and that rate decides how much context and concurrency you can sell."
pubDatetime: "2024-12-07T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - KV Cache
  - Inference
categories:
  - Math
---

Two 80 GB accelerators. A 70B model in fp16. A customer who wants 128K context. The weights take about 140 GB, leaving roughly 20 GB for everything that is not weights. The cache for a single 128K request on this model is 40 GiB. The arithmetic does not close, and batching does not rescue it, because the cache term multiplies with batch while the weight term does not.

The order usually runs backwards. The context length comes from a product spec, the model from an evaluation table, and the queueing behaviour is discovered at load time, when the decisions that mattered are already fixed.

The cache is usually introduced as an optimization. That framing is correct for one sequence and useless for a fleet. At serving time it is a budget with a rate and a quantity: bytes per token, set by the architecture, times tokens and sequences in flight, set by your scheduler. Everything else is accounting.

## The arithmetic fits on a napkin

For a decoder with $L$ layers, $n_{kv}$ key/value heads, head dimension $d$ and $s$ bytes per element, the cost is

$$\text{bytes per token} = 2 \cdot L \cdot n_{kv} \cdot d \cdot s$$

The 2 is the key tensor and the value tensor. Nothing else belongs in the product. No vocabulary term, no query head term, nothing from training. It is a property of the shape, and you can compute it before you download the weights.

| Model | Layers | KV heads | Per token | 128K context |
| --- | --- | --- | --- | --- |
| Llama-2 7B | 32 | 32 | 512 KiB | 64 GiB |
| Llama-2 13B | 40 | 40 | 800 KiB | 100 GiB |
| Llama-2 70B | 80 | 8 | 320 KiB | 40 GiB |
| Llama-3 8B | 32 | 8 | 128 KiB | 16 GiB |
| Mistral 7B | 32 | 8 | 128 KiB | 16 GiB |

The 7B row is the one that surprises people. At 512 KiB per token, an 8K conversation holds 4 GiB of cache, more than a quarter of the 14 GB the fp16 weights occupy. The 13B row costs 2.5 times the 70B row per token, because the 70B model uses grouped-query attention and the 13B model does not.

```python
def bytes_per_token(layers, kv_heads, head_dim, elem_bytes=2):
    return 2 * layers * kv_heads * head_dim * elem_bytes

print(bytes_per_token(32, 32, 128) / 1024)  # 512.0 KiB, Llama-2 7B
print(bytes_per_token(80, 8, 128) / 1024)   # 320.0 KiB, Llama-2 70B
```

The total is bytes per token times tokens per sequence times sequences in flight. Weights and activations compete for the same memory, so the leftover after both is the only number the cache can spend.

## Decode reads the whole thing every step

Prefill is compute bound. Decode is not. In decode, each step reads the entire cache for every sequence in flight, plus the weights once. At 32K context on the 7B multi-head model that is 16 GiB of cache and about 14 GB of weights to produce one token per sequence.

At a nominal 2 TB/s on an A100 80GB, the bandwidth floor is roughly 15 ms per token, or about 65 tokens per second, before compute is counted. Real kernels do worse. The floor is the honest ceiling.

The asymmetry that matters is that weight reads do not care how many sequences run while cache reads multiply. So the cache share of bandwidth grows with batch, and at some batch size the step is a cache-read loop.

Capacity usually binds first, and the failure presents as an out-of-memory error at admission rather than a slow loop.

## Grouped-query attention moves the constant

Grouped-query attention is the reason modern models are cheaper per token than their layer counts suggest. With $n_h$ query heads and $g$ query heads per key/value head,

$$n_{kv} = \frac{n_h}{g}$$

Llama-2 70B has 64 query heads and 8 KV heads, so $g = 8$. Full multi-head attention at that shape would cost 2.5 MiB per token. It costs 320 KiB. The shape of the curve is unchanged: grouped-query attention divides the constant, it does not touch the linear growth in $T$.

For 8 KV heads, $d=128$ and fp16, one layer is 4 KiB per token. Multiply by depth. That is small enough to hold in your head.

The constant is decided in training. At serving time you can change the dtype, keep fewer entries, or re-derive what you dropped. You cannot make the model narrower than it was built.

## Paging is virtual memory with one broken assumption

Paging is the most useful analogy here, and worth stating precisely. Logical token positions map through a block table to physical blocks in a pool. A fixed block size removes the contiguity requirement and the need to pre-size to a maximum. Waste is bounded by one partial block per sequence. Because blocks are referenced by pointer, requests sharing a prefix can point at the same physical blocks, with copy-on-write once they diverge.

That is a real win: the gain comes from removing fragmentation and enabling sharing. Bytes per token are unchanged and the linear term survives.

What breaks is the page fault. An operating system fault reloads identical bytes at microsecond cost, and the program cannot tell the difference. A cache miss has neither property. The bytes are gone, and the only way back is a forward pass over the prefix, whose cost scales with what you dropped. You cannot hide that at memory latency, only overlap it.

Overlap is now its own problem. Jiang and colleagues send a partial set of activations from the CPU, let the GPU begin recomputing KV values while the remaining cache transfers over PCIe, and use a profiler and scheduler to pick the split. They report up to 35.8 percent lower decode latency and 46.2 percent higher throughput against their baselines. That is their measurement on their setup. The structural point is the interesting one: once eviction is on the table, the work moves from allocation to joint scheduling across compute, I/O and memory.

## The standard framing is wrong about what is scarce

The common story is that long context breaks memory, so the fix is compression. That story is incomplete in a way that leads teams to buy the wrong thing.

At the context lengths most requests actually need, the binding term is usually concurrency, not length. The budget is $B \cdot T \le M / (\text{bytes per token})$, and $B$ is nearly always the cheaper factor to move. A team that compresses by 40 percent and keeps the same batch buys headroom it may never need. A team that doubles the batch on the same bytes serves twice the traffic. That is the same inequality approached from opposite ends, and only one of those is a serving change.

The second problem is that the framing treats memory as a single constraint. There are two regimes with opposite optima. At batch 1 you are bandwidth bound and the cache is a read cost, so reading less helps. At high batch you are capacity bound and the cache is a resident cost, so what helps is bytes resident, and a policy that reads less while storing the same amount buys nothing. A benchmark at batch 1 and a deployment at batch 64 are measuring different systems, and a capacity plan built on the first number tends to fail on the second.

My opinion, not a result: treat compression as the last lever, after admission control, after paging, after prefix sharing, and after deciding which context length you are willing to defend. Hypothesis, not observation: compression gets attention first because it is a model-shaped problem with papers, while admission control is a product-shaped problem that nobody writes papers about.

## What eviction policies assume about attention

Attention at step $t$ is a normalized weighted average over whatever is still in the cache:

$$o_t = \frac{\sum_{i \le t} \exp(q_t \cdot k_i / \sqrt{d})\, v_i}{\sum_{i \le t} \exp(q_t \cdot k_i / \sqrt{d})}$$

Drop an entry and its numerator term and its denominator mass go together. The output renormalizes over the survivors. That is why dropping the smallest weights does the least damage, and why heavy-hitter policies work at all.

The assumption underneath is that importance can be estimated and then kept. It cannot, strictly. The weight depends on $q_t$, which changes every step. A token irrelevant to this query may be central to the next one. Every eviction policy is a bet about the future distribution of queries, and the bet is about your workload, not about the model.

Two bets recur. Positional priors keep the first few tokens because they accumulate attention across many queries regardless of content. That is the streaming behaviour widely reported in 2023. Similarity priors keep keys that resemble the current query. The first hedges against an unknown future, the second assumes the future resembles the present. Neither is free, and the failure modes differ by task. Retrieval over many candidate passages has diffuse evidence, so heavy hitters can be the wrong set. Copy-heavy tasks have concentrated evidence, and the same policy becomes nearly harmless. Evaluate on your task, not on perplexity.

Eviction deletes terms, quantization perturbs them, and the second preserves the denominator. CSKV takes a third route and shrinks the width instead. The authors analyze the singular value distribution of the cache, find redundancy along the channel dimension, store low-rank features for the key and value projections, and keep a bi-branch cache with a full-precision window beside the compressed portion. They train with a layer-wise reconstruction loss rather than retraining the model, and report an 80 percent reduction in cache memory while preserving long-context capability, rising to 95 percent when combined with quantization. Those are their reported evaluations, not a general guarantee.

The distinction worth keeping is that shrinking channels reduces bytes per token without removing any token, so the denominator keeps all of its terms. Eviction removes terms. If your task depends on many weak, jointly relevant pieces of evidence, eviction is the error that will hurt, and a short-context benchmark will not show it.

## A budget, not a cache

The practical version of all this is a page of arithmetic, not a library choice. Write down bytes per token. Multiply by the context you will defend and the concurrency you intend to sell. Compare with the memory left after weights and activations. If it does not fit, you have three dials, and you should know which one you are turning before you edit a config file.

Then measure the failure mode you actually get. Capacity failures are loud. Scheduling failures are quiet: requests queue, tail latency climbs, and the cache sits half empty because the allocator could not find a contiguous run. Quality failures are quieter still, because a policy that drops the right-looking evidence produces a fluent wrong answer.

The cache is an exact algebraic object. Any policy that changes it changes the output, and no allocator makes that free. Treat the rate as fixed, the product as the thing you are selling, and every byte you decline to store as a promise about what the next query will not need.

## Sources

Wang, Luning, Shiyao Li, Xuefei Ning, Zhihang Yuan, Shengen Yan, Guohao Dai, and Yu Wang. "CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios." [arXiv:2409.10593](https://arxiv.org/abs/2409.10593), 2024. https://arxiv.org/abs/2409.10593

Jiang, Chaoyi, Lei Gao, Hossein Entezari Zarch, and Murali Annavaram. "Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation." [arXiv:2411.17089](https://arxiv.org/abs/2411.17089), 2024. https://arxiv.org/abs/2411.17089
