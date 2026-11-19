---
title: "State Space Models Versus Attention"
description: "A fixed-size state compresses the past while attention stores every token, and the gap between them is an information budget, not a tuning knob."
pubDatetime: "2025-10-04T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - State Space
  - Attention
categories:
  - Math
---

Serving a 70B model at 128k tokens of context leaves you holding about 43 billion bytes of key and value cache, and every new token appends another 320 KiB. Paging moves that memory, quantisation shrinks it, and neither changes its shape: the bill grows with the conversation. A state space model reads the same 128k tokens and keeps 128 KiB.

A smaller cache is the wrong way to describe it. A state is a summary, and a summary cannot be re-read. The engineering question is how much of the past your task must recover exactly, and what you are willing to pay to keep it recoverable, not which architecture wins in the limit.

## A state is a summary you cannot re-read

A continuous state space model is a linear system driven by the input:

$$h'(t) = A h(t) + B x(t), \qquad y(t) = C h(t).$$

Sampled at a step size $\Delta$, it becomes the recurrence every RNN shares:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t, \qquad \bar{A} = \exp(\Delta A).$$

Mamba's change was to make $\Delta$, $B$ and $C$ functions of the current token, so the model can hold the state or overwrite it. The shape stays the same. One vector $h_t$ of size $d_{model} \times d_{state}$ carries everything the model will ever know about the prefix.

```python
def step(h, x_t, A_bar, B_bar, C_t):
    # h: (d_model, d_state), allocated once and reused for every token
    h = A_bar * h + B_bar * x_t
    return C_t @ h          # one output vector per token
```

At $d_{model} = 4096$ and $d_{state} = 16$ in fp16 that state is 131,072 bytes, and it stays 131,072 bytes at a thousand tokens or a million. That is less than the cache cost of a single token at the settings below. The price is structural: $h_t$ is a function of the prefix, not the prefix. Two different 100k-token histories can land on the same state, and once they do, no amount of compute recovers the difference.

## What attention keeps, and what it charges

Attention does not summarise. It stores.

$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V.$$

The middle matrix is a lookup table with one row per query position, and the softmax makes it content-addressed, so a query can put almost all of its mass on one key out of a million. Nothing is compressed and nothing is forgotten. At decode time the table is materialised as the key and value cache:

$$M_{kv} = 2 \, L \, H_{kv} \, d_{head} \, n \, b.$$

For a 70B-class model with $L = 80$, $H_{kv} = 8$, $d_{head} = 128$, $n = 131072$ and $b = 2$ bytes, that product is about $4.3 \times 10^{10}$ bytes. The whole SSM state from the previous section is three parts per million of it.

That cache is also why decoding is not a compute problem. Each token reads the entire cache and performs on the order of $4 L d_{model} n$ multiply-adds against it, which works out to under ten flops per byte moved at these sizes. An H100 SXM balances at roughly three hundred flops per byte of HBM bandwidth in fp16, so decode attention runs an order of magnitude below that line. The cost is bandwidth, not arithmetic. That distinction matters later.

## Why a fixed state cannot be a lookup table

Take the simplest thing you might want from a context: repeat it back. To copy an arbitrary string of $n$ tokens drawn from a vocabulary of size $V$, the output must carry $n \log_2 V$ bits that depend on the input. A state of $b$ bits has at most $2^b$ values, so it can represent at most $2^b$ distinguishable prefixes. Copying needs $V^n$ of them. Therefore

$$b \ge n \log_2 V.$$

Thirty-two thousand tokens over a fifty-thousand token vocabulary needs about $5 \times 10^5$ bits, or 62 KiB, and the requirement grows without bound. A fixed state cannot meet it. That is the result Jelassi et al. prove and then confirm empirically: a two-layer transformer can copy strings of exponential length, a model with a fixed-size latent state cannot, and pretrained transformers retrieve from context far better than state space models do.

You can push on the constant. StateX expands the recurrent state after training and reports better recall without a large increase in parameters, which is consistent with recall tracking state size rather than architecture. You can also organise the state better, with gating or delta-rule writes instead of plain addition. Both buy capacity. Neither changes the exponent.

Interference is the mechanism. A linear recurrence with feature map $\phi$ is an associative memory:

$$S_t = \sum_{i \le t} \phi(k_i) v_i^\top, \qquad \hat{v}_t = \phi(q_t)^\top S_t = \sum_{i \le t} \big(\phi(q_t)^\top \phi(k_i)\big) v_i.$$

If those weights were one-hot, retrieval would be exact. In a linear recurrence they are inner products between the query and every key ever written, and the terms for every $i$ except the target do not vanish. That residue is interference, and it grows as more pairs are superposed into the same $d$ dimensions.

Compression is not the only ceiling. Merrill et al. show that SSMs cannot express computation outside $\mathsf{TC}^0$, so the recurrent form does not buy unbounded state tracking either. They cannot reliably compose permutations, and by extension cannot track a chess game or the entities in a long narrative. The state is a fixed-width summary with a provable ceiling, not a hidden database.

## The hybrid stack is the actual answer

Both ceilings are real, so production systems stopped choosing. Nemotron-H replaces the majority of attention layers with Mamba layers, which run at constant compute and constant memory per generated token, and reports accuracy at or above similarly sized open transformers such as Qwen-2.5-7B/72B and Llama-3.1-8B/70B, with inference up to three times faster. The convolutional multi-hybrid work from the StripedHyena line takes a different cut: several operators, each matched to a different recall task, co-designed with their kernels, reporting 40B-scale training 1.2 to 2.9 times faster than an optimised transformer.

The theory says the boundary is soft. Dao and Gu show that SSMs and attention variants are connected through decompositions of semiseparable matrices, and that the same computation can be materialised either as a scan or as a matrix multiply. Mamba-2 came out of that reframing with a core layer 2 to 8 times faster. If the two are views of one object, a hybrid is a choice about where to sit, not a bet on a species.

What the ratio buys is proportional. The cache scales with the number of attention layers, so eight attention layers in a 64-layer stack is an eighth of the cache, not zero. What it costs is surface area: two kernels, two sets of initialisation and learning-rate sensitivities, and a parallelism strategy that has to handle a scan and a matmul in the same step.

## The standard framing gets the axis wrong

The usual comparison is asymptotic. Attention is quadratic, SSMs are linear, therefore SSMs win at long context. I think that framing misplaces the constraint, and the misplacement pushes teams toward the wrong default.

First, the quadratic term is a prefill and training cost. At decode the per-token cost is linear in $n$ for attention and constant for the SSM, so what you are buying is a slope, and the slope is the cache. "Attention is quadratic" does not explain a slow decode. Converting layers does not help in proportion to $n$ either; it helps in proportion to the fraction you converted.

Second, the memory argument is really an argument for fewer attention layers. If your stack keeps one attention layer in eight, the case for removing the last one is an eighth of the case for removing the first seven. The decision is a number, not a doctrine.

Third, and this is the part I would push back on hardest: the claim that SSMs model state better is the weakest part of the usual story. The expressivity result puts them in the same class as transformers. The recurrent form does not give you a stack or a tape. It gives you a decaying summary, which is better at gist and local texture and worse at exact recall. Calling that superior state modelling confuses compression with memory.

There is a fourth problem, less theoretical. Much of the demand for long context is a demand for retrieval, which is a database question. If the workload is "find the clause in this contract", the fix is an index and a retriever. Buying a million-token window for a search problem is the expensive way to answer it.

None of this makes attention obsolete. It makes the choice a function of one quantity: how many bits of the past must be recoverable exactly, and how often.

## What to measure before you commit

The table is the version of this argument I would put in a design review. The rows are the axes that actually bind, and the last two are the ones teams forget.

| Axis | Full attention | Pure SSM | Hybrid |
|---|---|---|---|
| Training work per layer | $O(n^2 d)$, matmul bound | $O(n d)$, scan bound | between, two kernels |
| Decode memory | $2 L H_{kv} d_{head} n b$ | $d_{model} d_{state} b$ | attention layers only |
| Recall of one token | exact, by lookup | capacity limited | exact inside the attention band |
| Expressivity ceiling | $\mathsf{TC}^0$ | $\mathsf{TC}^0$ | $\mathsf{TC}^0$ |
| Dominant cost | bandwidth | quality on recall | integration |
| Failure mode | out of memory, loud | wrong detail, quiet | depends on the ratio |

My default for a team serving under 32k tokens is attention with a paged and quantised cache, and no architecture change. The hybrid earns its complexity when decode throughput per byte of memory is the binding constraint, which is roughly when context length or concurrency, rather than model quality, sets the GPU count. That is a business fact about your traffic, not a law about architectures.

The measurement I would run first is not a throughput benchmark. It is a recall audit: take the longest context you actually serve, hide facts at known positions, and measure exact recovery with a 4k window, a 32k window, and your candidate stack. If the 4k window passes, the rest of the budget is capacity you are buying for a problem you do not have. If it fails, you have measured the number of bits you need, and the architecture choice follows from the bound rather than from the trend.

## Sources

- Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach. "Repeat After Me: Transformers are Better than State Space Models at Copying." [arXiv:2402.01032](https://arxiv.org/abs/2402.01032), 2024. https://arxiv.org/abs/2402.01032
- Tri Dao, Albert Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." [arXiv:2405.21060](https://arxiv.org/abs/2405.21060), 2024. https://arxiv.org/abs/2405.21060
- William Merrill, Jackson Petty, Ashish Sabharwal. "The Illusion of State in State-Space Models." [arXiv:2404.08819](https://arxiv.org/abs/2404.08819), 2024. https://arxiv.org/abs/2404.08819
- NVIDIA. "Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models." [arXiv:2504.03624](https://arxiv.org/abs/2504.03624), 2025. https://arxiv.org/abs/2504.03624
- Jerome Ku, Eric Nguyen, David W. Romero, et al. "Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale." [arXiv:2503.01868](https://arxiv.org/abs/2503.01868), 2025. https://arxiv.org/abs/2503.01868
- Xingyu Shen, Yingfa Chen, Zhen Leng Thai, Xu Han, Zhiyuan Liu, Maosong Sun. "StateX: Enhancing RNN Recall via Post-training State Expansion." [arXiv:2509.22630](https://arxiv.org/abs/2509.22630), 2025. https://arxiv.org/abs/2509.22630
