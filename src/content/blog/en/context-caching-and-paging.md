---
title: "Context Caching and Paging"
description: "Prefix caching is a bet that tokens return before eviction, and the arithmetic of that bet should price your block size and your eviction policy."
pubDatetime: "2026-05-02T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - KV Cache
  - Serving
categories:
  - Math
---

Three thousand conversations sit open on one replica, each with its own twelve thousand token history. On an 8B model with grouped-query attention the cache costs 128 KiB per token, so one history is 1.5 GiB and the set is 4.4 TiB. A 48 GB card, minus 16 GB of fp16 weights, holds about twenty of them.

```python
KV_BYTES = 128 * 1024                  # bytes per token, 8B GQA
PER_HISTORY = KV_BYTES * 12288         # exactly 1.5 GiB
FREE = (48 - 16) * 10**9               # card minus fp16 weights
print(FREE / PER_HISTORY)              # 19.9
```

Everything past the twentieth is recomputed every turn, or it waits, or it is refused. Nobody who picked the twelve thousand token window wrote that number down. The window was a product decision, the cache an implementation detail.

A prefix cache does not delete work; it relocates work from the prefill of a request to the memory of a machine, under a guess about the future. The guess has a price, and the price can be written down before you build anything.

## A cache entry is a lease

A prefix of $n$ tokens costs $\lambda$ bytes per token. The entry occupies $\lambda n$ bytes from creation until eviction at time $T$. If $R$ later requests reuse it and a hit avoids a prefill of cost $C_p(n)$, the entry is worth

$$V = R \cdot C_p(n) - \rho \lambda n T$$

Here $\rho$ is the shadow price of one byte of cache memory held for one second. Nobody puts it in the design doc, and it is the only term that makes the decision honest. On a replica that is already batching, $\rho$ is set by the sequences you displaced, which is why a cache that is an obvious win on a quiet machine can be a loss on a busy one. The numerator does not change. The price does.

Setting $V = 0$ gives the break-even reuse count,

$$R^\star = \frac{\rho \lambda n T}{C_p(n)}$$

Longer prefixes raise both terms roughly linearly, so the ratio stays near flat until attention turns quadratic. Residency sits in the numerator, and eviction controls residency. A higher $\rho$ raises the bar for everything: caching and admission control are not separable problems.

The mechanism that makes any of this shareable is paging. The cache is cut into fixed size blocks, and a per-sequence table maps logical index to physical block. Two sequences with the same prefix point at the same blocks. On divergence the block is copied and the original stays shared. The key is a hash of the block's tokens combined with the hash of everything before it, so identity is content rather than ownership.

That property is the hinge. Once blocks are content addressed, an entry is a shared object, and its hit rate is a property of the whole population.

## The arithmetic of hit rate at fixed memory

Hold memory $M$ fixed with $N$ distinct prefixes of equal length. Each entry costs $\lambda n$ bytes, so at most $K = M/(\lambda n)$ fit. Under independent uniform reference the hit rate is at most $K/N$, and every replacement policy achieves exactly that, which is the useful part: when reuse is uniform, policy work is wasted work.

Substituting the capacity bound gives the ceiling on saving per request, with $C_p(n) = a n + b n^2$:

$$h \le \frac{M}{\lambda n N}, \qquad S = h \, C_p(n) \le \frac{M}{\lambda N}\left(a + b n\right)$$

At fixed memory, quadrupling prefix length quadruples the saving per hit and divides the hit rate by four. The product is flat in $n$ except for the attention term $b n$, so long prefixes earn their keep only through the quadratic part.

One case breaks the model. If $N$ collapses because the prefix is identical for every user, then $h \to 1$ regardless of capacity and none of this applies. That is the case that made prefix caching visible and the least interesting one economically, because it needs no policy. The hard regime is thousands of near-identical histories whose shared region is long but not exactly equal, because there the hit rate measures traffic rather than code.

## What a hit costs

A hit-rate dashboard prices three costs at zero, starting with memory you cannot sell: the whole $\rho$ term, invisible if you only count tokens per second.

The second is trust. Shared blocks exist as a single physical copy, and nothing validates them on the read path. Yamamoto and Matsuura inject faults into that surface and report that 13 of 16 bit positions in a BF16 value produce coherent but altered output, that only requests sharing the prefix are affected, and that the damage does not decay with time. A corruption is served to everyone who hits it. Residency sets a blast radius as well as a bill. Their countermeasure is a checksum at scheduling time, which bounds the damage to one batch.

The third is secrecy. A hit is faster than a miss and that difference is observable. Pennas and coauthors reconstruct another tenant's request from hit and miss patterns alone, then defend by watching cross-user reuse and isolating only the suspicious prefixes, reporting up to 70 percent higher reuse and 30 percent lower latency than switching sharing off. A shared prefix cache is a multi-tenant resource and should be governed like one, with quotas, integrity checks and an explicit answer to what may be shared.

My view is that integrity checking belongs on by default, since it runs at scheduling granularity rather than per token. The paper reports the countermeasure as cheap. I have not reproduced it.

## Eviction is a statement about your workload

Every policy names an assumption about inter-arrival time. Writing it beside the policy turns a configuration argument into a falsifiable claim.

| Policy | The assumption it encodes | Where it breaks |
| --- | --- | --- |
| LRU | the next use arrives soon after the last one | one tenant's burst refreshes its own history and evicts everyone else's |
| FIFO | age is what makes an entry stale | a shared system prompt still in use after every other entry turned over |
| LFU | popularity is stable across the window | the traffic mix shifts and the new prefix never gets admitted |
| Size aware | big entries are the expensive ones | the big entry is the one everybody shares |
| Belady | the future is known | offline only, but it is the ceiling you measure against |

LRU's failure mode on agent traffic falls out of its own definition. An agent that runs twenty tool calls touches its entire history on each call, so every node is refreshed together. Between bursts it touches nothing, so the whole prefix falls to the bottom of the stack at once. Eviction concentrates on the agents that are thinking, and when one returns its context is gone. You see it in the tail, because the longest silence leaves the largest gap to close.

The optimal offline policy evicts whatever is needed furthest in the future. Belady is not implementable, but the gap between your policy and Belady on your own trace prices prediction.

An eviction policy is a statement about who you are willing to make wait. Recency says the recently active, frequency says the popular, round-robin tail eviction says everyone a little. That is a product decision wearing a systems costume.

## Paging is where the architecture leaks

Block size is the knob everybody sets once and never revisits. Small blocks cost metadata: more table entries, more hashes per token, more scheduling work. Large blocks waste bytes where two sequences diverge, because a partially shared block is copied whole. For a sequence that diverges inside a block of $B$ tokens the expected waste is about $B/2$ tokens.

$$\text{overhead}(B) \approx c_h \frac{n}{B} + \lambda \frac{B}{2} S$$

Minimising gives $B^\star = \sqrt{2 c_h n / (\lambda S)}$. The square root is the useful part. Being a factor of four off costs about a factor of two in overhead, so the choice is forgiving. What deserves attention is that $B^\star$ moves with $S$, the number of concurrently diverging sequences, so a block size tuned for many short diverging turns is wrong for few long ones.

The deeper leak is that paging assumes the reusable state is a prefix of tokens. On hybrid and recurrent models it is not. A recurrent layer can resume from a stored state rather than from the token history, so the reusable object is a checkpoint at a position, not a block at an offset, and checkpoints do not compose the way blocks do. Shirokikh and Nikolenko formalise this as checkpoint placement under a distribution over overlap depths, give an exact dynamic program, and report that distribution-aware placement dominates every fixed budget baseline on the frontier while usually needing fewer checkpoints than block caching.

## The contrarian case against cache affinity

When hit rate disappoints, the standard move is affinity routing: send each request to the replica that already holds its prefix. The hit rate goes up. I think this is usually the wrong dial.

Affinity converts a load balancing problem into a locality problem without removing the load. Requests for a popular prefix all land on one replica, so it becomes the longest queue and its memory fills with entries the router keeps sending it. Mean time to first token can improve while p99 degrades, and the SLO is written against p99. The trade pays only when the prefill you avoid is large compared with the queueing you create, which means very long prefixes and moderate load. Below that line a least-loaded router in front of a smaller cache beats an affinity router in front of a bigger one, because load imbalance is first order and locality is second.

The evidence is a mechanism and my opinion, not a measurement. The sign could flip on a workload with a few enormous prefixes and gentle arrival rates, which is the regime affinity was designed for.

The alternative I prefer is to make reuse structural. If the shared region of a prompt has an identity, a document id or a tenant id plus a version, then any replica can rebuild the same prefix from the same id and affinity stops being necessary. Structural reuse also survives the hot prefix, because an id can be replicated and a physical block cannot.

## What to write down

Four numbers, all cheap. Bytes per token for the model you deploy. The distribution of prefix lengths, not the mean. The distribution of inter-arrival times per prefix, because every policy assumes something about it. And the recompute cost you are avoiding, in the same units as the latency budget.

The habit I would push back on is treating the cache as a free optimisation. It is a purchase. You spend memory you could have sold as concurrency, you take on an integrity surface, you take on a timing channel, and you accept a residency window during which a corrupted entry keeps being served. All of that can be worth it. None of it is free, and the bill arrives as a tail latency number that nobody attributes to the cache, because the cache was supposed to help.

## Sources

Pennas, Panagiotis Georgios, Konstantinos Papaioannou, Marco Guarnieri, and Thaleia Dimitra Doudali. "CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-Tenant LLM Serving Systems." [arXiv:2603.10726](https://arxiv.org/abs/2603.10726), 2026. https://arxiv.org/abs/2603.10726

Yamamoto, Yuji, and Satoshi Matsuura. "Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems." [arXiv:2604.17249](https://arxiv.org/abs/2604.17249), 2026. https://arxiv.org/abs/2604.17249

Shirokikh, Mikhail, and Sergey Nikolenko. "Sparse Prefix Caching for Hybrid and Recurrent LLM Serving." [arXiv:2605.05219](https://arxiv.org/abs/2605.05219), 2026. https://arxiv.org/abs/2605.05219
