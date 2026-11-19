---
title: "Sparse Attention in 2026"
description: "Sparse attention is a bet about which token pairs matter, and the three ways of placing that bet fail differently when the bet is wrong."
pubDatetime: "2026-04-11T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Sparse Attention
  - Long Context
categories:
  - Math
---

A query token sits at position 40,000. The fact it needs is at position 312. Nothing in the position index says so, and the model gets one forward pass to find it.

Dense attention does not have to find anything. It reads all 40,001 keys and lets the softmax put a small weight on the one that matters. Sparse attention must name a subset before it knows the answer. Choose $k = 2048$ of 40,001 and you have already made a claim about where the useful mass is. Every sparse method in production is that claim wearing different machinery.

## The assumption, stated plainly

The dense output for query $i$ over $n$ keys is

$$o_i = \sum_{j=1}^{n} a_{ij} v_j, \qquad a_{ij} = \frac{\exp\left(q_i^\top k_j / \sqrt{d}\right)}{\sum_{m} \exp\left(q_i^\top k_m / \sqrt{d}\right)}.$$

A sparse method picks a mask $M_i$ with $|M_i| = k$ and averages over that subset instead. Whether the truncation was harmless is decided by the mass left outside:

$$p_i^{\mathrm{drop}} = \sum_{j \notin M_i} a_{ij}.$$

Shen and coauthors prove in [arXiv:2511.20102](https://arxiv.org/abs/2511.20102) (2025) that approximation error scales linearly with this quantity, which makes $p^{\mathrm{drop}}$ the object to reason about rather than a proxy for it. The coefficient vector in $\tilde{o}_i - o_i = \sum_j c_j v_j$ sums to zero with $\sum_j |c_j| = 2 p_i^{\mathrm{drop}}$, so for values inside a ball of radius $R$,

$$\left\| \tilde{o}_i - o_i \right\| \le 2 R \, p_i^{\mathrm{drop}}.$$

What hurts is not how many pairs you dropped. It is how much normalised weight they carried. A mask that discards 98 percent of positions carrying 2 percent of the mass is nearly exact. A mask that discards 5 percent carrying 90 percent of the mass is not an approximation of attention at all. The three families below differ in what they know, before the softmax, about where that mass is.

## Fixed patterns pay once and hope

A sliding window assumes locality. A strided pattern assumes periodicity. Both derive the mask from position arithmetic, which costs nothing and needs no data. Grammar is local, so the assumption survives a good deal of text. Coreference is not local, and that is where it breaks.

Qiu and coauthors make the fixed mask linguistic rather than positional in [arXiv:2601.02819](https://arxiv.org/abs/2601.02819) (2026), treating punctuation as semantic boundary anchors and fusing a global summary with boundary features over two branches. For a 0.6B model on 32k inputs they report 10.8 percent less information loss at 97.3 percent sparsity.

Luo and coauthors make the budget empirical instead in [arXiv:2603.18636](https://arxiv.org/abs/2603.18636) (2026). They argue that attention sparsity is largely intrinsic to a layer, varying only slightly across inputs, so per-layer sensitivity can be profiled offline and applied at inference. Across seven video generation models they report up to 1.93x speedup at up to 29 dB PSNR on Wan2.1.

That is a claim about a distribution, not a token. It says layer 9 can afford 3 percent density whatever you feed it, not that any query at position 40,000 finds its fact inside that 3 percent. Right on average, silent on the tail.

## Learned patterns buy information and pay for it

A learned mask replaces position arithmetic with a scoring function. That is strictly more information, purchased with a second model that can itself be wrong.

Shen and Shen use a lightning indexer with sigmoid activations for bounded selection scores, plus an adaptive controller that sets the number of attended tokens from local uncertainty, in [arXiv:2601.15305](https://arxiv.org/abs/2601.15305) (2026). At 1.7B parameters and 400B training tokens they report 12 to 16x speedup at 128K context, perplexity improving from 6.03 to 5.70, RULER at 128K nearly doubling, and loss spikes down 98 percent. The controller matters most: it makes sparsity a function of the query rather than a constant.

Zhang and coauthors show in [arXiv:2602.13515](https://arxiv.org/abs/2602.13515) (2026) that the standard rules fail predictably. Top-k and top-p each break under specific conditions and the failures compound at high sparsity, which motivates a hybrid rule plus a distillation objective. They report 95 percent attention sparsity and 16.2x attention speedup with quality held.

Learning the mask costs more than compute. The SSA authors name two gaps. The attention gap is a distribution mismatch: a model trained dense and served sparse is answering a question it was never asked. The capability gap is worse, because a model trained only sparse lacks complete gradient flow and cannot match full attention even in principle. Their fix keeps both paths and aligns the sparse output to the full output in feature space, which is an admission that the mask cannot be trusted alone.

Yüksel and coauthors show the pattern is discovered rather than configured in [arXiv:2602.19143](https://arxiv.org/abs/2602.19143) (2026). In a high-order Markov setting, patterns emerge in discrete stages, competitive early with every head chasing the dominant one and cooperative later as heads specialise, and the number of stages completed depends on the size of the training set.

## Hashing asks a cheaper question

Hashing estimates poorly but cheaply. Instead of computing a score accurate enough to rank keys, it maps queries and keys to binary codes and recovers the relative order. Gong and coauthors report up to 7.2x speedup over full attention with accuracy held in [arXiv:2506.02572](https://arxiv.org/abs/2506.02572) (2025).

HATA exploits a narrower requirement: top-k needs a ranking, not a value. An estimator with large error but stable ordering is useless for everything except top-k, and top-k is the only thing being asked. It gives up the absolute score that other methods pay heavily to compute and keeps only the order.

The assumption is that locality in code space reflects relevance, so a key colliding with a query bucket is a candidate. But relevance is $q^\top k$, which is not a metric and obeys no triangle inequality. Two keys can be strongly relevant to a query while sitting far from it and from each other.

The failure is asymmetric. A collision is a false positive that costs arithmetic and nothing else, because the softmax will down-weight the irrelevant key once it is inside the mask. A missed bucket is a false negative that costs accuracy permanently, and its frequency depends on bucket occupancy rather than on the data. That is why hashed masks get paired with a retained window.

## Matching dense attention is the wrong target

Sparse results are usually reported as distance from dense attention: perplexity gap, PSNR against the dense render, points lost on RULER. The framing treats dense attention as ground truth.

At long context that is often false. Dense attention over 128K spreads probability across positions that carry nothing, and the softmax parks a large share of it on the first token. Shen and Shen report first-token attention, used as a proxy for attention sinks, falling from 47 percent to under 4 percent. If dense attention was routing nearly half its mass to a position with no content, the reference was miscalibrated.

Gao and coauthors sharpen the point in [arXiv:2602.03560](https://arxiv.org/abs/2602.03560) (2026). They interleave each full layer with several sparse layers and derive the sparse layers' token selection and KV caches directly from the preceding full layer, using it as an oracle instead of training a proxy. In an 80B mixture-of-experts model with 49 layers, only 5 use full attention, and the result beats both full attention and hybrid sliding-window baselines while cutting KV cache by nearly 10x.

A sparse model outperforming the dense model it supposedly approximates. My reading, and this is opinion, is that the quantity that mattered was never $\|\tilde{o} - o\|$ but whether the tokens carrying the answer survived the mask. Dense attention is a weak optimiser of that, because it has no incentive to concentrate mass. Desai and coauthors report a compatible finding in [arXiv:2510.05688](https://arxiv.org/abs/2510.05688) (2025): full model quality at up to 20x sparsity, and full quality on AIME2024 at 10x sparsity.

None of this removes the burden of proof. It relocates it: you no longer justify being close to dense, only that the mask preserves what the task needs.

## The cost of being wrong is in the tail

A mean over queries and heads turns discarded mass into a small perturbation. But $p^{\mathrm{drop}}$ has a thick tail, because the queries that matter most are the ones whose answer sits in one specific distant position.

Desai and coauthors put the diagnosis directly: prior approximate methods fail to give consistent approximations across heads and query vectors, and lack any guarantee on quality. Their response is to treat the mask as a contract rather than a guess, with user-specified $(\epsilon, \delta)$ guarantees. That is the only framing here that survives a deployment review, because it converts an assumption into something testable.

| Family | What it assumes | Mask from | Failure mode | Cheapest check |
|---|---|---|---|---|
| Fixed | Locality and periodicity | Position arithmetic, or an offline per-layer profile | A needed token outside the window | Widen the window on flagged layers |
| Learned | A cheap scorer tracks relevance | A trained indexer or adaptive controller | Proxy error at high sparsity, train-inference mismatch | Align to a retained dense path |
| Hashed | Code-space proximity implies relevance | Binary codes of $q$ and $k$ | Bucket misses, occupancy imbalance | Probe outside the selected buckets |

IndexCache, from Bai and coauthors in [arXiv:2603.12201](https://arxiv.org/abs/2603.12201) (2026), is a good example of checking rather than assuming. Top-k selections are highly similar across consecutive layers, so most layers can reuse the nearest full layer's indices. On a 30B model they remove 75 percent of indexer computations with negligible quality loss, for up to 1.82x prefill and 1.48x decode speedup. The layer assignment comes from a greedy search that minimises language modelling loss on a calibration set. Measured, not asserted.

## Pay for the check, not the guess

If error scales with dropped mass, and dropped mass is estimable from a sample, the check is cheap enough to run. The union of top-k and a uniform probe works because the two fail in opposite regimes: top-k is accurate when a few keys dominate, sampling is accurate when the distribution is flat, and attention is usually somewhere between.

```python
import numpy as np

def dropped_mass(scores, mask, probe=64, rng=None):
    """Estimate attention mass outside `mask` from a uniform probe."""
    rng = rng or np.random.default_rng(0)
    n = scores.shape[-1]
    w = np.exp(scores - scores.max())
    kept = w[mask].sum()
    idx = rng.integers(0, n, size=probe)
    hit = np.isin(idx, mask)
    # Each probed key stands for n / probe keys.
    missed = (w[idx] * (~hit)).sum() * (n / probe)
    return missed / (kept + missed + 1e-9)
```

This does not make the mask correct. It makes the error observable per query, which is the difference between a system that degrades and one that tells you it is degrading.

The sequence I would argue for, and this is the opinion part, has three steps. Write the assumption down as a claim about $p^{\mathrm{drop}}$. Measure how often it holds on the distribution you actually serve, not the calibration set you happened to have. Spend a fraction of the saved compute on that measurement.

The 2026 work has largely stopped arguing about which pattern wins. Fixed patterns borrowed linguistic structure. Learned ones got honest about the two gaps and kept a dense reference. Hashed ones got cheap enough to be worth their collisions. What stays open is what a sparse result means.

## Sources

- Alfred Shen and Aaron Shen, [arXiv:2601.15305](https://arxiv.org/abs/2601.15305) (2026). Gated Sparse Attention.
- Oğuz Kaan Yüksel, [arXiv:2602.19143](https://arxiv.org/abs/2602.19143) (2026). Incremental Learning of Sparse Attention Patterns in Transformers.
- Jintao Zhang, [arXiv:2602.13515](https://arxiv.org/abs/2602.13515) (2026). SpargeAttention2: Trainable Sparse Attention.
- Ping Gong, [arXiv:2506.02572](https://arxiv.org/abs/2506.02572) (2025). HATA: Hash-Aware Top-k Attention.
- Aditya Desai, [arXiv:2510.05688](https://arxiv.org/abs/2510.05688) (2025). vAttention: Verified Sparse Attention.
- Zhenyi Shen, [arXiv:2511.20102](https://arxiv.org/abs/2511.20102) (2025). SSA: Sparse Sparse Attention.
- Jiayi Luo, [arXiv:2603.18636](https://arxiv.org/abs/2603.18636) (2026). Attention Sparsity is Input-Stable.
- Yizhao Gao, [arXiv:2602.03560](https://arxiv.org/abs/2602.03560) (2026). HySparse: A Hybrid Sparse Attention Architecture.
- Yushi Bai, [arXiv:2603.12201](https://arxiv.org/abs/2603.12201) (2026). IndexCache: Cross-Layer Index Reuse.
- Junxiang Qiu, [arXiv:2601.02819](https://arxiv.org/abs/2601.02819) (2026). Punctuation-aware Hybrid Trainable Sparse Attention.