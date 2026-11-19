---
title: "Positional Encodings: The Model Has No Clock"
description: "Attention is order-blind by construction. Positional encodings are the only reason a transformer can tell a subject from an object."
pubDatetime: "2023-08-06T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Transformers
  - Positional Encoding
  - Attention
categories:
  - AI
---

The first time I explained attention to someone, they asked where the order comes from. I had a slide of attention weights and token embeddings, with no mention of positions. The honest answer is that it comes from nowhere. A transformer built as described in *Attention Is All You Need* has no notion of order until you put one in by hand.

That is not a footnote. Remove positional information and a sentence becomes a bag of words. Each token's output is a weighted sum over all tokens, and swapping two tokens swaps the weights that referenced them. "Dog bites man" and "man bites dog" yield the same set of token representations. If the task depends on who did what to whom, the model cannot know.

## Attention has no ordering operator

The symmetry is exact. Self-attention is permutation-equivariant: permute the input sequence and the output is permuted the same way. Scaling by $\sqrt{d_k}$, softmax normalization, the residual connection, the layer norm — none break it, because they act along the feature dimension or pointwise across positions.

Formally, for a permutation matrix $P$ acting on the sequence axis,

$$
\mathrm{Attention}(PX) = P\,\mathrm{Attention}(X)
$$

with one caveat: masking makes the statement directional. A causal mask is tied to positions, so a permuted input under it is not a meaningful reordering of the original computation. The mask is the only structure attention gets for free. Order otherwise has to enter as data: we add to each token embedding a vector that depends on where the token sits.

## Sinusoidal encodings: a unique pattern per position

The original construction uses fixed, non-learned functions:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)
, \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Position $pos$ gets a $d$-dimensional vector of sines and cosines at geometrically spaced frequencies, fast in the first dimensions and slow in the last. Each dimension is a wave in position; the vector is that position's phase across a bank of oscillators. Distinct positions get distinct patterns over any practical range, and since the frequencies span several orders of magnitude, the slow dimensions separate distant regions while the fast ones separate neighbours.

The reason for sine and cosine specifically is that relative offsets become linear maps. For fixed shift $k$, the encoding at $pos+k$ is a rotation of the encoding at $pos$, applied independently to each frequency pair:

$$
\begin{bmatrix} \sin(\omega(pos+k)) \\ \cos(\omega(pos+k)) \end{bmatrix}
=
\begin{bmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{bmatrix}
\begin{bmatrix} \sin(\omega pos) \\ \cos(\omega pos) \end{bmatrix}
$$

A matrix that depends only on $k$, not on $pos$. The paper's argument is that this makes it easy to learn to attend by relative position, since the transformation between "here" and "three tokens back" is the same everywhere. Whether networks exploit that cleanly is unclear to me. The construction at least makes relative position a simple function of the representation.

These vectors are added to token embeddings, not concatenated, because addition is cheap and keeps the width fixed, at the cost of mixing positional and semantic information in the same coordinates. The sinusoidal version has no parameters to train and no table-imposed length limit.

## Learned absolute embeddings and the table

The obvious alternative is to learn one vector per position and add it, as you learn a word embedding. The original Tensor2Tensor implementation did this, and most from-scratch implementations still do, because it is three lines of code.

The cost is a table of size $L_{\max} \times d$. Position 4000 has a vector only if the model trained on sequences that long, so maximum length is a constant fixed before training. You can initialize extra rows and hope, but they receive no gradient and stay near their initialization. You can resize the table and fine-tune, but that is a separate training run, not a property of the architecture.

There is a subtler cost. A learned table has no reason to place position 500 near position 501 in any meaningful sense. Each row is an independent parameter, and the only pressure toward smoothness is whatever the data supplies. With enough data the model can learn a usable geometry, and learned and sinusoidal embeddings perform comparably on translation benchmarks; the original paper reports roughly equal results. But the table is a lookup, not a function, and that shows when you leave the trained range.

## Rotary encodings: rotate the vectors, not the embeddings

Rotary position embedding, from the RoFormer work, takes a different route. Instead of adding a positional vector, it rotates query and key vectors by an angle proportional to position. The token representation is untouched; position enters through the geometry of the attention score.

For a coordinate pair $(x_{2i}, x_{2i+1})$ at position $pos$, apply a rotation by $\theta_i \cdot pos$, where $\theta_i$ is a base frequency for that pair. In complex notation: write the pair as a complex number and multiply by $e^{i\theta_i pos}$. Scores are then computed from the rotated vectors.

The payoff is what happens to the dot product. If query $q$ at position $m$ and key $k$ at position $n$ are each rotated by their own positions, the score depends on $q$, $k$, and the difference $m-n$:

$$
\langle R_{\theta m}\, q,\; R_{\theta n}\, k \rangle
= \langle q,\; R_{\theta(n-m)}\, k \rangle
$$

The absolute positions drop out and only the offset survives. That is the property we wanted from sinusoidal encodings, obtained by construction rather than by hoping the network learns to use a linear map. And because the rotation happens inside attention rather than to the input embedding, the semantic vector is never mixed with a positional one in the residual stream.

A few details. Rotary applies to queries and keys, not values, since values are aggregated and carry content rather than address. It composes per head, so heads can operate at different effective frequency scales. It costs a fixed number of extra operations and no parameters if the frequencies are fixed. The implementation is a handful of lines.

```python
import torch

def rotate_half(x):                      # (..., d) -> interleaved pairs rotated
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary(q, k, base=10000.0):
    dim = q.shape[-1]
    inv_freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = torch.outer(torch.arange(q.shape[-2], dtype=torch.float32), inv_freq)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

That is the mechanism in full.

## Why relative position travels better

The appeal of relative position is not aesthetic. The quantity the model conditions on, the distance between two tokens, is the same wherever the pair occurs. A pattern learned at offset three is a pattern at offset three everywhere. Absolute encodings hand the model position-specific vectors, so a head that learns "attend to the token immediately before me" must learn that rule separately for early, middle, and late positions, or learn something general enough to transfer.

This matters most when the test distribution differs from training. Train on short sequences and evaluate on longer ones, and absolute tables run out of rows while sinusoidal patterns enter index ranges the model never saw. Relative schemes keep producing the same kinds of offsets, which is why they are the default where length varies.

But relative position has its own assumptions. Rotary with fixed base frequencies produces angles that grow without bound. At some length the fastest dimensions alias: a rotation by $\theta$ and one by $\theta + 2\pi$ are indistinguishable, so in those dimensions the model can no longer tell positions apart at a distance of one full period. The frequencies were chosen for a range of lengths, and past it the encoding is ambiguous.

| Scheme | Where position enters | Parameters | Length behaviour |
| --- | --- | --- | --- |
| Sinusoidal absolute | Added to token embedding | None | Defined for any index; patterns unseen in training |
| Learned absolute | Added to token embedding | $L_{\max} \times d$ | Hard table limit |
| Rotary | Rotates $q$ and $k$ | None if base fixed | Offset preserved; high frequencies alias |

## The contrarian part: two claims that need correcting

The first claim I hear often: "the model learns position from the tokens anyway." It is offered as a reason positional encodings are optional. That is wrong in a precise sense. Self-attention has no ordering operator. No parameter, nonlinearity, or normalization in the block can distinguish position $i$ from position $j$ given identical token content. A model can use lexical and syntactic cues as proxies for order — function words, morphology, punctuation — but those are correlations in the data, not a mechanism. Give it a sequence where the same token appears twice and the proxy is silent; the occurrences are indistinguishable to attention. Any claim that order is recoverable from content alone is a claim about the statistics of natural language, not about the architecture.

The second claim is the one I see most: rotary encodings "extend context for free," as if the mechanism alone buys longer sequences. It does not. What rotary gives you is a score that depends on relative offset, the right *form* of dependence. It does not give you a model that has seen those offsets. Two things break past the trained length. The rotation angles for large positions lie outside the range the model encountered, so the query-key geometry is unfamiliar even though it is well defined. And the highest-frequency dimensions alias once the offset exceeds their period, after which attention cannot separate nearby from far positions in those dimensions. The result is the familiar curve: perplexity flat for a while past the training length, then climbing.

The fixes that work are not free. Position interpolation rescales indices so a longer sequence maps into the range the model saw, at the cost of compressing resolution between nearby positions. Frequency-base adjustments stretch the slow dimensions so their period structure covers more length. Both are fine-tuning interventions with their own hyperparameters, and both trade something — resolution, or fidelity of short-range offsets — for length. It is a reasonable trade, but not free extension.

What I take from all this: positional encoding looks like an implementation detail and is really a statement about what the model is allowed to know. Absolute schemes are adequate when the length distribution is fixed; relative schemes put the offset into the score by construction, which is why they travel better. But no scheme removes the need to think about range, and length range is a separate engineering problem with its own evaluation.

## Sources

- Ashish Vaswani et al., *Attention Is All You Need*, 2017. https://arxiv.org/abs/1706.03762
- Jianlin Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021. https://arxiv.org/abs/2104.09864
