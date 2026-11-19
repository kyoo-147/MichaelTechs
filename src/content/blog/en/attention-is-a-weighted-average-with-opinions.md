---
title: "Attention Is a Weighted Average with Opinions"
description: "Attention is three learned projections, a scaled inner product, and a softmax that makes every head output a convex combination of value vectors."
pubDatetime: "2026-11-13T09:30:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Transformers
  - Attention
  - Linear Algebra
categories:
  - Math
  - AI
---

Every explanation of attention I read early on used the same metaphor: the model decides what to look at. A query is formed, keys are compared, and the model attends to the relevant parts of the input. It is a useful picture. It is also an interpretation layered on top of four matrix multiplications and a softmax, and separating the two made the mechanism much easier to hold in my head.

## Three projections

Take a sequence of $n$ token vectors stacked into $X \in \mathbb{R}^{n\times d}$. Attention learns three linear maps:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V,
$$

with $W_Q, W_K \in \mathbb{R}^{d\times d_k}$ and $W_V \in \mathbb{R}^{d\times d_v}$. Nothing about these maps is semantically special. They are three different views of the same residual stream; what distinguishes them is only the role their outputs play downstream.

## Scores, then a normalisation

The score between positions $i$ and $j$ is the inner product $q_i \cdot k_j$, divided by $\sqrt{d_k}$:

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

The scaling keeps the magnitude of the scores from growing with $d_k$, which matters because softmax saturates: large logits produce near one-hot weights and near-zero gradients. The softmax runs along the key axis, one row per query, so each row of $A$ is non-negative and sums to one.

That last sentence is the part I want to keep in view. $A$ is row-stochastic. Not roughly, not in effect — by construction.

## The output is a convex combination

The head output is $AV$, and row $i$ of that product is

$$
\mathbf{o}_i = \sum_j a_{ij}\mathbf{v}_j, \qquad a_{ij} \ge 0, \qquad \sum_j a_{ij} = 1,
$$

which is a convex combination of the value vectors. The output at a position is a weighted average of the values at other positions, with weights set by query–key geometry. That is the mechanism. Everything else — that it focuses on the subject, that it routes information, that it decides what is relevant — is a story about what those weights tend to do in trained models.

The weighted-average reading also says what the mixing step cannot do. A convex combination cannot leave the convex hull of the value vectors. If a direction is absent from every $\mathbf{v}_j$, no choice of weights produces it. The head can only re-mix what $W_V$ already made available.

## Multiple heads, then an output projection

Multi-head attention runs $h$ of these in parallel with separate projections, concatenates, and applies $W_O$:

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W_O .
$$

Each head has its own $W_Q, W_K, W_V$, hence its own subspace for computing similarity and its own value subspace to mix. The concatenation is where the parallel-subspaces picture comes from, and it holds as far as it goes.

$W_O$ matters more than it usually gets credit for. The convex-combination constraint applies per head, before $W_O$. The output projection mixes across heads and can produce directions that no single head's value space contained. This is the first place in the block where genuinely new linear combinations appear — which is why "attention only averages" is true of the mixing step and false of the block.

## The residual stream

The framing I find most useful comes from Anthropic's *A Mathematical Framework for Transformer Circuits*. Instead of reading the model as a stack of layers transforming a hidden state, read it as a residual stream that persists across depth. Each block reads from the stream, computes something, and adds its result back:

$$
\mathbf{x}' = \mathbf{x} + \mathrm{Attn}(\mathrm{LN}(\mathbf{x})), \qquad
\mathbf{x}_{l+1} = \mathbf{x}' + \mathrm{MLP}(\mathrm{LN}(\mathbf{x}')).
$$

In this view, heads are independent additive writers to a shared channel. Each head has a QK circuit deciding where it reads from and an OV circuit deciding what it writes. Because the writes are additive, heads can be analysed semi-independently, and composition between heads — one head's output feeding another's query or key — becomes an object of study rather than a metaphor.

I do not think this is the only correct view of a transformer. It is the one that made the arithmetic stop feeling arbitrary to me. The stream is a sum of contributions; the block is a set of instructions for reading a slice of that sum and adding something back.

## "The model decides what to look at" is an interpretation, not a mechanism

Here is where the standard explanation misleads.

"The model decides what to look at" describes a behaviour we hope emerges after training. It does not describe the computation. The computation is: project, take inner products, exponentiate and normalise, then average. A decision implies a choice among alternatives with a criterion; what happens instead is a smooth, differentiable weighting whose shape is set by the learned $W_Q$ and $W_K$. There is no discrete selection step to point at, and no place where the model knows that token 7 is the subject.

The convex-combination property sharpens the point. Because a head's output is a weighted average of its value vectors, the head cannot introduce information that was not already in the value subspace. Its contribution to expressiveness is a re-weighting. Which directions exist to be mixed is set by $W_V$; the mixing pattern is set by $W_Q$ and $W_K$; the new directions come from $W_O$, from the MLP, and from the embeddings themselves. When people say attention is where the model understands relationships, they are usually describing the joint effect of all of these and attributing it to the one component whose weights happen to look like a heat map.

None of this makes attention uninteresting. A learned, input-dependent, differentiable weighted average is a lot of machinery, and the convexity is a constraint rather than a disappointment — constraints are what make a thing analysable. But the gap between "it averages values with learned weights" and "it decides what to look at" is exactly where most confusion about transformers lives, and I find it worth closing before reaching for the metaphor.

## Sources

- Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- Anthropic, [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021).
