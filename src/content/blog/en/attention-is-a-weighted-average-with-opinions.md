---
title: "Attention Is a Weighted Average with Opinions"
description: "Attention is three learned projections, a scaled inner product, and a softmax that makes every head output a convex combination of value vectors."
pubDatetime: "2023-09-30T09:00:00.000Z"
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

That last sentence is the part I want to keep in view. $A$ is row-stochastic. Not roughly, not in effect - by construction.

## The output is a convex combination

The head output is $AV$, and row $i$ of that product is

$$
\mathbf{o}_i = \sum_j a_{ij}\mathbf{v}_j, \qquad a_{ij} \ge 0, \qquad \sum_j a_{ij} = 1,
$$

which is a convex combination of the value vectors. The output at a position is a weighted average of the values at other positions, with weights set by query-key geometry. That is the mechanism. Everything else - that it focuses on the subject, that it routes information, that it decides what is relevant - is a story about what those weights tend to do in trained models.

The weighted-average reading also says what the mixing step cannot do. A convex combination cannot leave the convex hull of the value vectors. If a direction is absent from every $\mathbf{v}_j$, no choice of weights produces it. The head can only re-mix what $W_V$ already made available.

## Multiple heads, then an output projection

Multi-head attention runs $h$ of these in parallel with separate projections, concatenates, and applies $W_O$:

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W_O .
$$

Each head has its own $W_Q, W_K, W_V$, hence its own subspace for computing similarity and its own value subspace to mix. The concatenation is where the parallel-subspaces picture comes from, and it holds as far as it goes.

$W_O$ matters more than it usually gets credit for. The convex-combination constraint applies per head, before $W_O$. The output projection mixes across heads and can produce directions that no single head's value space contained. This is the first place in the block where genuinely new linear combinations appear - which is why "attention only averages" is true of the mixing step and false of the block.

## The residual stream

The framing I find most useful comes from Anthropic's *A Mathematical Framework for Transformer Circuits*. Instead of reading the model as a stack of layers transforming a hidden state, read it as a residual stream that persists across depth. Each block reads from the stream, computes something, and adds its result back:

$$
\mathbf{x}' = \mathbf{x} + \mathrm{Attn}(\mathrm{LN}(\mathbf{x})), \qquad
\mathbf{x}_{l+1} = \mathbf{x}' + \mathrm{MLP}(\mathrm{LN}(\mathbf{x}')).
$$

In this view, heads are independent additive writers to a shared channel. Each head has a QK circuit deciding where it reads from and an OV circuit deciding what it writes. Because the writes are additive, heads can be analysed semi-independently, and composition between heads - one head's output feeding another's query or key - becomes an object of study rather than a metaphor.

I do not think this is the only correct view of a transformer. It is the one that made the arithmetic stop feeling arbitrary to me. The stream is a sum of contributions; the block is a set of instructions for reading a slice of that sum and adding something back.

## "The model decides what to look at" is an interpretation, not a mechanism

Here is where the standard explanation misleads.

"The model decides what to look at" describes a behaviour we hope emerges after training. It does not describe the computation. The computation is: project, take inner products, exponentiate and normalise, then average. A decision implies a choice among alternatives with a criterion; what happens instead is a smooth, differentiable weighting whose shape is set by the learned $W_Q$ and $W_K$. There is no discrete selection step to point at, and no place where the model knows that token 7 is the subject.

The convex-combination property sharpens the point. Because a head's output is a weighted average of its value vectors, the head cannot introduce information that was not already in the value subspace. Its contribution to expressiveness is a re-weighting. Which directions exist to be mixed is set by $W_V$; the mixing pattern is set by $W_Q$ and $W_K$; the new directions come from $W_O$, from the MLP, and from the embeddings themselves. When people say attention is where the model understands relationships, they are usually describing the joint effect of all of these and attributing it to the one component whose weights happen to look like a heat map.

None of this makes attention uninteresting. A learned, input-dependent, differentiable weighted average is a lot of machinery, and the convexity is a constraint rather than a disappointment - constraints are what make a thing analysable. But the gap between "it averages values with learned weights" and "it decides what to look at" is exactly where most confusion about transformers lives, and I find it worth closing before reaching for the metaphor.

## Working one head by hand

Here is the smallest example I could build that keeps the whole shape of the operation. Take $d_k = 4$, so the divisor is $\sqrt{d_k} = 2$, one query $q = (1,2,0,1)$ and three keys $k_1 = (2,1,3,2)$, $k_2 = (0,1,1,0)$, $k_3 = (1,-1,0,1)$. The inner products are 6, 2 and 0, so the scaled logits are 3, 1 and 0. Exponentiating gives 20.0855, 2.7183 and 1, summing to 23.8038, so the row of $A$ is $(0.8438, 0.1142, 0.0420)$. With values $v_1 = (1,0)$, $v_2 = (0,1)$, $v_3 = (-1,-1)$ the head writes $(0.8018, 0.0722)$ at that position.

Now drop the divisor. The logits become 6, 2 and 0, the row becomes $(0.9796, 0.0179, 0.0024)$, and one position takes 98 percent of the mass while the other two share the remaining 2. That is what "softmax saturates" means as a number rather than a caution: the row is already near one-hot, so the gradient reaching the other two keys is small and the query has little purchase on them.

The divisor is also a temperature, and it is the part of the formula easiest to read as decoration. Softmax is invariant to shifts of the logits but not to scalings, so dividing by $\sqrt{d_k}$ sets how far the row sits from uniform. The entropy of the row above is 0.524 nats against a ceiling of $\ln 3 = 1.099$, about half the available uncertainty. Without the divisor the same row falls to 0.107 nats, roughly a tenth. Half the headroom still carries information about all three keys; a tenth is close to a lookup.

The divisor is also younger than attention. *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau, Cho and Bengio, 2014) computes the same kind of weighted average nearly three years before the transformer paper, and its scorer has no dimension-dependent divisor. The alignment model there is a small feedforward network evaluated on a pair, the decoder's previous state and one encoder annotation, with the softmax normalising over source positions. The weighted sum, the normalisation and the reading of the weights as a soft alignment are all already present. What changed in 2017 was the scorer: an inner product of two projections builds the whole score matrix in one matmul, where a network over pairs must be evaluated once per pair. The averaging survived that trade; the scorer did not.

So the constant under the square root is not notation. The inner product of two vectors with unit-variance entries has standard deviation $\sqrt{d_k}$ rather than 1, so raw logits would widen as the head widens, and a logit that widens pushes the softmax toward one-hot whether or not the model wanted that. Dividing by $\sqrt{d_k}$ puts the scorer back on a unit scale. Attention worked for years without the constant, because the average was never the fragile part.

## What the average costs to serve

The convex combination is cheap at each position. The expensive part is elsewhere, and it prices out in plain numbers. Take the published configuration of GPT-2 small: 12 layers, $d = 768$, 12 heads, $d_k = d_v = 64$, and a context of 1024 tokens.

Start with the projections. Each of $W_Q$, $W_K$, $W_V$ and $W_O$ is $768 \times 768$, so each holds $768^2 = 589{,}824$ weights. Four per layer is $4 \times 589{,}824 = 2{,}359{,}296$, and across 12 layers that is $12 \times 2{,}359{,}296 = 28{,}311{,}552$. At two bytes per weight in fp16 the whole attention stack is $28{,}311{,}552 \times 2 = 56{,}623{,}104$ bytes, which is 54 MiB. That number is fixed, whether the model serves one token or a million.

The state that decoding keeps is not fixed. Producing a token needs one query and every key and value already computed, so both must be cached. Per layer and per token that is one $K$ vector of 768 numbers and one $V$ vector of 768 numbers, so $2 \times 768 = 1536$ numbers. Across 12 layers, $12 \times 1536 = 18{,}432$ numbers per token. In fp16, $18{,}432 \times 2 = 36{,}864$ bytes, exactly 36 KiB per token. A full 1024-token context costs $1024 \times 36 = 36{,}864$ KiB, or 36 MiB, and a batch of eight such contexts costs $8 \times 36 = 288$ MiB, which is more than five times the 54 MiB of weights that produce it and appears in no parameter count.

Two details matter more than the totals, and I find the first one the more surprising. The cache does not depend on $h$. Heads are concatenated, so $h \times d_k = 768$, and 12 heads of 64 cost exactly what one head of 768 would. Second, scoring is quadratic in context length: at $n = 1024$ one head compares $1024^2 = 1{,}048{,}576$ query-key pairs, and 12 heads compare $12 \times 1{,}048{,}576 = 12{,}582{,}912$ pairs for a single sequence, before any value is mixed.

This is the gap that Noam Shazeer's *Fast Transformer Decoding: One Write-Head is All You Need* goes after. Let every head share one key head and one value head, and the cache per token per layer falls to $2 \times 64 = 128$ numbers instead of 1536; across 12 layers that is $12 \times 128 = 1536$ numbers, or 3 KiB in fp16. The saving is a factor of $36 / 3 = 12$, which is exactly $h$. The averaging is untouched: same queries, same softmax, same mixing. Only the state needed to recompute those weights shrinks, which is the paper's claim - that decoding is bandwidth-bound on loading keys and values, not on multiplying them.

So the convex combination is the cheap half of the operation. The bill arrives because every new token redoes the average over everything before it, so every past key and value stays resident. The weights that set the average are learned once and frozen; the keys and values that feed it accumulate per token, per layer, per sequence. That asymmetry is why a long context costs more to serve than a short one, and it is a memory problem before it is a compute problem.

## Sources

- Ashish Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
- Anthropic, [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021).
- Dzmitry Bahdanau et al., [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2014).
- Noam Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019).
