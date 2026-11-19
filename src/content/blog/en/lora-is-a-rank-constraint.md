---
title: "LoRA Is a Rank Constraint, Not a Decomposition"
description: "LoRA does not compress a fine-tune after the fact. It forbids the optimiser from leaving a low-rank subspace, and that changes what rank means."
pubDatetime: "2023-10-01T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - LoRA
  - Fine-tuning
  - Low Rank
categories:
  - AI
---

When I first read the LoRA paper, I filed it under compression. The name says low-rank adaptation, the diagram shows two thin matrices instead of one fat one, and the whole thing is sold on the parameter count. It took me a while to notice that the saving is not the interesting part. The interesting part is that the optimiser is never allowed to consider the alternative.

## The shape of the thing

Take a frozen pretrained weight matrix $W_0 \in \mathbb{R}^{m \times n}$. Full fine-tuning learns an update $\Delta W$ of the same shape, which is $mn$ free numbers. LoRA instead writes

$$\Delta W = \frac{\alpha}{r} BA$$

with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, and $r$ small. In a transformer block this targets the attention projections. The forward pass becomes

$$h = W_0 x + \frac{\alpha}{r} B A x$$

The $W_0 x$ term never changes. Only the correction moves.

The parameter count falls from $mn$ to $r(m+n)$. For a square matrix with $m = n = d$ the ratio is $2r/d$, so at $d = 4096$ and $r = 8$ the adapter holds about $0.4\%$ of the numbers in the matrix it adapts.

The point is that $BA$ is a product of two matrices with inner dimension $r$, so it has rank at most $r$ whatever values $A$ and $B$ take. This is not a penalty term nudging the solution toward low rank. It is a structural cap. The set of reachable updates is exactly the set of matrices of rank at most $r$, and no gradient step can produce anything outside it.

## The optimiser only sees a subspace

Precision matters here. Gradient descent on a full-rank parameterisation can move in $mn$ independent directions. The LoRA objective is instead minimised over the rank-$r$ matrices, a manifold of dimension $r(m+n) - r^2$ - less than $r(m+n)$ because $B$ and $A$ are only identifiable up to an invertible $r \times r$ change of basis. Every step stays on that manifold.

The consequence is that the correction is confined to an $r$-dimensional subspace of the input space, read through the row space of $A$, and an $r$-dimensional subspace of the output space, spanned by the columns of $B$. Directions orthogonal to both are invisible. The model is not corrected toward a target function in general, only toward the nearest reachable function inside a thin slice of the space of all corrections. If a task needs a modification depending on a high-dimensional slice of the representation, more training time will not get there. The cap is on capacity and it bites immediately rather than asymptotically.

## Why step zero changes nothing

The initialisation is the detail people skip, and it is what makes the method workable. $A$ starts with small random values - the original implementation used a Gaussian with a small standard deviation - and $B$ is initialised to exactly zero. At the first step $BA = 0$, so

$$\Delta W = 0, \qquad h = W_0 x$$

The adapted model is bit-for-bit the original at initialisation. Not approximately, not in expectation: the forward pass is identical because the added term is the zero matrix.

This buys two things. First, you can start from any pretrained checkpoint and know the loss at step zero is the base model's loss, so anything worse afterwards comes from training rather than from perturbing the weights arbitrarily. A full fine-tune has no such property; its initial loss depends on the noise in the initialisation, and some early budget goes to undoing it.

Second, the gradient signal at step zero is clean. Since $B = 0$, the gradient with respect to $A$ is zero - $\partial L / \partial A$ is proportional to $B^\top(\cdot)$ - while the gradient with respect to $B$ is not. The first update moves $B$ off zero and only then does $A$ receive signal, a mild asymmetry that resolves in a step or two. Later variants zero $A$ and randomise $B$, or give both sides a small nonzero start. The design space is wider than the paper's choice, but the principle holds: make the adapter an identity at step zero.

The other half is that $W_0$ is frozen, so the base model's knowledge cannot be overwritten. A full fine-tune on a narrow dataset can degrade general capability partly because every weight is a candidate for movement.

## Rank, alpha, and what the scale controls

The adapter output is multiplied by $\alpha / r$. Two symbols suggest two independent knobs, but the effects are entangled.

Holding $\alpha$ fixed while raising $r$ changes two things at once: it enlarges the reachable subspace and shrinks the multiplier $\alpha / r$. The paper's authors acknowledge this and suggest treating $\alpha$ as a constant to retune when $r$ changes, since the scaling mainly keeps the effective learning rate roughly stable across adapter widths. Entries of $BA$, as a product of two small near-random matrices, grow with $r$ in a way that partly cancels the $1/r$.

So the honest separation is: $r$ sets capacity, $\alpha$ sets how strongly the adapter contributes relative to the frozen path. Confusing them is a common reason people report that raising rank did not help. Raise $r$ at fixed $\alpha$ and you have changed two things, so you cannot attribute the result.

There is a subtler point. Because $A$ and $B$ are trained jointly through a product, the two factors can drift in scale against each other - one growing while the other shrinks - without changing $BA$ much. This is the same non-identifiability that makes the manifold dimension $r(m+n) - r^2$ rather than $r(m+n)$, and it means weight decay on the factors behaves differently than in a full fine-tune.

## Merging back is exact

At inference you need not keep the adapter separate. Since

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

has the original shape, you can compute it once and replace the frozen weights. The result is not an approximation. The adapted model computes $W_0 x + \frac{\alpha}{r} B A x$; the merged model computes $(W_0 + \frac{\alpha}{r} BA)x$. These are equal by distributivity for every input $x$, with no error term.

This is worth stating plainly because it is sometimes described as a folding that loses something. It does not. What you give up is un-merging and swapping adapters per request over one base - reasons to keep them separate, not reasons the merge is lossy.

| Quantity | Full fine-tune | LoRA |
| --- | --- | --- |
| Trainable parameters | $mn$ | $r(m+n)$ |
| Update rank | up to $\min(m,n)$ | at most $r$ |
| Model at step zero | perturbed | unchanged |
| Merge at inference | n/a | exact |
| Base weights | modified | frozen |

One caveat: after merging, the adapter's effect is entangled with the base weights, so a second adapter trained on the merged model is constrained relative to a different starting point. The composition of two rank-$r$ updates is generally rank $2r$.

## The contrarian part: there is no SVD

Here is where I think the standard framing is not merely imprecise but backwards.

LoRA is routinely described as compressing a fine-tune with an SVD - as if you first computed the full $\Delta W$ and then approximated it with its leading singular vectors. That description appears in blog posts, in slide decks, and in the mental model many practitioners carry. It predicts the wrong behaviour.

There is no SVD anywhere in LoRA. Nothing is decomposed. No full-rank update is ever computed and truncated. The constraint is imposed during training, on the parameterisation itself, before a single gradient step. The optimiser is never aware that a full-rank solution exists, because that solution is not in its search space.

The difference is not cosmetic. If you computed a full fine-tune $\Delta W$ and took its rank-$r$ SVD truncation, you would get the best rank-$r$ approximation to that matrix in the Frobenius norm, by Eckart-Young. That is a reconstruction problem with a well-defined optimum. LoRA solves a different problem: find the rank-$r$ update that minimises the training loss. Its answer is generally not the leading singular vectors of the full fine-tune's update.

The objectives differ. Truncation minimises $\lVert \Delta W - \Delta W_r \rVert_F$ for a given $\Delta W$. LoRA minimises $\mathcal{L}(W_0 + BA)$ over $B, A$. Nothing forces the second minimiser to coincide with the truncation of the first. The loss cares about the function computed, not the weights, so a rank-$r$ update far from the full fine-tune in weight space can achieve a lower loss.

The distinction predicts something specific: rank bounds capacity, not reconstruction error. Under the SVD reading, raising $r$ would monotonically shrink an approximation gap, giving smooth and predictable gains. The capacity reading predicts something messier. Enlarging $r$ enlarges the reachable set, so the best achievable loss is non-increasing in $r$ - you cannot do worse with more room. But the achieved loss depends on optimisation, and higher rank means a larger space to get stuck in, more interaction between the two factors, and more sensitivity to learning rate. It is entirely normal for $r = 16$ to fail to beat $r = 4$ on a given task, and that is not evidence the method is broken. It is evidence the bottleneck was not rank.

The framing also misleads about what the trained factors contain. The columns of $B$ are not the leading left singular vectors of anything, nor the rows of $A$ the leading right ones. They are whatever pair of thin matrices gradient descent landed on, identifiable only up to a change of basis - rotate $A$ by an invertible $r \times r$ matrix, counter-rotate $B$, and you have the same $\Delta W$ and the same loss.

I think the SVD story persists because it is comfortable: low rank is familiar from compression, and SVD is the familiar tool for low rank, so the analogy gives people a handle. But one that predicts the wrong scaling behaviour is worse than none, because it leads people to blame rank for problems that are actually about learning rate, target module choice, or a task that simply does not need many directions.

Read correctly, the method is a fine-tune with a restricted search space, chosen so the restriction is cheap to parameterise and free to remove at inference. Rank is a budget on how many independent directions the correction may use, not a fidelity knob against an ideal answer that was never computed. The frozen base, the zero-initialised product and the exact merge all follow from one decision: keep the correction in a separate, thin, removable object.

## Sources

- Edward Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. https://arxiv.org/abs/2106.09685
