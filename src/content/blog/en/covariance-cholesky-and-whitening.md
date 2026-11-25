---
title: "Covariance, Cholesky, and Whitening"
description: "A covariance matrix summarises a cloud of points, and its Cholesky factor is the cheapest honest square root of that summary."
pubDatetime: "2023-11-04T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Covariance
  - Cholesky
  - Linear Algebra
categories:
  - Math
---

Three objects keep showing up together in numerical work: a covariance matrix, the triangular factor that "takes its square root", and the change of coordinates that turns a correlated cloud into a sphere. They are usually taught apart, in a statistics course and a numerical linear algebra course, and the separation hides the fact that they are one idea with three faces.

## A cloud of points, compressed into a matrix

Take $n$ points $x_1, \dots, x_n$ in $\mathbb{R}^d$, subtract their mean $\mu$, and stack them as the rows of an $n \times d$ matrix $X$. The covariance matrix is

$$\Sigma = \frac{1}{n} X^\top X = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^\top .$$

The diagonal entry $\Sigma_{jj}$ is the variance of coordinate $j$; the off-diagonal $\Sigma_{jk}$ is the covariance between coordinates $j$ and $k$. Everything the cloud does at second order is in there, and nothing else is: $\Sigma$ does not know about clusters, tails, or curvature. For a Gaussian that is the whole story, which is why the matrix carries so much weight in statistics. For anything else it is a summary, and summaries lose things.

The geometric reading is the one I find most useful. The level set $\{x : (x-\mu)^\top \Sigma^{-1} (x - \mu) = 1\}$ is an ellipsoid centred at $\mu$, and $\Sigma$ fixes its axis lengths and orientations. Whitening, in this picture, is the linear change of coordinates that turns the ellipsoid into a sphere. Most of the algebra below is a way of doing that without ever forming $\Sigma^{-1}$.

## Positive semi-definiteness is a theorem, not a constraint

For any vector $v$,

$$v^\top \Sigma v = \frac{1}{n}\sum_{i=1}^{n}\left( v^\top (x_i - \mu) \right)^2 \ge 0 .$$

So a covariance matrix built from real centred data is positive semi-definite by construction. It is not an assumption you impose; it is a property you get for free. The only reason the word "constraint" appears in the literature is that people also build $\Sigma$ by other routes, and those routes can break the guarantee.

It is positive *definite* when the points span $\mathbb{R}^d$. If they lie in a proper subspace, some direction $v$ satisfies $v^\top(x_i - \mu) = 0$ for every $i$, so $\Sigma v = 0$: the matrix is singular, the ellipsoid is flat along that direction, and the inverse does not exist. That is the ordinary case, not the exception - duplicated columns, a one-hot block that sums to a constant, more features than rows.

Indefinite matrices appear when the estimate is no longer a second moment of the data. The classic route is pairwise deletion: each $\Sigma_{jk}$ comes from the rows where both $j$ and $k$ are observed, so different entries come from different subsets of the data, and the result need not be positive semi-definite. Adding $\epsilon I$ makes the factorisation run, but it does not make the estimate correct; it quietly changes the model by inflating every variance. I would rather find out why the matrix is indefinite.

## The Cholesky factor is a square root with structure

A square root of $\Sigma$ is any $S$ with $SS^\top = \Sigma$. There are many. Cholesky picks the lower-triangular one with a positive diagonal:

$$\Sigma = LL^\top, \qquad L \text{ lower triangular}, \quad L_{jj} > 0 .$$

For a positive definite matrix that $L$ is unique, and computing it costs about $n^3/3$ flops: a column-by-column recurrence, no iteration, no convergence criterion.

The triangularity is the part that earns its keep. Solving $Lu = b$ by forward substitution costs $n^2/2$ flops and is backward stable; you never form $L^{-1}$. More importantly, the structure is a statement about ordering. The first entry of $L^{-1}(x-\mu)$ depends only on the first entry of $x - \mu$, the second on the first two, and so on. Whitening with a Cholesky factor is a sequential decorrelation, each new coordinate cleaned against the ones before it, which is what makes it natural for conditional sampling and for anything that consumes the vector one entry at a time.

There is a neat way to see where $L$ comes from. Take the thin QR factorisation of the centred data, $X = QR$, with $Q$ orthonormal and $R$ upper triangular. Then $X^\top X = R^\top R$, so $\Sigma = (R^\top/\sqrt{n})(R/\sqrt{n})$ and $L = R^\top/\sqrt{n}$: the Cholesky factor is the $R$ of a QR factorisation of the data, transposed and rescaled, which amounts to Gram-Schmidt on the columns taken in reverse order.

One free byproduct: since the factorisation divides by each pivot, a non-positive pivot certifies that the matrix is not positive definite. Cholesky is the cheapest definiteness test I know, and it hands you a factor at the same time.

## Whitening is division by the factor you chose

Given $\Sigma = LL^\top$, define

$$z = L^{-1}(x - \mu).$$

Then

$$\mathrm{Cov}(z) = L^{-1}\Sigma L^{-\top} = L^{-1} L L^\top L^{-\top} = I .$$

The ellipsoid becomes the unit sphere. But notice what that equation does not pin down: if $W$ whitens, so does $RW$ for any orthogonal $R$, since $RW\Sigma W^\top R^\top = R R^\top = I$. Every whitening map has the form $W = R L^{-1}$. Cholesky whitening is the choice $R = I$. PCA whitening is a different $R$, and ZCA whitening picks the $R$ that keeps $W$ as close to the identity as possible. All of them produce identity covariance; they differ by a rotation, and that rotation is a modelling decision rather than a detail.

In code the whole thing is three lines, and none of them is an inversion:

```python
import numpy as np

L = np.linalg.cholesky(Sigma) # Sigma = L @ L.T
z = np.linalg.solve(L, x - mu) # whiten: Cov(z) = I
x_sample = mu + L @ np.random.randn(d) # sample: Cov(x_sample) = Sigma
```

A triangular solve is $O(d^2)$ per point and backward stable; forming $\Sigma^{-1}$ explicitly squares the condition number and throws away digits you had. Trefethen and Bau make this point for linear systems, and it applies unchanged: the factor is for solving with, not for inverting.

## Where it pays: conditioning, sampling, normalisation

**Conditioning.** Newton's method on a quadratic with Hessian $H$ takes steps $H^{-1}g$. If $H$ is covariance-like with a wide spread of eigenvalues, gradient descent crawls along the flat directions and overshoots the sharp ones. Whitening the parameters is preconditioning, and the Cholesky factor is the preconditioner you can build in one pass. A useful identity: since $\|L\|_2^2 = \|LL^\top\|_2 = \|\Sigma\|_2$, we get $\kappa_2(L) = \kappa_2(\Sigma)^{1/2}$. Working with $L$ rather than $\Sigma$ halves the condition number in the exponent, which is the numerical reason triangular solves are preferable to an inverse.

**Sampling.** To draw $x \sim \mathcal{N}(\mu, \Sigma)$, draw $z \sim \mathcal{N}(0, I)$ and set $x = \mu + Lz$. Triangularity again: you can generate the first coordinate, then the second conditional on it, and so on, which is exactly how you would sample a spatial field or an autoregressive process by hand. The same line of code is how correlated noise gets into a simulation of asset returns, a synthetic dataset, or a proposal distribution in a sampler.

**Normalisation.** Batch normalisation divides each coordinate by its own standard deviation. That is whitening with a diagonal map: it fixes the variances and leaves every correlation in place. Full whitening removes the correlations too, at $O(d^3)$ per update plus a $d \times d$ matrix that moves with the batch. If a normalisation layer does not do a matrix solve, it is not decorrelating anything, and the two are not interchangeable when features are correlated.

| Task | Cost | What you get |
| --- | --- | --- |
| Per-coordinate standardisation | $O(d)$ | the diagonal of $\Sigma$ only |
| Cholesky factorisation | $\sim d^3/3$ flops | unique lower-triangular $L$ |
| Symmetric eigendecomposition | a multiple of that, iterative | $Q$, $\Lambda$, principal directions |
| Applying a whitening map | $O(d^2)$ per point | one triangular solve |

## The contrarian case against eigendecomposition

The standard advice is: to decorrelate, diagonalise. Rotate into the eigenbasis, scale by $\Lambda^{-1/2}$, and you are done. It is correct, and for most decorrelation jobs it is the wrong tool.

First, cost. Cholesky is a direct method at roughly $n^3/3$ flops; the symmetric eigenvalue problem is iterative and costs several times that. Trefethen and Bau quote about $10n^3$ for a general matrix, and the symmetric case is cheaper but still a multiple. If all you need is some $W$ with $W\Sigma W^\top = I$, you are paying for principal directions and then discarding them.

Second, and this is the argument I care about more: the eigenbasis can be numerically meaningless even when the matrix is perfectly conditioned. Perturbation theory for eigenvectors carries a factor of $\|E\|/\text{gap}$. When two eigenvalues are close - precisely what redundant features or a covariance estimated from few samples produce - the eigenvectors are not determined to any useful accuracy, and the computed basis rotates arbitrarily between runs. Cholesky has no such problem: for positive definite $\Sigma$ it is a unique, continuous function of the entries, the same matrix gives the same $L$, and the map is differentiable. If you backpropagate through a whitening layer, or you simply want the same answer twice, that difference is not academic.

Third, "PCA then whiten" is not "whiten". PCA whitening sends $x - \mu$ to $\Lambda^{-1/2}Q^\top(x-\mu)$; Cholesky whitening sends it to $L^{-1}(x-\mu)$. Both outputs have identity covariance, and they differ by an orthogonal factor, so any downstream step that is rotation-equivariant cannot tell them apart. Almost nothing downstream is. A coordinate-wise nonlinearity, a diagonal regulariser, per-coordinate learning rates, L1 sparsity, quantisation, a tree split on a single feature - each of these cares which rotation you picked. Calling both operations "whitening" and expecting identical numbers is a category error. Whitening fixes the second moment and leaves the rotation free; you have to choose it for a reason.

Where the eigenbasis wins: you want to truncate, dropping small eigenvalues as denoising or compression; you want to read the spectrum; you want ZCA; or $\Sigma$ is only semi-definite and Cholesky needs a jitter you would rather not add. Those are real cases. They are not the case "I need to decorrelate".

A caveat on my own argument: since $\kappa(L) = \kappa(\Sigma)^{1/2}$ and $\kappa(\Lambda^{1/2}) = \kappa(\Sigma)^{1/2}$ too, the two factors are comparably conditioned. The advantage of Cholesky is not a better-conditioned square root. It is that the factor is unique, cheap, and stable in its entries, and that its basis does not collapse when eigenvalues crowd together. I would rather state that precisely than repeat "Cholesky is better conditioned" and be wrong about why.

## Sources

- Trefethen, L. N. and Bau, D. (1997). *Numerical Linear Algebra*. https://people.maths.ox.ac.uk/trefethen/text.html
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. https://probml.github.io/pml-book/
