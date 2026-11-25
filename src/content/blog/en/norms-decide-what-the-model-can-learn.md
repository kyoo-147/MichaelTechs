---
title: "Norms Decide What the Model Can Learn"
description: "A norm is not a knob: the shape of its unit ball decides which directions are cheap, which weights hit exactly zero, and which features survive."
pubDatetime: "2023-04-23T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Norms
  - Regularisation
  - Sparsity
categories:
  - Math
---

## A norm is a way of pricing directions

Every regularised objective I write has the same skeleton:

$$\min_w \; L(w) + \lambda \, \Omega(w)$$

where $L$ is the data term and $\Omega$ is a penalty on the parameters. Almost always, $\Omega$ is a norm. A norm is a function $\lVert \cdot \rVert : \mathbb{R}^d \to \mathbb{R}$ that is zero only at the origin, scales linearly, $\lVert \alpha w \rVert = |\alpha| \lVert w \rVert$, and satisfies the triangle inequality $\lVert u + v \rVert \le \lVert u \rVert + \lVert v \rVert$. The first two axioms say that a norm measures size. The third is the one that does the work, because it is exactly the condition that makes the unit ball convex.

That convexity is why the penalty and the constraint turn out to be the same object. The problem above has the same solution set, for some $t$ that depends on $\lambda$, as

$$\min_w \; L(w) \quad \text{subject to} \quad \lVert w \rVert \le t.$$

So when I pick a norm I am not picking a scalar knob. I am picking a shape, and the shape states which directions in parameter space are expensive to move in. Two penalties of the same nominal strength can produce very different fitted models because they disagree about the price of moving along different directions.

## Three balls, three geometries

The three norms that come up constantly are

$$\lVert w \rVert_1 = \sum_{i=1}^d |w_i|, \qquad \lVert w \rVert_2 = \Big( \sum_{i=1}^d w_i^2 \Big)^{1/2}, \qquad \lVert w \rVert_\infty = \max_{1 \le i \le d} |w_i|.$$

Their unit balls are the diamond, the sphere, and the cube. In two dimensions this is easy to draw, and it is worth drawing once, because the picture contains the entire argument.

| Norm | Unit ball in $\mathbb{R}^d$ | Cost of one coordinate | Behaviour at zero |
|---|---|---|---|
| $\ell_1$ | cross-polytope, $2d$ vertices | constant, independent of magnitude | non-differentiable cusp |
| $\ell_2$ | Euclidean ball, smooth boundary | grows quadratically with magnitude | smooth, no cusp |
| $\ell_\infty$ | axis-aligned cube, $2^d$ vertices | flat until the cap | non-differentiable at the cap |

The $\ell_2$ ball has no corners: every boundary point looks the same up to rotation. The $\ell_1$ ball is nothing but corners. The $\ell_\infty$ ball is corners and flat faces and very little else. Which of these I choose determines whether a fitted solution is able to sit exactly on a coordinate axis.

## The corner is the whole story

In two dimensions the $\ell_1$ ball is the diamond with vertices $(\pm 1, 0)$ and $(0, \pm 1)$. A vertex is the only kind of point on that boundary where one coordinate is exactly zero. In $d$ dimensions the vertices are the $2d$ signed coordinate axes, each of them carrying $d - 1$ zeros. So if the solution lands on a vertex, most coordinates are exactly zero.

Why should it land on a vertex? The first-order condition answers this. Minimise a smooth $L$ subject to $\lVert w \rVert_1 \le t$, and at an optimum $w^\star$ on the boundary the negative gradient $-\nabla L(w^\star)$ has to lie in the normal cone of the ball at $w^\star$. At a smooth boundary point the normal cone is a single ray, so the gradient must line up with it exactly, which is a condition that holds on a set of measure zero. At a vertex the normal cone is a full solid cone, spanned by the normals of the facets that meet there, so an open set of gradients all map to that same vertex. The vertex does not need luck. It attracts.

For the $\ell_1$ ball the normal cone at the vertex $e_j$ is the set of vectors $g$ with $g_j \ge \max_{i \ne j} |g_i|$. If the gradient of the data term has one coordinate that dominates the others in magnitude, the constraint makes that coordinate carry the weight and switches the rest off. For the $\ell_2$ ball every boundary point is smooth, the normal cone is a ray everywhere, and exact zeros are a coincidence rather than an attractor. That is the geometric content of the statement that $\ell_2$ does not produce sparse solutions. It is not a property of the optimiser. It is a property of where the corners are.

## Shrinking versus thresholding

The one-dimensional case strips the geometry away and leaves the arithmetic.

With an $\ell_2$ penalty, minimising $\tfrac{1}{2}(w - z)^2 + \tfrac{\lambda}{2} w^2$ gives

$$w = \frac{z}{1 + \lambda}.$$

With an $\ell_1$ penalty, minimising $\tfrac{1}{2}(w - z)^2 + \lambda |w|$ gives

$$w = \operatorname{sign}(z) \, \max(|z| - \lambda, \, 0).$$

The first divides, the second subtracts. Proportional shrinkage applies the same relative force to a large weight and to a small one, so a small weight stays small and stays nonzero. Thresholding works on an absolute scale: anything with magnitude below $\lambda$ becomes exactly zero, and everything above it is shifted inward by exactly $\lambda$. That constant is not a detail. It is why the sparsity pattern depends on the units the features happen to be measured in, and it is the hinge of the argument I make further down.

In more than one dimension these closed forms are exact only for an orthonormal design; when features are correlated, the coordinates interact and the solution no longer separates. The qualitative difference survives, though. An $\ell_2$ penalty keeps every coordinate alive. An $\ell_1$ penalty is willing to switch coordinates off.

## Two priors, one penalty

There is a second reading of the same penalty. Write the posterior over weights and take the negative logarithm:

$$-\log p(w \mid D) = -\log p(D \mid w) - \log p(w) + \text{const}.$$

A zero-mean Gaussian prior $p(w) = \prod_i \mathcal{N}(0, \tau^2)$ contributes $\frac{1}{2\tau^2}\sum_i w_i^2$, which is weight decay with $\lambda$ tied to the prior variance. A zero-mean Laplace prior $p(w) = \prod_i \frac{1}{2b}\exp(-|w_i|/b)$ contributes $\frac{1}{b}\sum_i |w_i|$, which is the $\ell_1$ penalty. This correspondence is standard, and it is usually where the explanation stops, which is a shame, because the useful part is what the two priors actually assert.

The Gaussian says that weights are small and that smallness is unremarkable; it has no preferred coordinate and no preferred value. The Laplace has a sharp peak at zero, a kink in the log-density, and that kink is the probabilistic restatement of the corner: exactly zero is a specially plausible value, not merely a small one. Neither prior is a claim about the world that I should expect to be true. Both are statements about which hypotheses I want the model to find cheap to represent, and the penalty is the price list.

## Clipping is a norm constraint on the step

The same geometry appears in an entirely different place: the update. Global norm clipping rescales the gradient $g$ so that $\lVert g \rVert_2 \le c$. Per-coordinate clipping caps each $|g_i|$ at $c$, which is an $\ell_\infty$ constraint. The difference is not cosmetic. Norm clipping preserves the direction of the update and changes only its length. Coordinate clipping changes the direction whenever one coordinate is much larger than the others, because it shrinks that coordinate and leaves the rest alone.

This constrains the decision variable "how far do I step", not the weights themselves, so it is a trust region rather than a prior. But the shape of the trust region is chosen by the same act of picking a norm: $\ell_2$ gives a ball, $\ell_\infty$ gives a box, and inside a box a single oversized coordinate does not drag the others down with it. If I clip by norm and one coordinate dominates, I slow the whole step to accommodate that one coordinate. That trade is a design decision about geometry, and it is worth making deliberately rather than by copying whichever line of code was in the last repository I read.

## The contrarian case: L1 does not select your features

The standard framing is that $\ell_1$ regularisation performs feature selection. I think that sentence is wrong in a way that matters, and it is wrong because it names the wrong cause.

The optimiser does not select features. It minimises a scalar. What produces zeros is the geometry, meaning corners and an absolute threshold, together with the scale of the problem, meaning how the features are scaled, how large $\lambda$ is, how many observations there are relative to parameters, and how correlated the features are. Change the scaling and the selected set changes. That is not a defect of the solver. It is the threshold doing exactly what the arithmetic in the one-dimensional case says it must do.

Correlation is where the framing fails hardest. Take two features that are near-duplicates. The data term is nearly flat along the direction that moves weight from one to the other, and the $\ell_1$ ball is not strictly convex, so the solution is not a single point. It is a face of the ball. Which point on that face comes back depends on the algorithm. Coordinate descent, proximal gradient, and an interior-point method run to a different tolerance can each return a different corner, and a different corner means a different feature. The sparsity level is well determined; the identity of the survivor is not. Reporting that the model selected feature A when A and B are interchangeable is reporting an artefact of solver order as a finding.

Two practical consequences follow. First, report the regularisation path rather than one selected set at one $\lambda$, because the set is not stable across $\lambda$ and, in the high-dimensional regime, a variable can enter the path and leave it again. Second, when features are correlated, do not read substantive or causal meaning into which one carries the weight. Either group them, or mix a small $\ell_2$ term into the penalty so the objective becomes strictly convex and the solution unique, at the cost of no longer being exactly sparse. Elastic net style penalties are a compromise, and they are a compromise precisely because the pure $\ell_1$ answer is not unique.

None of this makes $\ell_1$ a bad default. It makes the phrase "does feature selection" incomplete. An $\ell_1$ penalty selects one representative of an equivalence class, and the class is defined by the geometry and by the scaling of the problem. An honest report says which representative, under which scaling, along which path. The dishonest one says the data chose the feature.

## Sources

- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, 2004. https://web.stanford.edu/~boyd/cvxbook/
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning*, chapter 7, 2016. https://www.deeplearningbook.org/contents/regularization.html
