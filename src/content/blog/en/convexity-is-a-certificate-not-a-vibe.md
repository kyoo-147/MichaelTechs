---
title: "Convexity Is a Certificate, Not a Vibe"
description: "Convexity buys a checkable certificate of optimality, not speed. Where that certificate still pays in an ML pipeline, and where conditioning turns it into a boundary condition."
pubDatetime: "2023-08-20T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Convex Optimization
  - Duality
  - Optimization
categories:
  - Math
---

Convexity is one of the few properties in machine learning that arrives with a proof. That is its entire value. A convex problem does not merely tend to converge - it has a certificate saying the point you found is the best point, and the certificate can be checked without trusting the solver.

The definition is algebraic. A set $C$ is convex if the segment between any two of its points stays inside it; a function is convex if its epigraph is a convex set. Neither statement is about how the curve looks. Both can be verified by composition rules, by a Hessian check, or by construction.

The composition rules are what I actually use. A non-negative weighted sum of convex functions is convex. Composition with an affine map preserves convexity. Pointwise maximum preserves it. Partial minimisation over a convex variable preserves it. Most convex formulations in a pipeline are assembled from those rules rather than proved from scratch.

## Why a local minimum is global

One inequality does the work. For differentiable convex $f$, every gradient gives a global lower bound:

$$
f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}).
$$

The tangent plane at any point lies below the whole function. If $\nabla f(\mathbf{x}) = 0$, the right side is $f(\mathbf{x})$ for every $\mathbf{y}$, so nothing is better. Local optimality becomes global with no search and no restarts.

The same inequality shows where the guarantee thins out. It is a statement about the true function, not about an estimate of its gradient. With a noisy gradient the bound is noisy, and a stochastic method near the optimum oscillates in a neighbourhood whose size follows the noise. Convexity tells me the shape of the space; it does not tell me that a sampler reaches the bottom to arbitrary precision.

## Duality is the certificate

Take a constrained problem:

$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \le 0 .
$$

The Lagrangian is $L(\mathbf{x},\boldsymbol\lambda) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x})$ with $\lambda_i \ge 0$. The dual function $d(\boldsymbol\lambda) = \inf_{\mathbf{x}} L(\mathbf{x},\boldsymbol\lambda)$ is an infimum of affine functions of $\boldsymbol\lambda$, so it is concave, and $d(\boldsymbol\lambda) \le p^\star$ for every feasible $\boldsymbol\lambda$. That is weak duality, and it holds whether or not the problem is convex.

For a convex problem with a constraint qualification such as Slater's condition, strong duality holds and the best dual value equals the primal optimum. Then a feasible primal point with value $v$ and a dual point with value $d$ give $v - d \ge 0$, and that gap bounds the distance to optimality. The solver does not have to be trusted; the gap is the proof.

The multipliers carry meaning too. In an allocation problem $\lambda_i$ is the marginal price of relaxing constraint $i$, which is often the number the rest of the business actually wanted.

## Where convex formulations still win

The honest list is shorter than a course outline and more specific.

Calibration is usually convex. Temperature scaling on log loss is a one-dimensional convex problem. Isotonic regression is convex and solved exactly by pool-adjacent-violators. A small logistic regression with a convex penalty is something I hand to a solver rather than to an optimiser with a learning rate.

Subproblems carry most of the value. Assignment, scheduling, blending and allocation become linear or quadratic programs once the discrete choices are fixed, and the relaxation supplies both a solution and a bound. Entropic optimal transport is convex and fast for structural reasons. A support vector machine is a convex quadratic program whose dual is where the kernel trick lives.

And even then, the guarantee is to the surrogate. A convex loss used for classification is a convex proxy, and the quantity minimised is not the metric that gets reported. Convexity certifies the proxy. Whether the proxy is the right problem is a separate argument, and it is usually the one that decides whether the pipeline works.

## The framing breaks here: "convex equals easy" is a claim about solvers

This is where the tutorials mislead. Convexity removes local minima. It does not remove conditioning, scale, memory, or precision.

Take gradient descent with step $1/L$ on an $L$-smooth, $\mu$-strongly convex function. The rate depends on $\kappa = L/\mu$:

$$
f(\mathbf{x}_k) - f^\star \le \left(1 - \frac{1}{\kappa}\right)^{k}\left(f(\mathbf{x}_0) - f^\star\right).
$$

Badly scaled features make $\kappa$ enormous, and the bound is useless long before the loss looks flat. Convex, first-order, and slow. Rescaling, preconditioning or a second-order method changes the constant more than any theorem does.

Second-order methods hit their own wall. Interior-point methods form and factor a KKT system at each iteration, which is comfortable for a few thousand variables and unpleasant beyond that. The compromise that actually ships is first-order plus structure: sparsity, separability, a proximal operator. The guarantee survives and the tuning does not disappear.

Most deep learning is not convex. There the theory is a boundary condition, not a tool for the training loop. What it still offers is a set of local models: a convex upper bound on a non-convex term, a convex subproblem inside an alternating scheme, a layer that is convex in its input by construction. That is the move that pays, and it is a move about decomposition rather than about pretending the whole network is convex.

| Claim | What it buys | What it does not buy |
| --- | --- | --- |
| The problem is convex | A local optimum is global | Speed, conditioning, numerical accuracy |
| The dual gap is small | A checkable certificate | A small gap in the metric you report |
| A convex surrogate is used | A tractable objective | That the surrogate is the real objective |

## How I use it

Convexity is a design constraint and a diagnostic, not a mood. If a piece of a pipeline can be written as a convex problem, I write it that way and keep the certificate. If it cannot, I stop looking for a proof and start measuring conditioning, step-size sensitivity, and what the loss does when the inputs are rescaled.

So the question I ask is narrower than the textbook one: convex in what, solved by what, certified against which objective. A convex problem solved badly is still solved badly, and a non-convex problem with a good decomposition often wins.

## A certificate you can check by hand

The gap matters only if someone else can verify it, so it is worth checking one by hand. Take a two-variable covering problem: minimise $4x + 3y$ subject to $2x + y \ge 10$, $x + 3y \ge 12$ and $x, y \ge 0$. Its dual maximises $10\lambda_1 + 12\lambda_2$ subject to $2\lambda_1 + \lambda_2 \le 4$, $\lambda_1 + 3\lambda_2 \le 3$ and $\lambda \ge 0$.

Propose $x = (3.6, 2.8)$ alongside $\lambda = (1.8, 0.4)$. The primal point is feasible: $7.2 + 2.8 = 10$, $3.6 + 8.4 = 12$, value $14.4 + 8.4 = 22.8$. The dual point is feasible too: $3.6 + 0.4 = 4$, $1.8 + 1.2 = 3$, value $18 + 4.8 = 22.8$. Two feasible points, one value each, so the gap is zero and both are optimal. That is the entire proof: a dozen multiplications, no iteration, no dependence on a stopping rule.

The pairing is classical, and Dantzig's account of linear programming leans on it: a feasible dual point is a bound you can hand to a reviewer, and optimality is where the two bounds meet.

A loose certificate still certifies. The multiplier $\lambda = (1.5, 0.5)$ is feasible and worth $15 + 6 = 21$, which rules out anything below 21 and nothing more. Pair it with the feasible primal point $(4.5, 2.5)$, worth $18 + 7.5 = 25.5$, and the optimum is trapped inside $[21, 25.5]$. Swap in the multiplier $(1.8, 0.4)$ and the bracket narrows from 4.5 wide to 2.7 on that change alone: the same primal point now sits 2.7 above the optimum, 11.8 percent above 22.8. The certificate is never wrong, only loose until the multiplier is right.

Two details matter. Complementary slackness here is arithmetic rather than insight: both variables are positive, so both dual constraints have to be tight, and they are. And these numbers are rational but not binary-exact, so the feasibility test on a machine is a residual against a tolerance, and the honest thing to report is the residual, not a zero.

The same accounting certifies the answer nobody wants. To prove a model has no feasible point, weight its constraints by non-negative numbers so that every variable coefficient lands at or below zero while the constant turns positive. Adding the weighted constraints then asserts that a non-positive quantity is at least a positive one, which cannot hold. "Infeasible" from a solver is a claim; that combination is a proof.

Checking a candidate pair is one pass over the constraint matrix, a few operations per nonzero, while a solve repeats that work on every iteration. That asymmetry is why a review can audit a claimed optimum without re-running anything, and why rescaling a row is safe: the row's multiplier rescales by the same factor, so it stays a price per unit of that row, in the units you actually wrote down.

## Sources

- Stephen Boyd and Lieven Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), Cambridge University Press (2004).
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, [Deep Learning, ch. 8](https://www.deeplearningbook.org/contents/optimization.html) (2016).
- George B. Dantzig, [Linear Programming and Extensions](https://archive.org/details/linearprogrammin0000dant), Princeton University Press (1963).
