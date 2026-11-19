---
title: "Quadrature: When No Closed Form Exists"
description: "Numerical integration is not a lesser substitute for a formula; it is an engineering tradeoff between truncation error and floating-point roundoff."
pubDatetime: "2023-08-27T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Numerical Integration
  - Quadrature
  - Floating Point
categories:
  - Math
---

Most integrals in working code have no closed form. That is not a failure of the person who wrote the code; it is the normal condition.

Consider the error function, $F(t) = \int_0^t e^{-x^2}dx$, and the apparently simpler $G = \int_0^1 \sin(x^2)dx$. Both have their own names precisely because no combination of elementary functions produces them, and both are smooth, bounded, entirely well behaved, and completely resistant to symbolic integration.

Quadrature is the primary tool; symbolic integration is the special case that works on a small, curated class of functions. The exercises that resolve neatly into $\arctan$ were selected because they resolve neatly, and that selection is the whole distortion.

## Interpolation with weights, which is all quadrature is

A quadrature rule is a finite sum. Pick nodes $x_0, \dots, x_n$ in $[a,b]$ and weights $w_0, \dots, w_n$:

$$
Q(f) = \sum_{i=0}^{n} w_i f(x_i) \approx \int_a^b f(x)\,dx .
$$

Every rule you have met is this shape with a particular choice of nodes and weights. The trapezoid rule is piecewise linear interpolation followed by exact integration of the interpolant. On a panel of width $h$ you replace $f$ by the line through its endpoints, getting $h(f(x_i)+f(x_{i+1}))/2$. Summing over panels,

$$
T_n = h\left[\frac{f(a)}{2} + \sum_{i=1}^{n-1} f(a + ih) + \frac{f(b)}{2}\right],
\qquad h = \frac{b-a}{n}.
$$

Simpson's rule uses a parabola through three points instead. On a panel of width $2h$ it is $\frac{h}{3}(f(x_0)+4f(x_1)+f(x_2))$, and the familiar $1,4,2,4,\dots,4,1$ pattern is simply what falls out of integrating quadratics exactly.

The trapezoid rule is exact for linears; Simpson's is exact for cubics, not just parabolas, because a symmetry in the error term kills the cubic contribution. That cancellation is where its extra order comes from.

## The error terms, and what they actually say

The classical formulas are

$$
\int_a^b f\,dx - T_n = -\frac{(b-a)h^2}{12} f''(\xi),
\qquad
\int_a^b f\,dx - S_n = -\frac{(b-a)h^4}{180} f^{(4)}(\xi),
$$

for some $\xi \in (a,b)$ in each case. Two things follow, and only the first is usually taught. The first is the order: halve $h$ and the trapezoid error drops by four, Simpson's by sixteen.

The second is the structure of the constant. The error is proportional to a derivative evaluated at a point you do not know, which means it is governed by local behaviour in a region you have not identified. If $f''$ is enormous on one narrow interval and small elsewhere, that interval decides the error, and a uniform grid has no idea it exists.

Here the exam version parts company with practice. Exams give you a function and ask you to bound the error from a bound on $f''$. Real problems give you a function you did not write, defined by an experiment or a black-box simulation, and the derivative bound is unavailable.

## Why doubling the points does not double the accuracy

The order statements assume the integrand has enough derivatives. The $h^2$ rate needs $f''$ bounded; the $h^4$ rate needs $f^{(4)}$. When the integrand is not that smooth, the rates degrade — predictably.

Take

$$
\int_0^1 \sqrt{x}\,dx .
$$

The derivative is infinite at the origin. On a uniform grid the trapezoid error decays like $n^{-3/2}$, not $n^{-2}$, and Simpson's rule — needing four derivatives that do not exist — degrades to nearly the same rate. Refining helps, but each doubling of $n$ buys a factor of about $2.83$ instead of four or sixteen.

The general pattern: if the integrand behaves like $x^{\alpha}$ near an endpoint with non-integer $\alpha$, the trapezoid error decays like $n^{-(\alpha+1)}$ and higher-order rules do not rescue you.

Real integrands are frequently not smooth: a payoff with a kink at the strike, a density with a corner where a constraint binds, an empirical distribution that is a sum of step functions. The rate is worse than the textbook, and you cannot detect the degradation from a single run — you have to compute at several resolutions and fit the slope.

One standard trick is worth knowing: remove the singularity by substitution rather than fight it with points. Substituting $x = u^2$ makes the square-root integrand smooth and restores full order, which is almost always cheaper than refining.

## Truncation versus roundoff, and the U-shaped error

The formulas above describe truncation error: the gap between the exact integral and the exact value of the finite sum, which shrinks as $h$ shrinks. But you are not computing the exact value of the sum. You are computing it in floating point, where each evaluation carries relative error around $\epsilon \approx 2.2\times10^{-16}$ and each addition adds more. Over $n$ terms the roundoff grows roughly like $\sqrt{n}\,\epsilon$, or $n\epsilon$ in the worst case.

So the total error behaves approximately as

$$
E(h) \approx \underbrace{C h^{p}}_{\text{truncation}} + \underbrace{\frac{D \epsilon}{h}}_{\text{roundoff}} .
$$

The first term falls as $h$ falls. The second rises, because there are more terms and more additions. Differentiating and setting to zero puts the minimum where the two are comparable:

$$
h^{*} \sim \left(\frac{D\epsilon}{C}\right)^{1/(p+1)} .
$$

For trapezoid with $p=2$ that is $\epsilon^{1/3}$, around $10^{-5}$ in double precision. For Simpson with $p=4$, $\epsilon^{1/5}$, around $10^{-3}$. These are not tiny grids. Refine past them and the error goes back up.

| Interval width $h$ | Dominant term | Total error |
| --- | --- | --- |
| Large | Truncation | Falls as $h$ shrinks |
| Near $h^{*}$ | Comparable | Minimum |
| Small | Roundoff | Rises as $h$ shrinks |

That table is the argument in three rows, and the exam version has only the first. A finer grid can be less accurate than a coarser one, and the degradation is silent — no warning, no exception. The answer simply drifts.

This is why the summation loop matters: accumulating $\sum f(x_i)$ naively over millions of points accumulates rounding linearly in the worst case, and pairwise summation fixes it almost for free. The difference shows up only at high resolution, exactly when you are relying on refinement.

The diagnostic is simple. Compute at $n$, $2n$, $4n$ and compare differences. If they shrink at the predicted ratio you are in the truncation regime. If they stop shrinking or start growing, you have hit roundoff.

## Monte Carlo, and the different shape of its error

Everything above assumes a grid. Monte Carlo abandons it and estimates

$$
I \approx \frac{V}{N}\sum_{i=1}^{N} f(X_i),
$$

with $X_i$ drawn uniformly from a domain of volume $V$. The standard error is $\sigma_f/\sqrt{N}$ — a rate of $N^{-1/2}$ regardless of dimension or smoothness.

A product grid in $d$ dimensions uses $N = m^d$ points and achieves $O(N^{-p/d})$. With Simpson's rule in one dimension that exponent is four, and Monte Carlo loses badly; in four dimensions it is one; in ten it is $0.4$, and the grid is worse than sampling. That is the honest case for Monte Carlo: a dimensional argument, not a claim that random beats deterministic.

Non-smoothness is the second case. If the integrand is an indicator — a digital payoff, a constraint check, a classification decision — every grid rule degrades to first order or worse because derivative bounds fail everywhere. Monte Carlo does not degrade, because it never differentiated anything.

The cost is that $N^{-1/2}$ is unforgiving: one more decimal digit costs a hundred times the samples. Variance reduction — importance sampling, antithetic variates, control variates — is not decoration here, it is how the method becomes usable. Quasi-Monte Carlo recovers close to $N^{-1}$ for smooth integrands in moderate dimension, though its error estimates are harder to trust.

## The contrarian point: exactness is the wrong goal

Calculus courses teach integration as a search for the exact answer. The integral either has a closed form, which you find, or it does not, and you are told it cannot be expressed in elementary terms — at which point the topic ends. Numerical integration appears, if at all, as an afterthought: an approximation, a lesser result standing in for the real one.

That framing installs two habits, and both are wrong.

The first is that accuracy is monotone in effort. If the goal is to approach the exact answer and more points means closer, then the rational strategy is always to add points until the budget runs out. This is precisely the habit that leads people to refine past $h^{*}$ and report a worse number with more confidence. The framing offers no vocabulary for an optimum, because in a world where error only shrinks, no optimum exists.

The second is that error is a single quantity. In the exam framing it is truncation error, bounded by a formula you can evaluate, and there is no second error — so a competing error that grows with effort, and turns the problem from "maximize effort" to "locate an interior optimum", is absent entirely.

What this hides is the engineering question: given a fixed budget of evaluations and a required accuracy, what is the cheapest rule that meets it, and how do I know it did? That has no exact answer, only a procedure. Pick a rule whose order suits the smoothness you believe the integrand has. Compute at increasing resolution. Watch the differences. Stop when they stabilize or begin to grow, and report where you stopped so the number is auditable.

That requires a belief about smoothness, which is a modelling judgment; a convergence check, which costs extra evaluations; and reporting the resolution alongside the answer. An integral quoted to twelve digits with no stated grid is not more precise than one quoted to six digits with a stated grid — it is less informative, because the reader cannot tell whether the last six digits are signal or roundoff.

The right mental model is not "approximation of the exact value" but "a number with a stated error budget". The exact integral is a limit, not a target; credibility comes from the convergence evidence attached to the number, not from how many digits were printed.

There is a further consequence. If exactness is the goal, the highest-order rule is strictly better and you should always use it. If an error budget is the goal, the highest-order rule is frequently wrong: it assumes smoothness and fails worse when the assumption breaks. A second-order rule degrades gracefully; a higher-order one can drop below second order on an integrand with a kink, and it will still look converged on the panels that do not contain the kink.

I think the subject should be taught as optimization under two competing errors, with the smoothness assumption explicit and the convergence check part of the method. That is a more useful lesson than any particular rule.

## Sources

- Trefethen, L. N. and Bau, D. (1997). *Numerical Linear Algebra*. SIAM. https://people.maths.ox.ac.uk/trefethen/text.html
- Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys*. https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
