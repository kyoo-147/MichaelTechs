---
title: "Why Gradient Descent Works When It Shouldn't"
description: "The one inequality behind gradient descent, what convexity actually buys you, and why the optimizer advice you read online ages badly."
pubDatetime: "2023-05-07T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Optimization
  - Gradient Descent
  - Training
categories:
  - Math
---

Gradient descent is suspiciously simple. You have a loss surface in a billion dimensions, you look at the slope under your feet, you step downhill, and somehow the model learns. The explanation usually offered — "we follow the gradient" — is a description, not a reason. There is exactly one inequality that makes it a reason.

## The inequality

Assume $f$ is differentiable and its gradient is $L$-Lipschitz: $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L\|\mathbf{x} - \mathbf{y}\|$. Then for any step $\eta$,

$$
f\!\left(\mathbf{x} - \eta\nabla f(\mathbf{x})\right) \;\le\; f(\mathbf{x}) - \eta\left(1 - \frac{\eta L}{2}\right)\left\|\nabla f(\mathbf{x})\right\|^2 .
$$

Read it as a contract. If $0 < \eta < 2/L$, the bracket is positive, so every step strictly decreases the loss unless the gradient is exactly zero. That is the entire guarantee of gradient descent, and it is local: it says the loss goes down, not that you end up anywhere useful.

Two practical facts fall out immediately, and they explain most beginner pain:

- **Too small a step** and you crawl: the decrease per step is linear in $\eta$.
- **Too large a step** and the bracket goes negative: the "descent" step increases the loss. Divergence is not a mysterious numerical bug; it is the inequality being violated because your step size exceeded the local curvature.

$L$ is the largest curvature in the region you are walking through. This is why learning-rate warmup exists: early in training the effective curvature is large, so you take small steps until the landscape flattens.

## What convexity buys

If $f$ is convex, every local minimum is global, and the guarantee upgrades from "the loss goes down" to "you converge to the optimum" with a rate: $O(1/k)$ for gradient descent on smooth convex functions, $O(1/k^2)$ with momentum, and linear convergence if the function is also strongly convex.

Boyd and Vandenberghe's [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) is the canonical reference for why that class is special: duality, certificate of optimality, polynomial-time interior-point methods. It is a genuinely beautiful theory, and it is the theory I reach for when the problem actually is convex — a linear program, a small logistic regression, a calibration fit.

## What deep learning actually optimizes

Deep networks are not convex. Goodfellow, Bengio and Courville are unusually blunt about the consequence in [chapter 8](https://www.deeplearningbook.org/contents/optimization.html): the classical guarantees do not apply, and the interesting structure is not local minima but saddle points, plateaus, and cliffs. A random point in a high-dimensional non-convex landscape is far more likely to be near a saddle than a minimum, because a saddle needs only *some* directions to curve upward, while a minimum needs all of them.

This is the first place the standard narrative misleads. Students are taught that the danger is "getting stuck in a local minimum", and then spend time on a problem that is rarely the binding constraint. The binding constraints are conditioning (a ravine that is steep in one direction and flat in another), noise in the gradient estimate, and step-size schedules.

## Where stochasticity earns its keep

Full-batch gradient descent uses the exact gradient, which costs a pass over the data. Stochastic gradient descent uses a minibatch estimate:

$$
\mathbf{g}_t = \frac{1}{|B|}\sum_{i \in B} \nabla f_i(\mathbf{x}_t), \qquad \mathbb{E}[\mathbf{g}_t] = \nabla f(\mathbf{x}_t).
$$

The estimate is unbiased but noisy, and the noise is not purely a cost. It perturbs you away from sharp, narrow regions toward flatter ones, which correlates with better generalization — an effect that is still not fully explained and should be described as an empirical regularity rather than a theorem.

Momentum and Adam are best read as **preconditioners**: they rescale the step per coordinate using a running estimate of gradient magnitude, which is exactly what helps on badly conditioned ravines. Sebastian Ruder's [overview](https://ruder.io/optimizing-gradient-descent/) remains the clearest tour of the family, including the equations for momentum, RMSProp, and Adam side by side.

## The second reverse angle: the advice ages

Ruder's post is from 2016–2017, and it ends with the judgement that Adam is often the best default. That judgement was reasonable then and is only partly true now: AdamW decoupled weight decay in 2019 and changed the regularization story, and a long line of vision results found SGD with momentum generalizing better than Adam at equal tuning effort. None of this invalidates the post; it just dates it.

That is the honest lesson about optimizer advice on the internet: it is a snapshot of benchmarks, not a property of mathematics. The mathematical part — the descent inequality, the role of curvature, the unbiasedness of the minibatch gradient — does not age. The "best optimizer" part ages every two years.

## What I do in practice

- Set the learning rate by finding where training diverges, then back off by an order of magnitude; that is an empirical estimate of $2/L$.
- Use warmup for transformers, cosine or linear decay afterwards; schedules matter more than the optimizer family in most of my runs.
- Treat any "best optimizer" claim as a hypothesis to test on my own data, with the same budget for both candidates. Unequal tuning is the most common way these comparisons lie.
- When training is unstable, check curvature-related causes first: large learning rate, no normalization, exploding activations. Then blame the optimizer.

## Sources

- Stephen Boyd and Lieven Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), Cambridge University Press (free PDF).
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, [Deep Learning, ch. 8: Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html), 2016.
- Sebastian Ruder, [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/), 2016.
- Stanford CS231n, [Neural Networks Part 3: Learning and Evaluation](https://cs231n.github.io/neural-networks-3/).
