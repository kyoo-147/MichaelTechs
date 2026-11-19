---
title: "Taylor Series and Why Curvature Matters"
description: "Taylor expansion turns a function into a local polynomial; the gradient and Hessian explain step size, overshoot, and Newton's method."
pubDatetime: "2023-05-14T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Taylor Series
  - Curvature
  - Optimisation
categories:
  - Math
---

A function is rarely simple enough to reason about all at once. The Taylor expansion gives us a way to replace it, near a chosen point, with a polynomial. The constant term tells us the value. The linear term tells us the slope. The quadratic term tells us the curvature. Everything after that is a correction that matters only when we move far enough for the lower-order model to fail.

That last sentence is the part I care about. In optimisation, the practical question is almost never whether a function equals its Taylor series. It is whether the low-order truncation is still a decent description over the step we are about to take. Curvature is what answers that question.

## The local polynomial

For a smooth function $f:\mathbb{R}^n\to\mathbb{R}$, around a point $x$,

$$f(x+h)=f(x)+\nabla f(x)^\top h+\frac{1}{2}h^\top \nabla^2 f(x)h+o(\|h\|^2).$$

In one dimension this reads

$$f(x+h)=f(x)+f'(x)h+\frac{1}{2}f''(x)h^2+o(h^2).$$

The notation is compact, but it hides a condition. The expansion is local. The remainder vanishes faster than $\|h\|^2$ as $h\to 0$, and its size depends on higher derivatives. A Taylor polynomial can converge on a finite interval and diverge outside it. Even where it converges, a truncation that is accurate for small steps can be misleading for large ones. The polynomial is a model with a domain of validity, not a rewrite of the function.

This matters because every gradient-based method chooses a step by trusting some truncation. Gradient descent trusts the linear term. Newton's method trusts the quadratic term. The difference between them is not just arithmetic. It is a different claim about how far the local model can be pushed.

## The first-order term

The gradient $\nabla f(x)$ collects the partial derivatives. For a unit vector $v$, the directional derivative is $\nabla f(x)^\top v$, so the gradient points in the direction of steepest increase and its negative points in the direction of steepest decrease.

The first-order model is

$$f(x+h)\approx f(x)+\nabla f(x)^\top h.$$

If we minimise this model over a ball $\|h\|\le \epsilon$ with $\nabla f(x)\neq 0$, the solution is $h=-\epsilon \nabla f(x)/\|\nabla f(x)\|$. The same direction appears in the gradient step

$$x_{k+1}=x_k-\alpha \nabla f(x_k),$$

where $\alpha>0$ is a step size. The model predicts a decrease of $\alpha\|\nabla f(x_k)\|^2$. That prediction assumes the gradient is constant along the step. It is not. The gradient changes, and the rate at which it changes is curvature.

A large gradient and a large curvature are different problems. A large gradient means the local slope is steep, so a fixed step size produces a large update. If the curvature is small, that step may still be fine: the slope does not change much, and the linear model remains accurate. A large curvature means the slope changes quickly. A step that looked reasonable under the linear model can land where the gradient points in another direction, or uphill. The gradient gives a direction. Curvature tells you how long that direction remains a good idea.

## The second-order term

The Hessian $H(x)=\nabla^2 f(x)$ is the matrix of second partial derivatives. When $f$ is twice continuously differentiable, $H(x)$ is symmetric, so it has real eigenvalues and an orthonormal basis of eigenvectors. The quadratic model is

$$f(x+h)\approx f(x)+\nabla f(x)^\top h+\frac{1}{2}h^\top H(x)h.$$

If $H(x)$ is positive definite, this model is convex and has a unique minimiser at $h=-H(x)^{-1}\nabla f(x)$. The eigenvalues of $H(x)$ measure the curvature of the quadratic model along its eigenvectors. An eigenvector with a large eigenvalue is a direction in which the gradient changes quickly; a small eigenvalue marks a flat direction. The gradient itself obeys

$$\nabla f(x+h)\approx \nabla f(x)+H(x)h,$$

which is the clearest statement of what the second-order term buys: a prediction of how the gradient will change after the step.

For a positive-definite Hessian, the ratio $\kappa=\lambda_{\max}/\lambda_{\min}$ is the condition number. A large $\kappa$ describes a long, narrow valley. The gradient points across the valley rather than along it, so gradient descent zigzags. The curvature is not a small correction to the first-order picture. It is the part of the picture that determines whether the step is stable.

## Gradient descent and the curvature bound

Gradient descent is usually analysed under an assumption that bounds the curvature. A differentiable function is $L$-smooth if

$$\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$$

for all $x,y$. When $f$ is twice differentiable, this holds if the spectral norm of the Hessian is bounded by $L$. The assumption gives the descent lemma:

$$f(x-\alpha g)\le f(x)-\alpha\left(1-\frac{\alpha L}{2}\right)\|g\|^2, \qquad g=\nabla f(x).$$

For $0<\alpha<2/L$, the right-hand side is below $f(x)$ whenever $g\neq 0$, so the step is guaranteed to decrease the function. The common choice $\alpha=1/L$ maximises the guaranteed decrease in this bound. The step size is not a free parameter chosen for elegance; it is a stand-in for a bound on curvature.

When the bound fails, so does the guarantee. Take the quadratic $f(x)=\frac{1}{2}x^\top Hx$ with positive-definite $H$. Gradient descent gives

$$x_{k+1}=(I-\alpha H)x_k.$$

Along an eigenvector with eigenvalue $\lambda$, the error is multiplied by $1-\alpha\lambda$ at each step. The iteration is stable only if $\alpha\lambda<2$ for every eigenvalue, which means $\alpha<2/\lambda_{\max}$. If $\alpha\lambda_{\max}>2$, the component along that eigenvector changes sign and grows. The step was too large for the curvature, not necessarily for the gradient.

This distinction explains a common confusion. A large gradient produces a large proposed step, but it does not by itself cause overshoot if the curvature is small. A moderate gradient in a high-curvature direction can overshoot badly. The quantity that matters is the curvature along the direction actually taken. For this quadratic, the actual decrease is

$$\alpha\|g\|^2-\frac{1}{2}\alpha^2 g^\top H g,$$

while the first-order prediction is $\alpha\|g\|^2$. The ratio of actual to predicted decrease depends on the Rayleigh quotient $g^\top H g/\|g\|^2$, not on $\|g\|$ alone. A single global step size cannot be right in every direction of an anisotropic problem. Line search and trust-region methods shorten the step when the function does not decrease enough; in effect, they discover local curvature without forming the Hessian.

## Newton's method

Newton's method minimises the quadratic model exactly. At each iterate it computes

$$x_{k+1}=x_k-H(x_k)^{-1}\nabla f(x_k).$$

The step is the minimiser of the local quadratic, so Newton is making a stronger assumption than gradient descent: it assumes the quadratic term is a good enough description over the step. Near a strict local minimum, that assumption is justified. If $H$ is Lipschitz in a neighbourhood of a minimiser $x^*$ and $H(x^*)$ is positive definite, then for $x_0$ close enough to $x^*$,

$$\|x_{k+1}-x^*\|\le C\|x_k-x^*\|^2.$$

This is quadratic convergence: once the iterates are close, the number of correct digits roughly doubles each step. Newton is also affine invariant. If we replace $x$ by $Ax+b$ for an invertible $A$, the Newton iterates transform in the same way. Gradient descent does not have this property; its behaviour depends on the coordinate scaling. In exact arithmetic, that is a strong argument that Newton is the natural local method.

Far from a minimum, the same quadratic model becomes a liability. The Hessian may be indefinite, in which case the Newton direction need not be a descent direction at all. The Hessian may be singular or ill-conditioned, producing an enormous step. The quadratic model may be accurate only in a small neighbourhood, so a full Newton step can jump over the valley it was meant to follow. A saddle point is also a fixed point: if the gradient vanishes and the Hessian is nonsingular, Newton can converge to a saddle rather than a minimum. Damped Newton, trust-region methods, and Hessian modifications address these failures by modifying the step when the model is not trustworthy.

The cost is the other obstacle. Forming the Hessian requires second derivatives. Storing a dense Hessian takes $O(n^2)$ memory, and solving the linear system takes $O(n^3)$ arithmetic. The code is short:

```python
import numpy as np

def gradient_step(x, grad, step_size):
    return x - step_size * grad

def newton_step(x, grad, hessian):
    return x - np.linalg.solve(hessian, grad)
```

The difference between the two functions is not cosmetic. The first uses a scalar step size; the second solves a system that encodes curvature in every direction.

## The contrarian case: first-order methods win on algebra, not on principle

The standard argument for first-order methods is that they are cheaper, so they win at scale. I think that argument is incomplete in a way that changes the conclusion. First-order methods win because the exact second-order object cannot be formed, stored, or factored when the parameter count is large. That is an algebraic and memory constraint. It is not evidence that curvature stops mattering.

The constraint is easy to state. A dense Hessian has $n^2$ entries. Factorising it costs $O(n^3)$ arithmetic. For a model with $10^6$ parameters, the Hessian alone would have $10^{12}$ entries, and the factorisation is out of reach. No amount of hardware changes the exponent. So we approximate. Gradient descent is the crudest approximation: it acts as if the Hessian were a multiple of the identity, with one scalar step size. That is a modelling choice, not a discovery that the matrix is irrelevant.

The bill for that choice appears elsewhere. If the problem is well conditioned, a scalar step size is fine. If it is not, the same step is too large in steep directions and too small in flat ones. The iteration zigzags or crawls, and we respond with step-size schedules, clipping, momentum, or diagonal rescaling. Some of these approximate curvature; others change the effective conditioning of the iteration. Either way, they are responses to the same missing matrix, not evidence that it was irrelevant. They are a budget response to curvature, not a refutation of it.

This is why I find the common claim that first-order methods are cheaper and therefore win at scale misleading. The accurate statement is that exact curvature is too expensive at scale, so we use approximations, and the quality of the approximation determines how much tuning we need. That is a different research question from whether curvature matters. It matters. The practical issue is that we cannot afford the exact answer.

There are genuine reasons to prefer first-order methods beyond the Hessian cost: stochastic gradients, nonsmooth objectives, and memory limits are all real. But those are separate arguments. They should not be confused with the claim that curvature is a minor correction. If curvature were unimportant, step-size tuning would not be the persistent practical problem that it is.

## Sources

- Stephen Boyd and Lieven Vandenberghe (2004). *Convex Optimization*. https://web.stanford.edu/~boyd/cvxbook/
- Lloyd N. Trefethen and David Bau III (1997). *Numerical Linear Algebra*. https://people.maths.ox.ac.uk/trefethen/text.html
