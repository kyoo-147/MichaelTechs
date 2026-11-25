---
title: "Gradients and Jacobians Are the Same Object at Different Shapes"
description: "What a derivative of a vector-valued function actually is, why the Jacobian has the shape it has, and the object autograd returns instead."
pubDatetime: "2023-03-05T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Multivariable Calculus
  - Jacobian
  - Automatic Differentiation
categories:
  - Math
---

A derivative is not a slope. A slope is what you get in one dimension, when the only thing a linear map can do is scale. In general, a derivative is the **best linear approximation** of a function near a point:

$$
f(\mathbf{x} + \Delta) \approx f(\mathbf{x}) + J(\mathbf{x})\,\Delta .
$$

Everything else - gradients, Jacobians, Hessians, the whole vocabulary - is that one sentence with different shapes attached.

## The shape follows from the function

Take $f:\mathbb{R}^n \to \mathbb{R}^m$. To approximate a small step $\Delta \in \mathbb{R}^n$ with an output of size $m$, the linear object $J$ must eat an $n$-vector and produce an $m$-vector. That forces $J \in \mathbb{R}^{m \times n}$:

$$
J = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[6pt]
\vdots & \ddots & \vdots \\[6pt]
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}.
$$

Row $i$ is the gradient of the $i$-th output; column $j$ is how the whole output moves when only $x_j$ moves. Nothing about this is a convention - it is the only shape that makes the product $J\Delta$ type-check.

For a concrete case, let $f(x_1, x_2) = (x_1 x_2,\; x_1^2 + x_2)$. Then

$$
J = \begin{bmatrix} x_2 & x_1 \\ 2x_1 & 1 \end{bmatrix},
$$

and at $(2, 3)$ a step of $(0.01, -0.02)$ changes the output by roughly $J\,(0.01, -0.02)^\top = (0.03 - 0.04,\; 0.04 - 0.02)^\top = (-0.01, 0.02)^\top$. You can check that against the function itself.

## The gradient is a Jacobian wearing a hat

When $m = 1$ - a loss, typically - the Jacobian is $1 \times n$. That single row is what we call the gradient, and most libraries store it as a vector of length $n$ because a $1 \times n$ matrix is a silly data structure.

This is where the famous layout confusion starts. Two conventions are in circulation:

- Numerator layout: $\nabla f$ is a row, $\partial f/\partial \mathbf{x} \in \mathbb{R}^{1 \times n}$.
- Denominator layout: $\nabla f$ is a column, $\partial f/\partial \mathbf{x} \in \mathbb{R}^{n}$.

They differ by a transpose, and almost every "shape mismatch in the backward pass" bug is one of these two conventions meeting the other in a dark alley. Pick one and write it down at the top of the file. PyTorch's rule is simple and worth memorising: **the gradient of a tensor has the same shape as the tensor.** That is denominator layout, applied everywhere, including for matrices.

## The chain rule in this language

The reason the Jacobian is worth the trouble is composition. For $y = f(u)$ and $u = g(x)$,

$$
J_{y \circ g}(x) = J_f(g(x)) \; J_g(x),
$$

an $m \times n$ matrix built from an $m \times k$ and a $k \times n$. Eric Weisstein states the same identity entrywise on [MathWorld](https://mathworld.wolfram.com/ChainRule.html): $\partial y_i/\partial x_j = \sum_k (\partial y_i/\partial u_k)(\partial u_k/\partial x_j)$.

Parr and Howard's [matrix calculus paper](https://explained.ai/matrix-calculus/) is the reference I keep returning to here, because it derives this instead of asserting it, and because it is blunt that the scalar case people quote as "the chain rule" is a special case of this composition.

## The part the textbooks leave out: autograd never builds the Jacobian

Here is the gap between the mathematics and the software. The mathematics says the derivative *is* an $m \times n$ matrix of partials. A library like PyTorch or JAX, in the common case where the output is a scalar loss, never materialises that matrix. It computes a **vector-Jacobian product** instead:

$$
\mathbf{v}^\top J \quad \text{or} \quad J^\top \mathbf{v},
$$

depending on which side you push from. One backward pass answers "how does the loss move when the last layer's output moves" and keeps folding that vector backwards, never forming the full matrix. That is the entire reason training a large model is affordable: the full Jacobian of a loss with respect to a billion parameters would be a one-by-billion matrix, which is fine - but the Jacobian of an *intermediate* activation with respect to another intermediate activation is quadratic, and nobody builds those.

Two practical consequences follow, and I have hit both:

1. Broadcasting silently changes the derivative's shape. If a tensor of shape `(1, 128)` is broadcast against `(32, 128)`, the correct backward pass sums the gradient over the broadcast axis. Frameworks do this automatically; a hand-written backward pass that forgets it produces shapes that happen to broadcast again, and the bug hides until the loss plateaus.
2. Shape-checking the backward pass is a real test. If your manual gradient has the wrong shape and you reshape it into place with `.view()` or a transpose "to make it work", you have almost certainly transposed a Jacobian convention rather than fixed a bug.

## A rule I use

When a gradient looks wrong, I do not re-derive the maths first. I check three things, in this order:

- does the gradient have exactly the shape of the input it belongs to;
- is there a reduction in the forward pass (sum, mean, broadcast) whose axis is missing in the backward pass;
- is a variable used twice on the forward pass, and if so, is its gradient accumulated with `+=` rather than overwritten?

Only after those three does the calculus become the suspect. In my experience the chain rule is almost never the bug; the bookkeeping around it usually is - which is the subject of the next note.

## The determinant, when it exists at all

There is one thing you can ask of a Jacobian that you cannot ask of a gradient: its determinant. That is not a remark about optimisation; it is a remark about shapes. A determinant needs a square matrix, so it exists only when $m = n$. A loss has $m = 1$, so it never has one, and "the Jacobian of the loss" nearly always means the matrix rather than its determinant.

When the shapes do match, the determinant earns its place, because the single number $\det J(\mathbf{x})$ measures how the map rescales volume near $\mathbf{x}$. Polar coordinates are the example most people meet before anyone names the Jacobian. For $p(r,\theta) = (r\cos\theta,\; r\sin\theta)$,

$$
J = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix},
\qquad
\det J = r\cos^2\theta + r\sin^2\theta = r .
$$

At $(r, \theta) = (2, \pi/4)$ the columns are $(\sqrt2/2, \sqrt2/2)$ and $(-\sqrt2, \sqrt2)$, and the determinant is exactly $2$: a small patch of the $r\theta$-plane with area $\varepsilon^2$ lands in the plane with area about $2\varepsilon^2$. The $r$ in $r\,dr\,d\theta$ is this determinant, not a notational habit. The same factor is why a normalising flow tracks a Jacobian determinant: its log-density needs $\log|\det J|$, and the common designs keep $J$ triangular so that the determinant is a product of diagonal entries.

The determinant belongs to the point, not to the map, and that distinction gets lost. For

$$
g(x,y) = (x^2 y,\; x y^2),
\qquad
J = \begin{bmatrix} 2xy & x^2 \\ y^2 & 2xy \end{bmatrix},
\qquad
\det J = 4x^2y^2 - x^2y^2 = 3x^2y^2 .
$$

At $(2,3)$ that is $3 \cdot 4 \cdot 9 = 108$, so a square at that point with side $10^{-3}$ and area $10^{-6}$ is carried to a shape of area about $1.08 \times 10^{-4}$. At $(4,6)$ the same square scales by $1728$, sixteen times more, because doubling both coordinates multiplies $3x^2y^2$ by $2^4$.

It is worth being exact about what 108 is, since numbers like this get quoted as though the map were linear. It is the derivative's answer, not the map's. The true area is $108\varepsilon^2$ plus a correction of order $\varepsilon^3$, so at $\varepsilon = 10^{-3}$ the correction is invisible, and at $\varepsilon = 1$ the true area is about $234$ against an estimate of $108$. A linearisation is exact only in the limit.

The determinant also reports when the linear approximation discards a direction. A zero determinant is not proof that the map itself is non-invertible, since $x \mapsto x^3$ is the standing counterexample, but it does mean the approximation is not invertible, and a Newton step solving $J\Delta = -F$ has a division by zero waiting. For $g$ the collapse is blunt: the whole line $y = 0$ maps to $(0,0)$. MathWorld's entry on the [Jacobian](https://mathworld.wolfram.com/Jacobian.html) states the volume fact in the same form, as the ratio of $n$-dimensional volumes, and points to the change-of-variables theorem.

## Sources

- Terence Parr and Jeremy Howard, [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/), 2018.
- Stanford CS231n, [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) (MIT-licensed course notes).
- Eric W. Weisstein, [Chain Rule](https://mathworld.wolfram.com/ChainRule.html), MathWorld.
- Eric W. Weisstein, [Jacobian](https://mathworld.wolfram.com/Jacobian.html), MathWorld.
