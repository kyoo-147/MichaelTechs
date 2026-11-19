---
title: "The Chain Rule Is Not the Hard Part"
description: "Path sums, reverse-mode accumulation, and why the derivative you compute in practice is often a surrogate for one that does not exist."
pubDatetime: "2023-03-26T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Chain Rule
  - Backpropagation
  - Automatic Differentiation
categories:
  - Math
  - AI
---

Every explanation of backpropagation eventually says "and then we apply the chain rule". That sentence is true and almost useless. The chain rule is one line. What makes backpropagation interesting is what you do *with* the line, and what happens when the line does not apply.

## The rule, stated once

Chris Olah's essay on [calculus on computational graphs](https://colah.github.io/posts/2015-08-Backprop/) gives the cleanest statement: on a graph, the derivative of one node with respect to another is the **sum over all paths** between them of the product of the edge derivatives along each path. "This general 'sum over paths' rule is just a different way of thinking about the multivariate chain rule."

That is the whole theorem. For the graph below — three nodes, three edges per hop — it produces nine terms.

![Left: enumerating all nine paths. Right: factoring the sum at each node, which is what reverse mode does.](/assets/blog/math-chain-rule/paths.svg)

Nine terms for three edges per hop is fine. Nine terms becomes $3^{L}$ for $L$ layers, which is not fine for $L = 50$. The fix is not a new theorem; it is factoring the sum:

$$
\sum_{a \in \{\alpha,\beta,\gamma\}} \sum_{b \in \{\delta,\varepsilon,\zeta\}} a\,b
= \left(\alpha + \beta + \gamma\right)\left(\delta + \varepsilon + \zeta\right).
$$

Same value, four additions instead of eight, and the cost is now linear in depth. That factorisation is *all* reverse-mode automatic differentiation is. The cleverness is in noticing that you can compute the inner sums once per node and reuse them, not in the calculus.

## Why the framing still misleads

Three things bother me about the standard telling.

**One: the rule assumes the derivative exists.** The interesting cases in deep learning are exactly the ones where it does not. ReLU has no derivative at zero. `max` has no derivative at a tie — and CS231n's notes write its gradient as $\mathbb{1}[x \geq y]$, where that "$\geq$" is a choice at the tie, not a derivative. Sorting, argmax, quantization, and straight-through estimators all ship backward passes that are gradients of nothing. The framework is happy to run them. No convergence guarantee survives that substitution, and none of the tutorials mention it.

**Two: the rule does not mention accumulation, which is where the bugs are.** If a variable feeds two branches, its gradient is the *sum* of the contributions. Frameworks implement that with `+=` into an existing buffer. A hand-written backward pass that assigns instead of accumulates produces a number that looks plausible, never errors, and is silently wrong whenever a tensor is reused. Half the "my gradient is off by a factor" incidents I have seen are this.

**Three: the mathematical object and the computed object differ.** The chain rule composes Jacobians. Nobody composes Jacobians. What a backward pass actually computes is a vector–Jacobian product, folded one layer at a time, so the $m \times n$ matrix is never built. Parr and Howard present the matrix product because it is the correct statement; the product they write is also precisely the thing your library never forms.

## A worked example, small enough to check

Take $f(x) = \log(1 + e^{x})$ and compose it twice: $y = f(f(x))$. The local derivative is $\sigma(x) = 1/(1+e^{-x})$, so the chain rule gives

$$
\frac{dy}{dx} = \sigma\!\left(f(x)\right)\cdot \sigma(x).
$$

At $x = 0$: $f(0) = \log 2 \approx 0.693$, $\sigma(0) = 0.5$, $\sigma(\log 2) \approx 0.667$, so $dy/dx \approx 0.333$.

You can verify that in four lines of NumPy, and you should, once, for a function whose derivative you can also write by hand. It is the cheapest way to discover that your mental model of the backward pass is off by a term.

## The diamond, worked by hand

The simplest graph where paths actually multiply is a diamond: one input feeding two branches that rejoin.

$$
a = 2x, \qquad b = 5x, \qquad y = 3a + 7b .
$$

**Sum over paths.** There are two paths from $x$ to $y$:

$$
\frac{\partial y}{\partial x}
= \frac{\partial y}{\partial a}\frac{\partial a}{\partial x}
+ \frac{\partial y}{\partial b}\frac{\partial b}{\partial x}
= 3 \cdot 2 + 7 \cdot 5 = 6 + 35 = 41 .
$$

**Substitute and differentiate.** Eliminating the intermediates first gives $y = 3(2x) + 7(5x) = 41x$, and the derivative is $41$. Same number. That equality is not a coincidence; it is the statement that the chain rule agrees with algebra.

**Reverse mode.** Start at $y$ with an incoming gradient of $1$. Pushing through the addition sends $3$ into $a$ and $7$ into $b$. Pushing through the branches multiplies: $3 \cdot 2 = 6$ into $x$ from the left, $7 \cdot 5 = 35$ from the right. Where the two paths meet at $x$, the contributions **add**: $6 + 35 = 41$.

That last step is the whole point of the example. It is also the bug I described above: if the implementation assigns instead of accumulating at $x$, it produces $35$ and the loss quietly drifts. The arithmetic is one line; remembering to add is the engineering.

## What I take from this

The chain rule is the *licence* to compute derivatives locally. Reverse mode is the *algorithm* that makes it cheap. Everything that actually breaks in production sits between those two sentences: non-differentiable points, accumulation semantics, and the convention for which axis is which.

So when someone says they understand backpropagation because they understand the chain rule, I think they understand the licence. The engineering is elsewhere — and the next note is about the twenty lines of NumPy where the engineering becomes visible.

## Sources

- Christopher Olah, [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/), 2015.
- Stanford CS231n, [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/).
- Terence Parr and Jeremy Howard, [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/), 2018.
