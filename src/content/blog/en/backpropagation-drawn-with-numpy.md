---
title: "Backpropagation, Drawn: One Graph, Twenty Lines of NumPy"
description: "A two-layer network with a hand-written backward pass, a finite-difference check, and the three bugs that survive the maths."
pubDatetime: "2023-05-28T09:00:00.000Z"
locale: en
author: Michael
featured: true
tags:
  - Backpropagation
  - NumPy
  - Automatic Differentiation
categories:
  - Math
  - AI
---

I understood backpropagation twice. The first time was after reading the derivation and being able to reproduce it on paper. The second time - the one that stuck - was after writing the backward pass by hand and having it disagree with a finite-difference check.

This note is the second version. One small network, forward and backward written out, no framework.

![Forward pass left to right; backward pass carrying gradients right to left.](/assets/blog/backprop-numpy/graph.svg)

## The forward pass

Two layers, ReLU hidden, softmax cross-entropy:

```python
import numpy as np

rng = np.random.default_rng(0)

def init(n_in, n_hidden, n_out):
 return {
 "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
 "b1": np.zeros(n_hidden),
 "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)),
 "b2": np.zeros(n_out),
 }

def forward(params, X):
 z1 = X @ params["W1"] + params["b1"]
 a1 = np.maximum(z1, 0.0)
 logits = a1 @ params["W2"] + params["b2"]
 return z1, a1, logits

def softmax_xent(logits, y):
 shifted = logits - logits.max(axis=1, keepdims=True)
 log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
 n = len(y)
 loss = -log_probs[np.arange(n), y].mean()
 probs = np.exp(log_probs)
 return loss, probs
```

The shift by the row maximum is not cosmetic. $\exp$ of a large logit overflows float64 around 709, and subtracting a constant per row changes nothing mathematically because softmax is shift-invariant.

## The backward pass

For softmax plus cross-entropy the gradient with respect to the logits collapses to something almost embarrassingly clean:

$$
\frac{\partial L}{\partial \mathbf{z}} = \frac{1}{N}\left(\mathbf{p} - \mathbf{y}_{\text{onehot}}\right).
$$

Everything else is the chain rule applied locally:

```python
def backward(params, X, y, cache):
 z1, a1, logits = cache
 n = X.shape[0]

 _, probs = softmax_xent(logits, y)
 dlogits = probs.copy()
 dlogits[np.arange(n), y] -= 1.0
 dlogits /= n # shape (n, n_out)

 dW2 = a1.T @ dlogits # (n_hidden, n_out)
 db2 = dlogits.sum(axis=0) # (n_out,)

 da1 = dlogits @ params["W2"].T # (n, n_hidden)
 dz1 = da1 * (z1 > 0.0) # relu mask
 dW1 = X.T @ dz1 # (n_in, n_hidden)
 db1 = dz1.sum(axis=0) # (n_hidden,)

 return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

Twenty lines, and every one of them is doing one of three jobs: propagate the incoming gradient, apply a local derivative, or reduce over the batch axis.

Three details are worth naming, because each one is a bug I have written:

- **`dlogits = probs.copy()`.** Without the copy, the subtraction mutates `probs` in place, which later code may still need. Cheap copies prevent expensive confusion.
- **`(z1 > 0.0)` is a choice, not a derivative.** ReLU is not differentiable at zero. We pick the subgradient $0$ there, which matches the convention in every framework. It is fine; it is just not mathematics.
- **`dz1 = da1 * mask`, not `da1 @ something`.** The ReLU is elementwise, so its "Jacobian" is diagonal and the product collapses to a mask. Writing it as a matrix product would be correct and absurdly slow.

## The check that makes it trustworthy

Finite differences turn "I think this is right" into "this is right to seven digits":

```python
def check_gradients(params, X, y, eps=1e-6):
 cache = forward(params, X)
 grads = backward(params, X, y, cache)
 worst = 0.0
 for name, p in params.items():
 for idx in np.ndindex(p.shape):
 old = p[idx]
 p[idx] = old + eps
 loss_plus, _ = softmax_xent(forward(params, X)[2], y)
 p[idx] = old - eps
 loss_minus, _ = softmax_xent(forward(params, X)[2], y)
 p[idx] = old
 numeric = (loss_plus - loss_minus) / (2 * eps)
 analytic = grads[name][idx]
 denom = max(1e-8, abs(numeric) + abs(analytic))
 worst = max(worst, abs(numeric - analytic) / denom)
 if abs(numeric - analytic) / denom > 1e-6:
 print(f"mismatch {name}{idx}: {numeric:.6e} vs {analytic:.6e}")
 return worst
```

In float64, for this network, a correct implementation lands below $10^{-6}$ relative error - the run above bottoms out at $5.8\times10^{-7}$. If a single parameter disagrees, the output tells you exactly which one, which is more than a framework will ever tell you.

The central-difference form $(\text{plus} - \text{minus})/2\varepsilon$ matters: the one-sided difference has $O(\varepsilon)$ error, which is roughly the same size as the effect you are trying to measure.

## The three bugs that survive the maths

**One: accumulation at forks.** If a tensor is used twice in the forward pass, its gradient is the sum of both contributions. In this network `a1` is used once, so nothing accumulates. Add a residual connection and the same code silently overwrites. That is why every autograd engine accumulates with `+=` into an existing buffer instead of assigning.

**Two: batch reductions.** `db2 = dlogits.sum(axis=0)` disappears if you forget it, and instead of an error you get a shape that broadcasts. Broadcasting is the most forgiving bug surface ever shipped: wrong shapes usually *work*, at reduced magnitude.

**Three: in-place mutation on the forward pass.** Frameworks that save activations for the backward pass break when you modify those activations afterwards. The error surfaces as a confusing "one of the variables needed for gradient computation has been modified by an inplace operation" much later in the code. Hand-written backprop has the same hazard without the error message.

## What this changed for me

After writing this once, reading framework code stopped being mysterious. `loss.backward()` is the loop above, generalised over a DAG instead of a chain, with each node storing its own local derivative. Optimizer steps are one line each. The rest is bookkeeping - and the bookkeeping is precisely where models fail quietly.

Olah's framing is the one I would keep: gradients are sums over paths, and reverse mode is the trick of factoring that sum at each node. Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is the same idea in about a hundred lines of Python, with an autograd engine attached; if you want to see the general case after this note, that is the next file to read.

## One step, by hand

The code above runs and tells you nothing about magnitudes. Freeze the weights with both biases at zero, add one sample, and the arithmetic becomes concrete.

```python
x  = (2.0, -1.0)
y  = 1
W1 = [[0.5, -1.0], [0.25, 0.75]]
W2 = [[1.0, -0.5], [0.25, 2.0]]
```

Forward: $z_1 = (0.75, -2.75)$, so $a_1 = (0.75, 0)$. The logits are the first row of $W_2$ scaled by $0.75$, namely $(0.75, -0.375)$, giving $p = (0.754915, 0.245085)$ and $L = -\ln 0.245085 = 1.406150$. Backward, $p - y = (0.754915, -0.754915)$, and then

$$
dW_2 = \begin{bmatrix}0.566186 & -0.566186\\ 0 & 0\end{bmatrix},\qquad
dz_1 = (1.132372,\ 0),\qquad
dW_1 = \begin{bmatrix}2.264745 & 0\\ -1.132372 & 0\end{bmatrix}.
$$

There are four things in those numbers that the code does not show.

**The gradient sums to zero.** $0.754915 - 0.754915 = 0$, and this is forced rather than lucky: adding a constant to every logit leaves softmax, and hence the loss, unchanged, so the loss has no slope along the all-ones direction. The bias gradient is pure contrast, with no common mode. One practical consequence: if $db_2$ ever fails to sum to zero, the bug is in the loss, not the ReLU. Nielsen states the same collapse for a softmax layer with log-likelihood cost, and leaves the derivation as an exercise.

**The zeros are structural.** Because $a_1[1] = 0$, the loss does not depend on the second row of $W_2$ at all, so a central difference there returns $0/2\varepsilon$ exactly, not something small. The mask then drives $dz_1[1] = 0$, which zeroes the second column of $W_1$. For this sample the unit is not slow, it is disconnected; tuning its learning rate would be tuning nothing.

**The weight gradient is rank one.** Each row of $dW_1$ is $x_i$ times $dz_1$, so the rows are proportional to the inputs and their ratio is $x_1/x_0 = -0.5$. One sample can push $W_1$ in a single direction; a batch of $N$ samples assembles it from at most $N$ such outer products. It also means an input feature that is zero in every sample leaves its whole row at zero, and that weight never moves, whatever the learning rate.

**The $1/N$ is a convention, not a law.** The loss is a mean, so every gradient here carries a factor of $1/N$: doubling the batch roughly halves the step at a fixed rate, which is why batch size and learning rate get tuned together.

It is arithmetic, not a run: substitute another input and every line recomputes. The finite-difference check agrees with these values by construction, because it estimates the same derivative, up to its own roundoff. What it cannot do is tell you that the loss you differentiated is the loss you meant. It compares backward against forward, so a forward pass that computes the wrong loss makes both sides agree, and the check passes with complete confidence.

## Sources

- Christopher Olah, [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/), 2015.
- Andrej Karpathy, [micrograd](https://github.com/karpathy/micrograd) - a minimal scalar-valued autograd engine.
- Stanford CS231n, [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/).
- Terence Parr and Jeremy Howard, [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/), 2018.
- Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap3.html), 2015 - the softmax chapter, where the bias gradient is shown to be the output error itself.
