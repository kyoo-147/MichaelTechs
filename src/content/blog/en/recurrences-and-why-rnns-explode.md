---
title: "Recurrences, Eigenvalues, and Why RNNs Explode"
description: "A linear recurrence unrolls into powers of a matrix; here is what the spectrum does and does not tell you about exploding gradients."
pubDatetime: "2023-09-17T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Recurrence
  - Eigenvalues
  - Recurrent Networks
categories:
  - Math
---

## Unrolling the recurrence

Start with the simplest object that deserves the name. A linear recurrence with an input term:

$$h_t = W h_{t-1} + x_t$$

Setting $h_0 = 0$ only keeps the algebra clean. Then $h_1 = x_1$, $h_2 = W x_1 + x_2$, $h_3 = W^2 x_1 + W x_2 + x_3$, and in general

$$h_t = \sum_{k=1}^{t} W^{\,t-k} x_k .$$

The state is a weighted sum of everything that came before, and the weights are powers of a single matrix. That is the whole story in one line: questions about memory, stability, and the size of gradients are questions about what $W^n$ does as $n$ grows. The rest of this article is about which scalar summarises $W^n$, and when that scalar lies.

## The spectrum tells the truth for normal matrices

Suppose $W$ is diagonalisable, $W = V\Lambda V^{-1}$. Then $W^n = V \Lambda^n V^{-1}$, so each eigenmode evolves as $\lambda_i^n$ and the spectral radius $\rho(W) = \max_i |\lambda_i|$ is the asymptotic growth rate. The bound

$$\|W^n\|_2 \le \kappa(V)\,\rho(W)^n, \qquad \kappa(V) = \|V\|_2 \|V^{-1}\|_2$$

is where the fine print lives. If $W$ is normal, meaning $W W^\top = W^\top W$, the eigenvectors can be chosen orthonormal, so $\kappa(V) = 1$ and the inequality becomes an equality:

$$\|W^n\|_2 = \rho(W)^n .$$

For a normal matrix the spectral radius is not an asymptotic comfort, it is the norm of every power. This is the case in which the textbook sentence is exactly right: $\rho(W) < 1$ means every power contracts, and the recurrence forgets its input at a geometric rate. When people say that the eigenvalues control the dynamics, this is the picture in their head.

## In general, the singular values decide

For a non-normal $W$ the eigenvalues control only the limit. The quantity that controls size at a finite horizon is the largest singular value $\sigma_{\max}(W) = \|W\|_2$, and the two are related by

$$\rho(W) \le \sigma_{\max}(W),$$

an inequality that can be arbitrarily loose. A matrix can have every eigenvalue at $0.9$ and a norm of ten. It can also have every eigenvalue at zero and a norm of a hundred, which is exactly what a nilpotent Jordan block does. Nothing about the eigenvalues tells you the norm.

| Diagnostic | Governs | Reliable when |
| --- | --- | --- |
| $\rho(W)$ | rate of growth as $n \to \infty$ | $W$ is normal, or the horizon is long |
| $\sigma_{\max}(W)$ | amplification of a single step | always, as an upper bound |
| $\sigma_{\max}(W^n)$ | amplification over exactly $n$ steps | always, exactly |

Only the third row answers the question you actually have while training a model, and it is the one nobody plots.

## The gradient is a product of Jacobians

Now put a nonlinearity in. With $h_t = \phi(W h_{t-1} + U x_t + b)$ and a loss $L = \sum_t \ell_t(h_t)$, backpropagation through time gives

$$\frac{\partial L}{\partial h_t} = \sum_{s \ge t} \left( \prod_{k=t+1}^{s} J_k \right)^{\!\top} \frac{\partial \ell_s}{\partial h_s}, \qquad J_k = \operatorname{diag}\!\left(\phi'(a_k)\right) W .$$

The inner object is a product of $s-t$ matrices, one per step of the chain. Because the norm is submultiplicative,

$$\left\| \prod_{k} J_k \right\|_2 \le \prod_k \|J_k\|_2 \le \prod_k \left( \max_i |\phi'(a_{k,i})| \right) \sigma_{\max}(W).$$

The sum matters as much as the product. Every term covers a different span, so the total gradient mixes contributions of very different sizes, and the largest one sets the scale of the update. That asymmetry is why explosion announces itself in the loss while vanishing never does.

There are two factors per step: how much the nonlinearity lets through, and how much $W$ stretches. If the product of the two sits below one on average, the gradient from step $s$ back to step $t$ shrinks geometrically in $s-t$ and the distant past contributes nothing measurable. If it sits above one, the gradient grows geometrically and the update is dominated by whichever steps happened to be loud. Note which norm appears in the bound. It is the singular value, not the spectral radius, and it appears because a product of matrices is not a power of a matrix.

## Saturation, gating, and clipping

The vanishing side barely needs $W$. For $\tanh$, $\phi'(z) = 1 - \tanh^2 z \le 1$, with equality only at the origin, and the derivative decays like $e^{-2|z|}$ in the tails. For the logistic sigmoid, $\phi'(z) = \phi(z)(1-\phi(z)) \le 1/4$. A driven recurrent network spends most of its time away from the origin, so most factors are already well below one before any matrix is applied. Stack a few dozen of them and the product is gone. It is small long before that. The vanishing gradient is the default behaviour of any product of contractive factors, not an accident of a badly conditioned $W$.

Gating is the response, and it is best understood as surgery on the Jacobian rather than as a new architecture. In an LSTM the cell state obeys

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t,$$

and the Jacobian of $c_t$ with respect to $c_{t-1}$, along the memory path alone, is $\operatorname{diag}(f_t)$. If the forget gate sits near one, that block is near the identity, so a gradient travelling along it is neither amplified nor crushed. The network does not become normal or well-conditioned; one path is simply given an eigenvalue close to one on purpose. Initialising the forget-gate bias so that $f_t$ starts near one is a standard trick for precisely this reason.

Explosion is the other failure, and it is easier to see. It shows up as a loss that runs to infinity or to NaN, and the historically standard fix is to clip the gradient by its norm: if $\|g\| > c$, replace $g$ by $c\,g/\|g\|$. Clipping preserves direction and throws away magnitude, which is exactly the information you would want to know how bad the problem is. It is also, in effect, a trust region with no curvature model: loud steps are shortened, quiet ones are left alone. It is blunt, it introduces a threshold to tune, and it is honest. It does not pretend to repair the recurrence; it keeps the step from leaving the region where the model is defined. Pascanu, Mikolov and Bengio made it a standard part of the recurrent toolkit, and it is still the first thing I reach for. Explosion is loud. Vanishing is silent: the model simply fails to learn long dependencies, which looks like a data problem rather than a numerical one.

## The contrarian case: the spectral radius is the wrong diagnostic

The sentence I was taught, and the sentence that shows up in most courses, is that a recurrence is stable when the eigenvalues of $W$ lie inside the unit circle. For normal matrices that is a theorem. For general matrices it is not, and recurrent weight matrices have no reason to be normal. Nothing in the parameterisation, the initialisation, or the loss pushes them there.

Here is the smallest counterexample I know. Take

$$W = \begin{pmatrix} 0.9 & 10 \\ 0 & 0.9 \end{pmatrix}.$$

Both eigenvalues are $0.9$, so $\rho(W) = 0.9 < 1$ and the stable verdict is immediate. But

$$W^t = \begin{pmatrix} 0.9^t & 10\,t\,0.9^{\,t-1} \\ 0 & 0.9^t \end{pmatrix},$$

so the off-diagonal entry grows linearly before the exponential takes over, peaking around $t = 10$, while $\sigma_{\max}(W) \approx 10.1$. A perturbation injected in the first coordinate is amplified by an order of magnitude in the second long before it decays. The spectral radius is not wrong about the limit. It is answering a question about $n \to \infty$ when the training signal lives at $n = 10$ or $n = 50$.

An obvious objection is that recurrent models are often initialised with orthogonal or identity matrices, and for those the spectral radius does tell the truth. Orthogonal matrices are normal, so $\sigma_{\max} = \rho = 1$ and the two diagnostics agree. But that agreement is a property of the initial point, not of the model. Once training moves the weights off that manifold, non-normality returns and the eigenvalues stop describing the map.

Two practical consequences follow. First, measure $\sigma_{\max}(W)$, not $\rho(W)$: one singular value decomposition per matrix, no training loop required, and it is the quantity that appears in the bound on the Jacobian product. Second, watch the gradient norm itself, and the fraction of steps that trigger clipping. That fraction is a cheap and honest explosion detector, and it needs no theory at all. On the vanishing side neither scalar helps much; the product of Jacobian norms along the chain is the object of interest, and the only reliable fix is architectural.

There is a sharper version of the complaint. For a non-normal matrix the eigenvalues are ill-conditioned functions of the entries: a perturbation of size $\epsilon$ can move them by $\kappa(V)\epsilon$, and $\kappa(V)$ can be enormous. A training run that monitors the spectral radius is therefore monitoring a number that is both discontinuous in the parameters and absent from the bound that governs the gradient. It is not a conservative proxy for the right quantity. It is a different quantity, and for the case at hand it is the wrong one.

## Sources

- Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning*, chapter 10, 2016. https://www.deeplearningbook.org/contents/rnn.html
- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, 1997. https://people.maths.ox.ac.uk/trefethen/text.html
