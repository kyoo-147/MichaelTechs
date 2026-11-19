---
title: "Condition Number Decides Whether Float16 Survives"
description: "Most low-precision instability is not an optimizer problem. It is a layer whose condition number times the format's mantissa has already exceeded one."
pubDatetime: "2023-07-09T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Floating Point
  - Conditioning
  - Numerical Stability
categories:
  - Math
---

A training run diverges. The first thing anyone tries is a smaller learning rate, then a different optimizer, then more warmup. Sometimes one of those helps. Often the actual cause is a layer whose computation is badly conditioned, running in a format that cannot represent the result - and no schedule repairs that.

## Conditioning is a ratio

$$
\kappa(A) = \|A\|\,\|A^{-1}\|
$$

For $A\mathbf{x} = \mathbf{b}$, a perturbation in the right-hand side propagates as

$$
\frac{\|\delta \mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(A)\,\frac{\|\delta \mathbf{b}\|}{\|\mathbf{b}\|}
$$

plus a second-order term. The word that matters is *relative*. A layer can carry large weights and be perfectly conditioned; a layer with modest weights and a huge condition number will still lose the answer. Conditioning is scale-free, which is exactly why it is the right quantity, and why "the activations are large" is not.

In the 2-norm, $\kappa(A) = \sigma_{\max}/\sigma_{\min}$, so the singular-value picture carries straight over. When the ratio is one, every direction is treated the same and relative error in equals relative error out. When it is $10^6$, some directions are amplified a million times more than others, and error poured into the quiet ones arrives at the output at full volume.

## Cancellation is the mechanism underneath

Goldberg's account of floating point makes the point that subtraction is where digits die. Subtract two nearly equal numbers and the leading digits cancel; what survives is the part that was already rounding error. The absolute error of the result stays small, and the relative error can be enormous, because the result is small exactly when the inputs were close.

The textbook examples are $b^2 - 4ac$ when the terms nearly agree, and $1 - \cos x$ for small $x$. Both exact in the reals, both catastrophic in floating point. The same pattern appears inside a layer whenever a large common part is subtracted out: a mean removal, a centred difference, a residual branch sitting close to its input. It is invisible if you only track absolute magnitudes, which is why the failure looks like noise rather than a wrong number.

## What the formats give you

| format | exponent bits | mantissa bits | spacing at 1.0 | decimal digits |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | $2^{-23}\approx1.2\times10^{-7}$ | ~7 |
| fp16 | 5 | 10 | $2^{-10}\approx9.8\times10^{-4}$ | ~3 |
| bf16 | 8 | 7 | $2^{-7}\approx7.8\times10^{-3}$ | ~2 |

Decimal digits are roughly $p\log_{10}2$ for a $p$-bit mantissa. fp16 gives about three, bf16 about two - fewer, not more, which surprises people who assume the newer format is the wider one. bf16 buys something else: it keeps fp32's exponent range, so it does not overflow where fp32 would not. It pays for that entirely in mantissa. fp16 has the mantissa but a narrow exponent range, so it also underflows and overflows where bf16 survives. Neither is simply better; they fail differently.

## The product that decides the outcome

A backward-stable solve of $A\mathbf{x}=\mathbf{b}$ carries a forward relative error of order

$$
\kappa(A)\,2^{-p}
$$

for a mantissa width $p$. Read it as a budget. fp16 gives a spacing near $10^{-3}$. A layer with $\kappa \approx 10$ spends about $10^{-2}$ of its relative accuracy - coarse, but answerable. A layer with $\kappa \approx 10^{4}$ spends about $10$, so the bound exceeds one and the computed answer has no valid digits at all. In fp32 the same layer spends about $10^{-3}$: uncomfortable, survivable.

That is the whole interaction in one expression. Low precision does not break a well-conditioned layer. It removes the digits that a badly conditioned layer was already close to exhausting. The layer did not become bad when you cast it; it was already bad, and fp32 was hiding it.

## Why normalisation and residual scaling are numerics

Normalisation sits in an awkward spot: it subtracts a mean and divides by a spread, which is a cancellation site by construction. That is why those statistics are usually accumulated in the wide format even when the arithmetic around them is narrow. Done that way, the layer keeps its input in a band where the operator's gain does not have to span many decades.

Residual connections help for a different reason. The identity path has condition number one, so composing many of them does not multiply the condition number by depth. The branch only has to represent a correction, and if the branch output is scaled down before the sum, the conditioning of the sum stays near that of the identity path. That is a numerical argument, not only an optimisation one, and it explains why residual scaling shows up in exactly the configurations where low-precision training is otherwise fragile.

## The optimizer is probably not what broke

The knobs a divergence makes visible are the ones the optimizer exposes, so those get blamed first. But the gradient is computed through the same linear maps as the forward pass. If the forward map is losing digits, the gradient is noise, and no step-size schedule turns noise into a direction.

Two signatures are worth learning. Divergence that survives an order-of-magnitude reduction in step size is not a step-size problem. A blow-up confined to particular tensors, rather than a global rise in the loss, is a locality problem - and conditioning is local.

Now the honest caveat. The bound above is worst-case: it holds for the perturbation the operator amplifies most, over all inputs. A neural network is not linear, its inputs are not adversarial, and the nonlinearity frequently compresses the very directions the bound worries about. So $\kappa(A)2^{-p}$ is pessimistic, sometimes very, and it should not be read as a prediction that the loss will spike.

What survives the pessimism is the ordering. Given two hundred layers and one reduced format, the bound tells me which layers are closest to running out of digits, and that ranking is usually right. That is a better starting point than guessing at the optimizer, and it costs one singular-value computation per suspect layer instead of a week of tuning.

## Turning the bound into a measurement

The test is small. Run the suspect layer in the wide format on inputs from the real distribution, run it again in the narrow format, and look at the relative disagreement. If the disagreement tracks the singular-value ratio rather than sitting at the roundoff floor, the diagnosis is conditioning and the fix belongs in the layer. If it sits at the floor, the layer is fine in that format and the instability is somewhere else - which is also worth knowing before touching the learning rate.

## The solve can be worse than the problem

The bound has been read so far as a property of a layer. It is a property of a computation, and the same matrix solved two ways carries two different amplification factors.

Least squares makes the gap visible. Minimising $\|A\mathbf{x}-\mathbf{b}\|$ by QR or by the SVD costs a factor $\kappa(A)$, plus a term that grows with the residual. Forming the normal equations $A^\top A\mathbf{x}=A^\top\mathbf{b}$ and solving those costs $\kappa(A^\top A)$, and for full-rank $A$

$$
\kappa(A^\top A)=\kappa(A)^2 .
$$

The singular values of $A^\top A$ are the squares of the singular values of $A$, so the ratio between the largest and the smallest is squared with them. A number that looked survivable becomes its own square.

Take a case where every quantity is exact. With

$$
A=\begin{pmatrix}16.5 & 15.5\\ 15.5 & 16.5\end{pmatrix},
$$

the eigenvalues are $16.5+15.5=32$ and $16.5-15.5=1$, so $\kappa(A)=32$. Squaring,

$$
A^\top A=\begin{pmatrix}512.5 & 511.5\\ 511.5 & 512.5\end{pmatrix},
$$

whose eigenvalues are $1024$ and $1$, so $\kappa(A^\top A)=1024=2^{10}$, exactly.

Spend the budget three ways. Directly in fp16, $32\times2^{-10}=2^{-5}\approx0.031$, about a digit and a half. Normal equations in fp32, $2^{10}\times2^{-23}=2^{-13}\approx1.2\times10^{-4}$, about four digits. Normal equations in fp16, $2^{10}\times2^{-10}=1$ exactly, which is the bound sitting at the point where the answer has no valid digits. The data did not change between those three. The largest entry of $A^\top A$ is $512.5$, and fp16 spacing is $0.5$ at $512$ and $0.25$ just below it, so every entry is stored exactly. The format holds the problem and loses the solution.

Two things follow. A rewrite that is an algebraic no-op can move the cost from $\kappa$ to $\kappa^2$, so the format decision attaches to the algorithm, not to the layer's name. And "this block ran in fp16" and "this block diverges in fp16" can both be true, because they were not the same computation. A covariance matrix is the case to watch: it is formed as $X^\top X$ after centring, so it inherits the square, and a layer that normalises by its inverse starts spending the squared budget before any arithmetic of its own runs.

Refinement is the cheapest check on which cost you are paying, and Higham's chapter on least squares is the place to follow it. Solve, compute the residual $\mathbf{b}-A\mathbf{x}$ in the wide format, solve for the correction, repeat. Each pass contracts the error by roughly $\kappa(A^\top A)\,2^{-p}$, so at $\kappa=1024$ in fp16 the factor is one and the iteration stalls at whatever the first solve produced, while in fp32 the same pass contracts by $1.2\times10^{-4}$ and one extra solve is enough. The wide residual is what makes it work: computed in fp16, the residual of a nearly converged solve is mostly its own rounding error.

The bound stays worst case and stays pessimistic. The ratio between the two costs does not: it is exactly $\kappa(A)$.

## Sources

- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, SIAM - [people.maths.ox.ac.uk/trefethen](https://people.maths.ox.ac.uk/trefethen/text.html).
- David Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), Oracle.
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM - [maths.manchester.ac.uk/~higham/asna](https://www.maths.manchester.ac.uk/~higham/asna/).
