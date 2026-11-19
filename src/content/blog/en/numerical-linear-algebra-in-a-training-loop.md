---
title: "Numerical Linear Algebra in a Training Loop"
description: "What actually costs time and precision inside one training step: arithmetic intensity, accumulation order, and the precision choices that paper FLOP counts ignore."
pubDatetime: "2026-11-16T09:30:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Numerical Linear Algebra
  - Performance
  - Training
categories:
  - Math
---

A training step is a short list of linear algebra: matrix multiplies, a softmax, a normalisation, an optimiser update. The FLOP count of that list is easy to write down and almost useless for predicting how long it takes. The gap between the two is where most performance work lives, and it is also where most numerical surprises live.

## A matmul is not limited by multiplication

Multiplying an $m \times k$ by a $k \times n$ matrix costs $O(mkn)$ floating-point operations. On paper that number is the cost. On hardware it is an upper bound that is rarely reached, because the multiply-add is fast and moving the operands is not.

The ratio that matters is arithmetic intensity: operations per byte moved. A small matmul streams its inputs from memory, does very little arithmetic per element, and waits. A large one reuses each loaded value many times, and the arithmetic finally dominates. Batching, tiling, and larger hidden dimensions all work by raising that ratio, which is why throughput improves with batch size until memory runs out.

Trefethen and Bau's book is about the mathematics of these operations — stability, conditioning, the cost of factorisations — and the hardware story sits underneath it. Both matter. A method that is numerically stable and memory-hostile is still slow; a method that is fast and unstable is worse than slow.

## Accumulation order is a numerical decision

Floating-point addition is not associative. $(a + b) + c$ and $a + (b + c)$ can differ in the last bits, and in a long reduction the differences accumulate. A GPU does not sum a vector in the order you wrote it; it splits the work across lanes and combines partial sums, and the order depends on the shape and the kernel.

Most of the time this is invisible. It becomes visible when the summands differ in magnitude by many orders, which is exactly the case when you add a small gradient to a large activation, or sum a long tail of probabilities. Goldberg's article is the canonical account of why: each addition rounds to the nearest representable value, and the error of a naive summation grows with the number of terms while a compensated or pairwise sum keeps it near constant.

The practical consequence is that identical code can produce slightly different losses on different hardware. That is not a bug in your model; it is the reduction tree. If a result must be bit-reproducible, you have to pin the order, and that is usually slower than accepting the noise.

## Precision is a budget, not a switch

The reason mixed precision works is that different parts of a step tolerate different amounts of error. Matrix multiplies dominate the FLOPs and tolerate reduced precision well. Accumulations, normalisations and the optimiser state tolerate it badly, which is why they stay in fp32 while the multiplies run in fp16 or bf16.

| where | typical precision | why |
| --- | --- | --- |
| matmul inputs | fp16 / bf16 | Voluminous, error-tolerant |
| matmul accumulate | fp32 | Long reduction, error accumulates |
| normalisation statistics | fp32 | Cancellation site (mean subtraction) |
| optimiser state | fp32 | Small increments on large values |
| loss scaling | fp16 only | Keeps small gradients representable |

Loss scaling exists for a specific reason: fp16's exponent range is narrow, so small gradients underflow to zero. Multiplying the loss by a constant shifts every gradient up into the representable band, and dividing it back out afterwards changes nothing mathematically. bf16 keeps fp32's exponent range and does not need the trick — it pays in mantissa instead.

## Gradient accumulation is an approximation

When a batch does not fit, the standard recipe is to accumulate gradients over micro-batches and step once. Mathematically this approximates a larger batch. Numerically it is not identical: the accumulation happens in the parameter's dtype, the order differs, and if the accumulation buffer is low precision the small contributions get rounded away as the buffer grows.

That last point is easy to miss. If you accumulate many micro-batches into an fp16 buffer, later contributions are added to a value that already dominates, and the effective gradient becomes the early micro-batches. Keeping the accumulation buffer in fp32 costs memory and buys correctness — a trade I would take every time.

## The framing breaks here: FLOPs do not predict speed

Papers report training cost in FLOPs, and readers compare those numbers as if they were times. They are not, for three reasons.

First, most layers are memory-bound, not compute-bound. Moving activations between high-bandwidth memory and the compute units is the bottleneck, and a kernel that does more arithmetic per byte is faster even with the same nominal FLOP count.

Second, kernels differ. Two implementations of the same operation can differ by a large factor depending on tiling, fusion, and whether the intermediate is written out at all. A fused softmax and a separate exp/sum/divide are the same mathematics and not the same speed.

Third, precision changes both accuracy and throughput, and the paper's FLOP count is precision-agnostic. Running in bf16 does not halve the FLOPs in any meaningful comparison; it changes the units.

So when I estimate whether a run is feasible, I start from memory traffic and measured kernel throughput on the target hardware, not from a FLOP table. The table is for sanity checks on asymptotics.

## What I actually check

- Is the bottleneck arithmetic or bandwidth? The answer changes what to optimise.
- Is the accumulation buffer wide enough for the number of micro-batches?
- Do the normalisation statistics stay in fp32?
- Is the reported number bit-reproducible, or is it a reduction-order artefact?
- Does a change in batch size change the result beyond noise? If yes, look at accumulation, not at the model.

None of these questions appears in the mathematics of the layer. All of them appear in the training loop.

## Sources

- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, SIAM — [people.maths.ox.ac.uk/trefethen](https://people.maths.ox.ac.uk/trefethen/text.html).
- David Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), Oracle.
