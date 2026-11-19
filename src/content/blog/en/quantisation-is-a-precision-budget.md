---
title: "Quantisation Is a Precision Budget"
description: "Quantising a tensor means picking a scale, an offset and a grid; the error is half a step, and it is not distributed evenly across layers."
pubDatetime: "2023-11-18T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Quantisation
  - Precision
  - Inference
categories:
  - AI
---

A float32 weight carries twenty-four bits of significand, and it spends them where the number needs them: near $10^{-3}$ it resolves to roughly $10^{-10}$, near $10^{3}$ it resolves to roughly $10^{-4}$. An int8 weight carries eight bits in total and spends them on a fixed grid that someone had to choose in advance. Going from one to the other is not compression in the sense that a zip file is compression, because nothing is restored later. You decide, once, with a calibration set and an hour of arithmetic, where the 256 points in the grid go, and every weight and activation the model computes afterwards passes through that decision.

That is why I think of quantisation as a budget rather than a format. The interesting questions are not "how many bits" but "which values get the fine grid" and "which layers can afford the coarse one".

## A tensor becomes a grid

Uniform affine quantisation is two numbers and a round:

$$ x_q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max}\right), \qquad \hat{x} = s\,(x_q - z). $$

The scale $s$ is the width of one step. The zero-point $z$ shifts the grid so that a particular real value, usually zero, lands exactly on an integer code. For symmetric weights, $z = 0$ and the grid is $s \cdot \{-127, \dots, 127\}$; for activations after a ReLU, where zero is meaningful and must stay meaningful, an asymmetric grid with a nonzero $z$ costs one extra term in every dot product and buys exact representation of the padding and the dead units.

Two choices are hidden in those two numbers. The first is the range: you can let the scale be set by the largest magnitude in the tensor, or by a high percentile of it, and clip the rest. The second is the granularity: whether $s$ and $z$ are shared by the whole tensor, by each output channel, or by each small block along the contraction dimension. Everything else in the field is downstream of those.

## The error is half a step

Round-to-nearest on a uniform grid cannot be wrong by more than half a step:

$$ |\hat{x} - x| \le \frac{s}{2}, $$

and often much less, with a mean square error of about $s^2/12$ for values spread across the interval. Absolute error is therefore constant everywhere in the range, which sounds fair and is not, because the quantity that matters is relative error:

$$ \frac{|\hat{x} - x|}{|x|} \le \frac{s}{2|x|}. $$

At $|x| = 2s$ that bound is 25 percent. At $|x| = 100s$ it is half a percent. A float keeps roughly constant relative precision across many decades; an int8 keeps constant absolute precision across one interval and progressively destroys everything below it. Values smaller than the step are not merely noisy, they are gone, and a weight of $0.003$ that rounds to zero stops contributing to the sum at all.

This is the trade in its bluntest form. The grid is uniform, so relative precision is bought only by using the range well. What decides the range?

## Why one outlier sets the scale for everything

If $s = \max |x| / 127$, a single large value widens every step in the tensor. Take a signal of variance $\sigma^2$ whose peak is $k$ times its RMS. Quantisation noise has power $s^2/12$; substituting $s = k\sigma/2^{b-1}$ gives

$$ \mathrm{SQNR} \;\approx\; 6.02\,b + 4.77 - 20\log_{10} k \;\;\text{dB}. $$

The first two terms are the familiar six decibels per bit. The third is the tax. A tensor whose peak is twenty times its RMS pays $20\log_{10}20 \approx 26$ dB, which is about 4.3 bits of the eight you thought you had. The bits did not disappear; they were spent covering a range that almost no value occupies.

The standard fix is to stop sharing one scale. Per-channel quantisation gives each output channel its own $s_j$, so a row with a loud weight does not coarsen the rows around it. Per-group quantisation goes further and uses one scale per block of 32, 64 or 128 contiguous weights along the input dimension, which is about as fine as the stored scales can go before the metadata outweighs them.

```python
def quantise_groups(w, group=64, qmax=127):
    # w: (out_features, in_features), one scale per block of in_features
    blocks = w.reshape(w.shape[0], -1, group)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = (blocks / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale
```

The trade is explicit: smaller groups shrink $k$ inside each group, and each group pays a few bytes for its scale plus a kernel that handles the extra multiply; below about thirty-two elements the metadata and instruction overhead usually dominate.

## Weights and activations are not the same problem

Weights are static. You know them before the request arrives, you can run them through a calibration set, and you can choose the scale that minimises reconstruction error rather than the scale that merely contains the maximum. You can even compensate: quantise a column, compute the residual it introduces in the layer output, and push that residual into the next column before quantising it. Layer-wise post-training methods along these lines are what made four-bit weight quantisation credible, and the error they chase is the one that actually matters — the difference in the layer's output, not the difference in its parameters.

Activations are dynamic. Their range depends on the token, the batch, and the prompt, and a scale chosen from a calibration set is a guess about inputs you have not seen. Per-tensor activation scales are therefore the weakest link in a naive scheme, and per-token scales help but cost a reduction before every matmul.

There is also a distinction that gets lost in product copy. Weight-only quantisation keeps the activations in sixteen bits, so the multiply-accumulate still runs in floating point: you save memory capacity and memory bandwidth, and the arithmetic is untouched. Quantising both operands puts integer kernels on the critical path, which is where the throughput comes from, but it is also where the accuracy risk doubles. When a vendor says "int8", the first question is which of these two they mean, because the answer changes both the speedup and the failure mode.

## A few channels, and the mixed-precision escape

The LLM.int8() work reported a structure that per-tensor scaling cannot survive: a small number of feature dimensions take on magnitudes far outside the rest of the distribution, and the pattern becomes more pronounced as models grow. If a few dozen hidden dimensions are an order of magnitude louder than the rest, then either the scale accommodates them and the remaining thousands pay about three bits for the privilege, or the scale ignores them and they clip.

Their answer is to split the matmul by column rather than to choose a better average:

$$ XW \;=\; X_{\text{out}}W_{\text{out}} \;+\; X_{\text{int8}}W_{\text{int8}}, $$

where the first term carries the handful of outlier columns in sixteen-bit precision and the second carries everything else in int8 with a thirty-two-bit accumulator. Because the outlier set is small, the expensive term is cheap; because the remaining columns are well behaved, the cheap term is accurate. It is not a cleverer rounding rule. It is an explicit statement that the tensor contains two populations with different precision needs, and a refusal to average them.

## The contrarian case: bits are not the unit of loss

The standard framing is that quantisation gives you the same model in less memory, with some accuracy loss you can measure on a benchmark. I think the first half of that sentence is wrong and the second half is misleading, and both errors come from treating bits as the unit of accounting.

Quantisation does not compress a fixed object. It replaces the function the model computes with a different function, one whose arithmetic has been reparametrised, and calibration data is part of the definition of that new function. Two int8 checkpoints of the same weights, calibrated on different data, are different functions with the same file size. That is a stronger statement than "slight accuracy loss" and it changes what you should measure.

The second error is uniformity. A drop in average perplexity is an average over tokens, and tokens are wildly unequal. Most of the loss lands on the bulk of easy predictions, which have large margins and absorb perturbation without changing rank. The predictions that matter — the ones where the top two logits are close, which is where a model's behaviour actually lives — have small margins, and there the same perturbation flips the answer. A model can lose a fraction of a point of perplexity and lose a capability, because the capability was concentrated in the thin part of the distribution that the average hides.

Layer by layer, the same argument applies with more force. For $y = Wx$,

$$ \frac{\|\delta y\|}{\|y\|} \;\le\; \kappa(W)\,\frac{\|\delta x\|}{\|x\|}, $$

with $\kappa$ the ratio of largest to smallest singular value. Two layers given identical quantisation error on their inputs produce output errors that differ by however much their spectra differ, and the residual stream decides what survives: a perturbation added where the layer update is small relative to the residual is damped, and one added where the update is large is not. Uniform bit-width is a uniform allocation of bits. It is not a uniform allocation of error, and error is the thing that costs you accuracy. This is precisely why keeping the first and last layers wide, allocating bits per layer by measured sensitivity, and fine-tuning through the quantiser all work — each of them spends precision where the margin is thin instead of spreading it evenly.

So the question I ask of a quantised checkpoint is not how many bits it has. It is where the error budget was spent, and whether anyone checked the layers with the least slack. In practice that means injecting the real quantisation error rather than simulating it by zeroing bits, reading the per-layer change in output rather than only the change in loss, and testing the tasks where two answers were nearly tied.

The number of bits is a budget. Where you spend it is the engineering.

## Sources

- David Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", 1991. https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
- Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022. https://arxiv.org/abs/2208.07339
