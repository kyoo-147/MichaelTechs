---
title: "Numeric Formats in 2026"
description: "Block scaling spends bits on a shared exponent so elements keep their mantissa. The error budget per operation decides, not the model."
pubDatetime: "2026-10-24T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Precision
  - Quantisation
categories:
  - Math
---

An MXFP4 matmul holds within tolerance on the calibration batch and drifts on traffic from a different source. The kernel did not change. The weights did not change. What changed is the largest magnitude inside one block of 32 numbers, and every other number in that block now shares the grid that maximum sets.

This is the characteristic failure of block-scaled arithmetic, and it is not a modelling failure. Nothing in a model specification says how much dynamic range thirty-two consecutive activations should carry. The format says it. The format is chosen by whoever owns the datapath, usually before the model exists.

Block scaling trades exponent sharing for mantissa bits. The trade has a price you can write down per operation, and the price is paid where the loss curve does not look.

## Where the exponent goes

A float spends bits on sign, exponent, mantissa. E4M3 spends 1, 4 and 3. Go down to four bits and something has to give. Drop the exponent to 2 and the mantissa to 1 and you have E2M1, whose eight representable magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6. One mantissa bit gives a worst-case relative error of $2^{-2} = 25\%$ against the value's own binade. That is the per-element grid, and no kernel improves it.

What you can do is stop paying for an exponent per element. One exponent per block of $k$ elements costs $b_s/k$ bits per element, where $b_s$ is the width of the shared scale. In the published MX formats the scale is E8M0, a power of two, over $k = 32$, so the scale costs 0.25 bits per element. The accounting is

$$b_{\text{eff}} = 1 + e_{\text{elem}} + m + \frac{b_s}{k}.$$

At 4 bits, 0.25 bits of overhead is a six per cent tax on the budget. It buys no precision. It buys range localisation, which is worth more than it looks. A per-tensor scale has to cover the largest value in the tensor. A per-block scale only has to cover the largest value in the block. That is the premise, and it is about data distribution, not arithmetic.

Hypothesis: the claim that localisation is what makes four-bit inference possible is probably right for activations and wrong for weights. Weight blocks are near-stationary, so the block maximum sits at a fixed multiple of the block standard deviation and a per-tensor scale with good clipping does most of the work. Activations are not stationary.

## What a block actually costs

The cost is not the bits. The cost is that one number sets the grid for thirty-two.

Write $r_i = |x_i| / \max_j |x_j|$ for the relative magnitude of element $i$ inside its block. For block floating point, where elements are fixed point under a pure shared exponent and carry $m$ mantissa bits, the rounding error obeys

$$\frac{|\delta x_i|}{|x_i|} \le \frac{2^{-(m+1)}}{r_i}.$$

An element a factor of 32 below the block maximum carries 32 times the relative error of the largest element. At $m = 1$, an element at $r_i = 2^{-5}$ has a bound of 8, which is to say the bound is useless and the element is noise.

MXFP4 is not block floating point, and the difference matters. E2M1 carries two exponent bits of its own, so an element can step down a binade and keep its single mantissa bit. What it cannot do is step down more than a few. Give a block a shared scale $s$ and the representable magnitudes run from $0.5s$ to $6s$, a ratio of twelve. Any element smaller than one twelfth of the block maximum quantises to zero. That is not a bound, it is an identity about the grid, and it is the most useful number to know about MXFP4.

So MXFP4 handles enormous variation between blocks and almost none inside one. Its dynamic range is not a property of the format. It is a property of the block, and the block is 32 elements.

NVFP4 changes two things at once. The block shrinks to 16, so the maximum sits closer to the typical element, and the shared scale becomes E4M3, which is not restricted to powers of two. It also spends more: 4 bits for the element, 0.5 for the per-block scale, plus a per-tensor FP32 term that is negligible at tensor scale. Call it 4.5 bits against MXFP4's 4.25. The extra quarter bit buys a scale with three mantissa bits and a block half as wide.

## The error budget for one multiply-accumulate

For a dot product $y = \sum_i a_i b_i$ with both operands quantised, write $u_a = 2^{-(m_a+1)}$ and $u_b = 2^{-(m_b+1)}$ for the per-element rounding units. Then

$$\frac{|\delta y|}{|y|} \lesssim (u_a + u_b)\, \kappa, \qquad \kappa = \frac{\sum_i |a_i b_i|}{\left| \sum_i a_i b_i \right|}.$$

The format gives you $u$. The data gives you $\kappa$. Neither alone tells you whether a layer survives, which is why per-layer sensitivity profiles work as a bit-allocation signal. A layer with heavy cancellation has $\kappa \gg 1$ and needs more mantissa bits. A layer whose products all point the same way has $\kappa \approx 1$ and tolerates almost anything.

$\kappa$ is cheap to estimate. Run one forward pass in higher precision, record $\sum |a_i b_i|$ and $|\sum_i a_i b_i|$ per matmul, and you have the amplification factor without a gradient, a Hessian or a calibration set.

Across $L$ layers the bookkeeping is

$$\varepsilon_{\text{total}} \lesssim \sum_{\ell=1}^{L} \varepsilon_\ell, \qquad \text{or} \qquad \varepsilon_{\text{total}} \approx \sqrt{\sum_{\ell=1}^{L} \varepsilon_\ell^2}$$

under an independence assumption. The first is a bound. The second is a model of the error, and models of the error are what people quote. My opinion is that the square-root form flatters block-scaled formats, because format error correlates with activation magnitude, and activation magnitude correlates across layers. When a block overflows the grid in layer 12, it is often because the same token produced a large value in layer 11.

## Why two formats with the same bits behave differently

| Format | Element | Block | Shared scale | Bits per element | Worst-case relative error |
|---|---|---|---|---|---|
| MXFP8 | E4M3 | 32 | E8M0, power of two | 8.25 | $6.25\%$ |
| MXFP6 | E3M2 | 32 | E8M0, power of two | 6.25 | $12.5\%$ |
| MXFP4 | E2M1 | 32 | E8M0, power of two | 4.25 | $25\%$ |
| NVFP4 | E2M1 | 16 | E4M3, plus per-tensor FP32 | 4.5 | $25\%$ |

Read the last row against the third. Same element. Same worst-case relative error. 0.25 more bits. The difference is a smaller block and a richer scale, not expressiveness in the modelling sense.

Shadowing the table is a kernel consequence no accuracy column shows. A power-of-two scale is applied by adjusting the exponent, which is arithmetic the datapath already has. An arbitrary scale costs a real multiply per element in the decode path. Dequantisation sits between the load and the multiply-accumulate, so block size also decides how much of the tensor must be resident before the inner loop starts. A block of 32 along the reduction axis is a tile constraint as much as a numerical one.

M2XFP is the clearest statement of this I have read this year. Hu and coauthors accept that the power-of-two shared scale is where MXFP4 loses accuracy, then recover it with a small amount of metadata rather than with more mantissa bits, and pair the encoding with a hardware unit so the decode stays cheap. They report an average 70.63% reduction in accuracy loss against MXFP4 and 37.30% relative to NVFP4, with up to 1.91x speedup and 1.75x energy savings over the accelerators they compare against ([arXiv:2601.19213](https://arxiv.org/abs/2601.19213), Hu et al., 2026).

That is not a claim that the format is more expressive. It is a claim that the format recovers accuracy per unit of decode cost. Those are different objectives, and only the second one is measured in joules.

## The contrarian case: the format leaderboard measures an interaction

Zhang and coauthors benchmark post-training quantisation under MXFP across more than seven algorithms, fifteen benchmarks and three model families ([arXiv:2601.09555](https://arxiv.org/abs/2601.09555), Zhang et al., 2026). Two findings matter here. MXFP8 is close to lossless while MXFP4 degrades substantially. And the effectiveness of a quantisation algorithm under MXFP depends strongly on format compatibility, with some algorithmic paradigms consistently better than others.

The second finding should change how you read every format comparison you see. If format and algorithm interact, then a table that fixes the algorithm and varies the format measures an interaction term rather than a main effect. A table that lets each format use its own best algorithm measures the ecosystem around the format, which is a real thing to measure but not the thing most readers think they are reading. Either way the rankings are conditional. You cannot evaluate a format without committing to a quantisation algorithm first, and you cannot evaluate the algorithm without committing to a block size, because the block size is what the algorithm works against.

The fourth finding is sharper. The quantisation scale is a critical error source in MXFP4, and a simple pre-scale optimisation substantially mitigates it. That gives a specific engineering order: fix the scale before you change the format. If a pre-scale correction moves the number, the format comparison was partly measuring a scaling bug.

My own position, stated as a position rather than a result. Most teams unhappy with four-bit inference on MXFP4 do not have a format problem. They have a scale-estimation problem, a per-tensor fallback nobody noticed, and a calibration set that does not resemble production traffic. Moving to a richer format fixes the symptom on the evaluation and leaves the scale problem in place, waiting for the next distribution shift.

## Choosing a format is a deployment decision

The HAMP-LIC paper is not about language models, and that is why it is instructive. Zhang allocates bit widths per block under a global model-size constraint, using Hessian-trace sensitivity to decide where the bits go, and reports up to 4.85x compression with as little as 0.59% BD-rate loss while eliminating cross-platform encoding and decoding mismatches ([arXiv:2608.12239](https://arxiv.org/abs/2608.12239), Zhang, 2026).

Both constraints in that sentence are deployment constraints. A model-size budget is a memory number. Cross-platform mismatch is what happens when the same compressed bitstream is decoded by two different kernels, which is a correctness problem inherited from a hardware fleet rather than from a model. A quantisation scheme that removes it is doing systems work under a systems constraint.

The same structure holds for language models. A format choice fixes four things at once: the arithmetic the tensor core implements, the decode instruction count in the inner loop, the block granularity the scheduler must respect, and the accuracy you can claim in a paper. Only the last is a modelling quantity. The other three are properties of the machine you will run on for the next three years.

So the order I would follow is not format first. Estimate $\kappa$ per matmul, which costs one forward pass. Fix the scale computation and re-measure, because that term is first order and the benchmarking result above says so. Cost the block size in kernel terms. Then choose bits, per layer, against a size budget rather than an accuracy target. The format falls out as an implementation detail, which is the correct status for it.

A format is an interface between a model that specifies real numbers and a datapath that cannot represent them. Treating the choice as a modelling question is how you end up arguing about expressiveness while the scale estimate is wrong by a factor of four.

## Sources

Hu, Weiming, Zihan Zhang, Haoyan Zhang, et al. "M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization." [arXiv:2601.19213](https://arxiv.org/abs/2601.19213). 2026. https://arxiv.org/abs/2601.19213

Zhang, Manyi, Ji-Fu Li, Zhongao Sun, et al. "Benchmarking Post-Training Quantization of Large Language Models under Microscaling Floating Point Formats." [arXiv:2601.09555](https://arxiv.org/abs/2601.09555). 2026. https://arxiv.org/abs/2601.09555

Zhang, Yuefeng. "HAMP-LIC: Hessian-Aware Mixed-Precision Post-Training Quantization for Learned Image Compression." [arXiv:2608.12239](https://arxiv.org/abs/2608.12239). 2026. https://arxiv.org/abs/2608.12239
