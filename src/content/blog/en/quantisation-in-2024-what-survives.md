---
title: "Quantisation in 2024: What Survives the Rounding"
description: "The layer-wise view of post-training quantisation explains both why low-bit models work and why memory savings rarely become latency savings."
pubDatetime: "2024-08-10T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Quantisation
  - Post-Training
categories:
  - Math
---

Every model that has to run on hardware you own starts with the same budget: how many bytes the weights take. Seven billion parameters at sixteen bits is fourteen gigabytes before a single token of KV cache exists. A 12 GB card does not hold it. A 24 GB card holds it and leaves roughly ten for everything else. Moving the weights to four bits brings the number to about three and a half gigabytes, and the model still answers questions.

Then you run the workload you actually care about and something is off. A cell in a table changes. An instruction gets half followed. A refusal turns into a confident non-answer. Perplexity moved by a tenth of a point and told you nothing.

The useful way to think about that gap is as a regression problem solved one layer at a time, with a rounding decision at each step, rather than as compression.

## Quantisation is a per-layer regression

Post-training quantisation does not touch gradients. It takes a finished checkpoint, runs a few calibration sequences through the model, and for each linear layer picks the rounded weight matrix that best preserves that layer's output on those sequences.

Write $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ for the fp16 weight and $X \in \mathbb{R}^{d_{\text{in}} \times n}$ for the calibration inputs that layer actually sees, stacked as columns. The objective is

$$\mathcal{L}(\hat{W}) = \lVert WX - \hat{W}X \rVert_F^2 .$$

Set $E = \hat{W} - W$. Then

$$\mathcal{L} = \operatorname{tr}(EX X^\top E^\top) = \operatorname{tr}(E H E^\top), \qquad H = XX^\top ,$$

and because the trace runs over output rows, the rows of $E$ decouple completely. Each row is an independent problem:

$$\min_{\hat{w}} \; (w - \hat{w})^\top H (w - \hat{w}).$$

Three things follow, and they are the whole of the layer-wise view.

First, $H$ is $d_{\text{in}} \times d_{\text{in}}$ and it is fixed by the calibration set, not by the weights. A layer with 4096 inputs gives a 4096 by 4096 matrix, which is why practical methods work on a damped Cholesky factor of it rather than on $H$ itself.

Second, if $H$ were the identity the objective would collapse to the squared Euclidean norm and independent nearest rounding would be exactly optimal. Everything interesting lives in the off-diagonal entries. Correlated input channels make rounding errors partially cancel, or compound.

Third, the search space is discrete and enormous. A 4096-input row has $16^{4096}$ candidate assignments at four bits. The methods differ almost entirely in how they walk that space.

The framing is not specific to language models. COMQ minimises this same per-layer error by coordinate-wise descent over scales and integer codes, using only dot products and rounding, and reports negligible loss at 4-bit on vision transformers and convolutional networks. The objective is generic. What changes with LLMs is the scale of the rows and the hostility of the activation distributions.

## Why the curvature changes which weight you round

Nearest rounding treats every coordinate as equally expensive to get wrong. Curvature says otherwise. If two input channels are strongly correlated, moving one weight up by a step and the other down by a step costs almost nothing in output error. Independently rounding both to nearest can cost a great deal.

This is the machinery GPTQ introduced and later work refined. Process the row left to right. Quantise one coordinate, then push the error it created into the coordinates not yet quantised, using the inverse Hessian as the transport:

$$
\delta_j = -\,\frac{w^{(q)}_j - w_j}{[H^{-1}]_{jj}}\,[H^{-1}]_{:,j}.
$$

Later rounding decisions are then made against a target already corrected for earlier mistakes, which is why one greedy pass beats a vectorised round-to-nearest by more than the final difference in reconstruction error would suggest.

CDQuant pushes on the same idea from a different angle. Instead of a fixed left-to-right order, it uses greedy coordinate descent, choosing at each step the coordinate whose quantisation increases the layer-wise reconstruction loss least. The authors report it consistently beating GPTQ in the two to four bit weight regime, and improving other methods when dropped in as a replacement for their GPTQ stage.

That second result is more informative than the headline numbers: if a better optimiser transfers across methods, the objective is doing real work and the headroom sits in the search procedure.

My reading, a hypothesis rather than a measured result: reconstruction loss is a first-order picture of how far a layer's output moves, and output movement is only a proxy for task accuracy. The proxy holds when the layer is close to linear over its input distribution and weakens when the layer has to resolve a decision boundary. That would explain why two methods with nearly identical reconstruction error can separate on a downstream benchmark.

## Outliers, and what mixed precision buys

Weight-only quantisation is the easy case. Weights are static and roughly bell-shaped per channel. Activation quantisation is the hard case, and the reason has a name.

A small number of input channels carry values one or two orders of magnitude above the rest. A per-tensor scale has to cover the largest value, so the step size is set by the outlier:

$$s = \frac{\max_i |x_i|}{2^{b-1}-1}.$$

With a 50 in one channel, 1 nearly everywhere else, and $b = 8$, the step is about $0.39$ and the bulk of the distribution lands on two or three levels out of 255. The quantiser is not badly tuned. It is being asked to cover a range that almost none of the data occupies.

Mixed precision is the usual answer: keep the outlier channels wider and everything else narrow. APTQ allocates with the Hessian trace as a sensitivity metric and adds an attention-level term, on the argument that attention outputs propagate nonlinearity a per-layer view misses. SliM-LLM allocates at group granularity rather than per channel, precisely so the result still maps onto a hardware kernel instead of degenerating into irregular gather.

Both are real gains, and both cost something the bit-width headline leaves out.

## The contrarian case: the outlier map is not a property of the model

The standard framing: large language models have activation outliers in specific channels. Run a calibration pass, find those channels, protect them with wider precision. The map is treated as a fact about the checkpoint.

I think that framing is incomplete in a way that bites in deployment. The outliers depend on the input distribution, not only on the weights, and the cleanest evidence for it is a method that removes them rather than accommodating them.

CushionCache does that. The authors search for a short key-value prefix that minimises the maximum activation in subsequent tokens, then tune that cache so later activations sit in a range the quantiser can handle. With the prefix in place, per-tensor W8A8 activation quantisation improves substantially, and the method composes with existing activation quantisation work. The paper is explicit that the usual channel-mixing remedies add overhead and consume the speedup they are supposed to buy.

If a handful of prefix tokens can move the outlier structure, then the outlier map was never a fixed property of the model. It was a property of the model plus whatever context it was given. The practical consequence is uncomfortable: an outlier map measured on one calibration set is a fixture of that calibration set. Deploy against prompts with a different shape and the protected channels are the wrong ones. That is one route by which a quantised model passes its calibration evaluation and disappoints on real traffic.

I would not overstate this. CushionCache reports it on their models and benchmarks, and per-tensor activation quantisation is not what most production stacks ship today. The narrower claim is the one I trust. Activation statistics are context-dependent, so any scheme that hard-codes them has taken on a distribution assumption it has not tested.

## Memory savings are not latency savings

Here the arithmetic and the wall clock diverge, and most of the disappointment lives in the gap.

Weight storage scales linearly in bits per parameter:

| Precision | Bytes per parameter | Weights, 7B params | Relative weight traffic per token |
|---|---|---|---|
| fp16 | 2 | 14.0 GB | 1.00 |
| int8 | 1 | 7.0 GB | 0.50 |
| int4 | 0.5 | 3.5 GB | 0.25 |
| int3 | 0.375 | 2.6 GB | 0.19 |

Those are divisions, not measurements. Whether any of it becomes time depends on which side of the roofline you are on.

Prefill is compute bound. You push a few thousand tokens through at once, the matmuls have real dimensions, and the cost is arithmetic. A weight-only kernel that dequantises to fp16 before the multiply performs the same number of multiply-accumulates at the same precision. You saved cache footprint and some bandwidth, not FLOPs, and time per prefill token barely moves.

Decode is bandwidth bound. One token at a time, each weight read once, arithmetic intensity close to one operation per byte. Time per token is roughly the bytes that must move divided by achievable bandwidth, so cutting weight bytes by four is close to a four-fold cut in weight traffic. That is the regime where weight-only quantisation pays, and it is the regime most serving workloads live in.

Two things erode it. The KV cache does not shrink when you quantise the weights, and at long context it becomes the dominant term. WKVQuant exists because quantising weights while leaving the cache in fp16 leaves most of the memory unclaimed, and its framing is that covering both approaches what weight-activation quantisation saves while holding weight-only quality. And dequantisation is not free. Every four-bit weight has to be widened before it can be multiplied, and if the kernel cannot keep that step inside the memory pipeline the traffic saving leaks out as compute.

SliM-LLM states its own result carefully. The authors claim close to a six-fold memory reduction at two bits and, in the same breath, that GPU inference speed is maintained. Maintained, not improved. That is the honest accounting for a weight-only method.

## What survives

The layer-wise reconstruction view survives, because it is the cheapest correct statement you can make about what rounding does. It yields a scalar you can compute per layer without a forward pass over a benchmark, and it correlates well enough with damage to be worth optimising.

Curvature survives as the thing that makes the objective non-trivial. Without off-diagonal structure in $XX^\top$ there is no decision to make beyond rounding each weight where it falls, and every method here would collapse into the same algorithm.

The distinction between bytes and time survives, and I would rank it above the others. Almost every inflated expectation about quantisation comes from reading a memory number as a speed number.

What does not survive is the bit-width headline as a summary of anything. Four bits describes a container. It does not describe a kernel, a calibration set, an outlier policy, or the context length at which the KV cache takes over. A checkpoint described only by its average bit-width has told you what it weighs and nothing about how fast it runs or what it will get wrong.

## Sources

- Pranav Ajit Nair et al. "CDQuant: Greedy Coordinate Descent for Accurate LLM Quantization." [arXiv:2406.17542](https://arxiv.org/abs/2406.17542), 2024. https://arxiv.org/abs/2406.17542
- Ziyi Guan et al. "APTQ: Attention-aware Post-Training Mixed-Precision Quantization for Large Language Models." [arXiv:2402.14866](https://arxiv.org/abs/2402.14866), 2024. https://arxiv.org/abs/2402.14866
- Seungwoo Son et al. "Prefixing Attention Sinks can Mitigate Activation Outliers for Large Language Model Quantization." [arXiv:2406.12016](https://arxiv.org/abs/2406.12016), 2024. https://arxiv.org/abs/2406.12016
- Wei Huang et al. "SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models." [arXiv:2405.14917](https://arxiv.org/abs/2405.14917), 2024. https://arxiv.org/abs/2405.14917
- Yuxuan Yue et al. "WKVQuant: Quantizing Weight and Key/Value Cache for Large Language Models Gains More." [arXiv:2402.12065](https://arxiv.org/abs/2402.12065), 2024. https://arxiv.org/abs/2402.12065
- Aozhong Zhang et al. "COMQ: A Backpropagation-Free Algorithm for Post-Training Quantization." [arXiv:2403.07134](https://arxiv.org/abs/2403.07134), 2024. https://arxiv.org/abs/2403.07134
