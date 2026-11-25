---
title: "FP8 Training at Scale"
description: "FP8 is limited by scaling strategy, not the format. Where per-tensor breaks, what per-block costs, and which operations survive eight bits."
pubDatetime: "2025-11-08T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - FP8
  - Training
categories:
  - Math
---

The gradient norm doubles in one step and does not come back. The loss follows a few hundred steps later. Cutting the learning rate does not recover the run, and restarting from the last checkpoint reproduces the same failure within a few thousand steps.

Fishman et al. document this shape of failure in their trillion-token FP8 work. They trained to two trillion tokens, twenty times further than earlier FP8 runs, and the instability only appeared on that horizon. The cause they identify is SwiGLU, not the matmul: the activation amplifies outliers, and the amplification is a slow function of training time, tied to the SwiGLU weights aligning. Their fix, Smooth-SwiGLU, changes the parametrisation without changing the function.

That result should reframe how you plan a run. The format rarely fails at the scale you validate on. It fails at the scale you ship.

## What eight bits actually costs

E4M3 spends one sign bit, four exponent bits and three mantissa bits. Between two consecutive powers of two there are eight representable values. Spacing is $2^{-3}$ of the leading power, so the worst relative error for a single stored value is

$$\varepsilon = 2^{-4} = 6.25\%.$$

Better kernels do not move that number, because it is the grid.

The exponent side is more generous than people expect. Normal values run from $2^{-6}$ up to $448$, about $2^{8.8}$, a ratio of $2^{14.8}$, or roughly four and a half decades. A tensor whose elements span less than that fits. A tensor whose elements span more does not, and the elements that fall out do not disappear cleanly. They land in the subnormal grid, where the spacing is fixed at $2^{-9}$ and the relative error grows as the value shrinks.

Now the scaling decision. With one scale per tensor,

$$x_q = \mathrm{fp8}\left(\frac{x}{s}\right), \qquad \hat{x} = s\,x_q, \qquad s = \frac{\max |x|}{448},$$

the largest element receives the full mantissa and every other element is placed relative to it. An element four decades below the maximum sits near the bottom of the normal range. An element five decades below is subnormal, with a relative step near 40 percent. Gradient tensors in transformer training routinely have that shape: a handful of large coordinates and a long tail.

Per-tensor scaling is a bet that the tensor is roughly unimodal. That bet holds for many activations and fails for the ones that matter.

## The standard framing is wrong about granularity

The usual advice is that finer granularity is strictly better, that per-block beats per-tensor, and that microscaling beats per-block. Read the kernels instead of the advice and the tradeoff becomes visible.

Per-block scaling along the reduction dimension puts a division inside the K loop. The scale tensor has to be broadcast across the MMA tile, and on hardware without native block-scaled MMA that broadcast is ALU work sitting between the tensor cores. You buy mantissa and pay for it in issue slots.

```python
def block_scale(x, block=128):
    # x: [..., K], one fp32 scale per 128 contiguous elements along K
    b = x.reshape(*x.shape[:-1], -1, block)
    s = b.abs().amax(dim=-1).clamp_min(1e-12) / 448.0   # 448 is the E4M3 max
    return (b / s.unsqueeze(-1)).to(torch.float8_e4m3fn), s
```

There is second-order evidence in the published work for the same point. Hernández-Cano et al. observe that existing FP8 approaches either rely on fine-grained kernels they call suboptimal or fall back to higher precision on sensitive components, and they list attention projections among those components. That is a field with fine-grained kernels available, choosing not to use them everywhere.

Granularity is also not the only axis. μnit Scaling trains models from 1B to 13B parameters with all hidden linear layer computation in FP8, and it uses no dynamic scaling factors and no special hyperparameters. Its claim is that treating each operation from first principles makes the scale largely predictable, which removes most of the machinery. Whether that transfers to your architecture is an open question. That the result exists is enough to show the granularity ladder is not the only route to stability.

The practical reading is that granularity is a budget. Spend it on the layers with heavy-tailed outliers and leave the rest per-tensor.

## The amax you scale by is already stale

A scale has to come from somewhere. The obvious source is the maximum absolute value of the tensor being quantised, but that maximum does not exist until the tensor does. So systems choose between two costs. Either they compute the amax of the current tensor on the fly, which puts a full reduction on the critical path ahead of the GEMM, or they keep a short history of past amax values and use the largest of those.

History is the cheaper choice and the more common one. It is also wrong by construction on exactly the steps that matter. A scale fitted to earlier iterations clips the current tensor when the distribution shifts upward, and throws away range when it shifts down. Both errors land on the same forward pass, and the pass where a distribution shifts is usually the pass that was already unstable.

The on-the-fly alternative is not free either. An amax over an activation tensor is a reduction over everything the producing kernel just wrote. Unless it is fused into that kernel's epilogue, it becomes an additional read of memory that will be read again moments later.

μnit Scaling is worth reading here because it argues the scale can be derived rather than measured for the operations inside a standard transformer block, and it validates that on models from 1B to 13B. Where the argument holds, the history window and its failure mode both disappear. My own reading is that it holds best for linear layers whose inputs are already normalised, and least for anything whose range is set by the data.

## Where the master weights live

Where the master copy lives decides whether a run converges, and it is arithmetic rather than tuning.

A weight $w$ at magnitude near 1 with a per-step relative update of $10^{-4}$ has to absorb a change of $10^{-4}$. In E4M3 the gap to the next representable value near 1 is $2^{-3}$, and the worst rounding error is half of that, so the update is several hundred times smaller than the smallest change the format can record. Apply it and the weight does not move. The result is a model stuck in a quantisation well, which is a different failure from being slightly less accurate.

So the master copy stays in fp32. Nothing in the published work argues otherwise. What can move down is everything else.

FP8-LM reports that decomposition directly: 8-bit gradients, 8-bit optimizer states, and FP8 distributed training, applied to a 175B model on H100, with 39 percent lower real memory use and 75 percent faster than the BF16 Megatron-LM baseline. Fishman et al. go further and quantise both Adam moments to FP8, which they describe as the first demonstration of that. COAT attacks the same memory from another angle, reshaping the optimizer state distribution so it fits the FP8 range, and reports 1.54 times lower end-to-end memory than BF16 with a 1.43 times speedup.

The pattern across all three is the same. Moments and gradients tolerate FP8 far better than weights do. A moment is a small number with a heavy tail and no absolute meaning attached. A weight has an absolute meaning and a tiny update.

One claim worth correcting. "FP8 halves memory" is usually false end to end. Weights are a fraction of the footprint. Optimizer state is two fp32 tensors per parameter, and activations scale with batch and sequence length. Making the GEMM faster moves a compute term. It does not move the memory term unless you also compress the state, which is what COAT does.

## Which operations tolerate eight bits

| Operation | Precision | Reason |
| --- | --- | --- |
| Linear and MLP GEMMs | FP8 | Long reductions with benign distributions, and the main throughput source |
| Attention QK | Higher | Exponentiated scores amplify small errors |
| Attention PV | FP8 or higher | Depends on the value outlier profile |
| Softmax, LayerNorm, RMSNorm | Higher | Reductions and divisions over full rows |
| Residual adds | Higher | Accumulating small deltas over layers |
| Embedding and output head | Higher | Vocabulary-sized outliers and logit sensitivity |
| Adam moments | FP8 | Verified at scale, heavy tails handled by scaling |
| Master weights | fp32 | Update is far below the FP8 grid |
| Gradient all-reduce | BF16 or scaled FP8 | Cross-rank sums need a shared scale |

The two entries that surprise people are attention and the residual stream. QK scores pass through an exponential, so a relative error of $2^{-4}$ in a score becomes roughly a $2^{-4}$ relative change in the attention weight after normalisation, and those weights are what the whole layer mixes. It is a small effect per head and a systematic one across a layer.

The residual stream is subtler. A residual add is exact, but the values being added are contributions from many layers, and quantising each contribution to $2^{-3}$ spacing means the small ones are lost relative to the running sum whose magnitude grows with depth. Keep the accumulator wide.

## What to watch instead of the loss curve

Loss is a lagging indicator. By the time it moves, the run has already committed to a trajectory. Hernández-Cano et al. make the point that low-precision training needs its own metrics, ones that predict divergence rather than report it.

Three are worth logging per layer. The first is the fraction of elements below the normal threshold after scaling, which tells you how much of the tensor is living in the subnormal grid. The second is the ratio between the maximum and the 99.9th percentile of the abs value, which is a direct measure of how much range the outliers are eating. The third is the update-to-weight ratio, because a ratio that decays toward the FP8 grid spacing is a run that is quietly freezing.

None of these require a reference model, and all three are cheap. The last one I would compute in fp32 from the master copy before the quantised update is applied, since that is where the information still exists.

Then there is the shadow batch. Run a fixed batch through the quantised model and through a bf16 copy of the same weights, and track the relative error of the logits. A single number tells you whether the numerics are drifting, and the drift is monotone when it is real. This costs one extra forward pass per logging interval and it is the cheapest early warning available.

The honest summary is that FP8 at scale is a scaling problem wearing a format costume. Mantissa is fixed at three bits and will stay there. What you control is where the scale is computed, how often it is refreshed, which tensors get fine granularity, and whether the master copy is wide enough to record an update. Get those four right and the format stops being the variable. Get the last one wrong and nothing else matters.

## Sources

Fishman, Maxim, Brian Chmiel, Ron Banner, Daniel Soudry. "Scaling FP8 training to trillion-token LLMs." [arXiv:2409.12517](https://arxiv.org/abs/2409.12517). 2024. https://arxiv.org/abs/2409.12517

Hernández-Cano, Alejandro, Dhia Garbaya, Imanol Schlag, Martin Jaggi. "Towards Fully FP8 GEMM LLM Training at Scale." [arXiv:2505.20524](https://arxiv.org/abs/2505.20524). 2025. https://arxiv.org/abs/2505.20524

Narayan, Saaketh, Abhay Gupta, Mansheej Paul, Davis Blalock. "μnit Scaling: Simple and Scalable FP8 LLM Training." [arXiv:2502.05967](https://arxiv.org/abs/2502.05967). 2025. https://arxiv.org/abs/2502.05967

Peng, Houwen, Kan Wu, Yixuan Wei, et al. "FP8-LM: Training FP8 Large Language Models." [arXiv:2310.18313](https://arxiv.org/abs/2310.18313). 2023. https://arxiv.org/abs/2310.18313

Xi, Haocheng, Han Cai, Ligeng Zhu, Yao Lu, Kurt Keutzer, Jianfei Chen, Song Han. "COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training." [arXiv:2410.19313](https://arxiv.org/abs/2410.19313). 2024. https://arxiv.org/abs/2410.19313
