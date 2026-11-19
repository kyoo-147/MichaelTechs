---
title: "Diffusion Language Models"
description: "Diffusion language models swap left-to-right factorisation for iterative denoising, and discrete token states are where the clean theory strains."
pubDatetime: "2025-09-06T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Diffusion
  - Language Models
categories:
  - Math
---

A thousand-token answer costs a thousand forward passes. That is the arithmetic the serving stack is built around: one token per call, each call reading the weights and the cache, each waiting on the one before. The arithmetic per token is small. The number of sequential steps is not, and at batch size one much of the wall clock is spent between launches.

There is a way to break the equality. Instead of committing token $l$ before computing $l+1$, corrupt the whole sequence and refine every position at once. The number of evaluations becomes a choice rather than a consequence of length.

For continuous data the theory is clean. Text has discrete states, and that is where it stops being clean.

## The left-to-right contract

Autoregressive generation is a factorisation before it is an architecture:

$$p_\theta(x_{1:L}) = \prod_{l=1}^{L} p_\theta(x_l \mid x_{<l}).$$

Almost everything in a serving stack follows from that line. The factorisation is exact and normalised, so one pass over a sequence yields its exact log probability and perplexity is a defined number. Because conditionals run left to right, a prefix cache is valid: state for computed positions never changes, so each new token is one query against a stored prefix. Speculative decoding rests on the same property.

The bill is in the same line. Dependency depth is $L$. Position $l+1$ cannot start until the distribution at $l$ is resolved and fed back. Where throughput grew faster than latency, depth binds, and batching does not remove it for one user waiting on one answer.

## Corrupting the sequence and learning to walk back

Diffusion keeps the sequence and replaces the factorisation with a process. A forward corruption turns a clean sequence $x_0$ into progressively less structured $x_t$, and a network learns to invert one piece of it. Continuous data allows

$$q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)I\right),$$

which is convenient twice: the posterior $q(x_s \mid x_t, x_0)$ is Gaussian and closed form, and the regression target can be written as the noise that was added. Sampling runs the recursion backwards.

Two things change. The model is never asked for $p(x)$, only for one conditional step; the generative model is the composition. And evaluations are decoupled from tokens: $L$ tokens in $T \ll L$ steps is less sequential work, and each step sees the whole sequence rather than a prefix.

Likelihood pays for it. The exact $\log p_\theta(x_0)$ becomes a bound, because the trajectory is latent:

$$\log p_\theta(x_0) \ge \mathbb{E}_q\left[\log p_\theta(x_0\mid x_1) - \sum_{t=2}^{T} D_{\mathrm{KL}}\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right) - D_{\mathrm{KL}}\left(q(x_T\mid x_0)\,\|\,p_\theta(x_T)\right)\right].$$

It is optimisable, and in the discrete case it is what the better objectives target. It is still a bound, and a diffusion model's reported likelihood is not the same kind of number as an autoregressive one.

## Why the Gaussian machinery does not survive the trip

A token is an index into a finite set. There is no midpoint between cat and dog, no ordering under which one perturbs the other, no variance parameter that carries meaning on its own.

So you must decide what the noise acts on. One route lifts each position into the probability simplex, a one-hot $e_{x_l} \in \{0,1\}^{V}$ or a softening of it, and puts noise there; the network predicts a distribution per position and is trained by cross-entropy against the clean token at the sampled corruption level. The other keeps the state space finite and makes the forward process a chain with transition matrices over the vocabulary,

$$q(x_t \mid x_{t-1}) = \mathrm{Cat}\left(x_t;\, x_{t-1}Q_t\right), \qquad Q_t \in \mathbb{R}^{V \times V}.$$

The absorbing choice $Q_t = (1-\beta_t)I + \beta_t \mathbf{1}e_{\textsc{mask}}^{\top}$ masks each token with probability $\beta_t$ and otherwise leaves it alone. Under it the posterior factorises into independent per-position categoricals, which is why the masked variant is the easiest to train.

The schedule stops being interpretable. In the Gaussian case $\bar\alpha_t$ reports a signal-to-noise ratio, so two schedules compare directly. Discrete corruption is a sequence of operators, and the same scalar amount means different things at different tokenisations: masking half a byte sequence and masking half a subword sequence leave different information standing.

The learning problem also changes shape: a denoiser emits $V$ numbers per position at every noise level, and has to be calibrated across the whole range of corruption rather than at the single regime an autoregressive model ever sees. Zhao et al. attack that cost. They simplify the variational bound into a cheaper and more accurate form, derive a closed-form backward step that makes sampling exact and faster, and unify the discrete-time and continuous-time formulations so probabilities can take any noise distribution, including different noise for different elements of one object. That is real progress on bookkeeping, not a restoration of the Gaussian intuition. The process is a jump process, not a diffusion in the $L_2$ sense, and its reverse is another jump whose rates have to be learned.

## Parallel decoding and the dependence you cannot delete

The quoted comparison is passes against passes; wall clock is work against work.

| | Autoregressive | Masked discrete diffusion |
|---|---|---|
| Passes for $L$ tokens | $L$ | $T$ |
| Work per pass | one query against a cached prefix | full attention over all $L$ positions |
| Total weight arithmetic | $\approx 2PL$ | $\approx 2PLT$ |
| Independence within a pass | exact factorisation | $L$ marginals given $x_t$ |
| Exact likelihood | yes | bound only |
| Conditions on | prefix | every position, noisily |

Dependence is the other half, and it gets glossed over. A step is written as if positions were independent given the noisy sequence:

$$p_\theta(x_0 \mid x_t) = \prod_{l=1}^{L} p_\theta(x_{0,l} \mid x_t).$$

Inside one step that is mean-field. It cannot express that two positions must agree: a name and the pronoun that refers to it, a number and its digits, a bracket and its partner. Every marginal can be sensible while the combination is not.

The dependence is deferred, not destroyed. Across $T$ steps the model conditions on its own partial commitments, so correlation enters through the trajectory instead of through one factorisation. That is what progressive unmasking buys, and why committing every position at the first step gives text that reads well token by token and falls apart by the sentence.

Parallelism and dependence sit at opposite ends of one dial. Where the target is close to a product of marginals given the noise level, as for local syntax and short spans, small $T$ is enough. Where long-range consistency matters you need more steps or stronger conditioning, and the parallelism shrinks toward the sequential case you were leaving.

Zucchet et al. give a reason to expect the second regime to be common. On a synthetic factual recall task they find three learning phases, with a plateau before precise factual knowledge appears, coinciding with the formation of attention circuits that support recall. Imbalanced training distributions shorten the plateau. Hallucinations emerge alongside the knowledge, and fine-tuning new facts tends to corrupt existing parametric memories. On that account a fact is a circuit binding several tokens, not a property of one position. My hypothesis is that this binding is part of why diffusion samples can be locally fluent and globally wrong: the marginals can each be reasonable while the circuit tying them is a constraint no single step represents.

## The contrarian case: the speed story is the weaker half

The standard framing says these models matter twice. They decode in parallel, and they fill in the middle because generation is not a left-to-right commitment. Both claims are weaker than the framing implies, and the capability that actually differs is a third one.

Speed first. Total weight arithmetic grows with $T$ while the autoregressive total does not, so what a diffusion step buys is a different regime rather than fewer operations. An autoregressive step at batch one reads every weight and does little arithmetic per byte, so it is memory bound. A diffusion step is a full forward over $L$ positions, which looks like prefill: compute heavy, high intensity, efficient. Whether that is faster depends on machine balance and batch size, and batching repairs most of the memory-bound problem on the other side. At high batch the gap narrows, so the advantage sits in the low-batch latency regime, which is also where paying $T$ steps hurts most.

Infilling second. That follows from the training objective, not the sampler: a fill-in-the-middle objective gives an autoregressive model the same ability.

What is exclusive is that every position is visible at every step, so a constraint spanning several positions can be enforced during sampling rather than checked afterwards. A schema with cross-field invariants, a fixed-width field, a unit that must agree with the number before it. In an autoregressive decoder a constraint on the last token of a span cannot shape the first token's distribution except through a pass that has not happened yet. In a denoiser those positions share a step, and the per-position probabilities multiply into a score for the constraint holding, so you can project, reweight or reject inside the sampler.

Schiff et al. make the guidance side concrete. They derive classifier-free and classifier-based guidance for discrete diffusion and introduce models that use uniform noise and are more guidable because they can continuously edit their output. They report improvement over autoregressive and diffusion baselines on genomic sequences, small molecule design and discretised image generation. Those are not natural language results, so the claim has to be scoped: the mechanism is general, and the demonstrations sit in short-sequence domains with hard structural constraints. My reading is that this is not a coincidence, and that the constraint story will outlast the speed story.

## What one pass can and cannot compute

A single pass gives $L$ independent distributions over the vocabulary given the current corrupted sequence. That is $L \times V$ numbers, every one a marginal. Several functionals come free. You can read per-position confidence and decide what to commit this step. You can score a multi-token constraint by multiplying per-position probabilities, then project, reweight or resample. The approximation that loosens the likelihood bound is the same one that turns constraint checking into a product instead of an integral.

What it does not give you is the joint. The argmax of the marginals is not the argmax of the sequence. The entropy of the product is not the entropy over sequences. Anything that depends on correlation between positions has to come from the trajectory, and the trajectory costs steps.

Likelihood is the sharpest case. An autoregressive model returns $\log p(x)$ for a candidate in one pass, exactly. A discrete diffusion model returns a bound, or a Monte Carlo estimate of one, and getting it means noising the candidate and averaging over $t$. Putting those two numbers in one column without saying which is a bound is common, and it is wrong.

The engineering version matters more than the benchmark one. Quantities available in one pass are local in the positions; quantities that need the joint are global in the sequence: whether the argument holds, whether the code compiles, whether the answer follows from the premise. Denoising turns each global property into a search over $T$ steps, and the autoregressive model gives the opposite: exact likelihood for a candidate, and a local decision that cannot see the future.

Neither asymmetry is a defect. They are the terms of the trade, and it is worth knowing which side your problem sits on.

## Sources

- Nicolas Zucchet, Jörg Bornschein, Stephanie Chan, Andrew Lampinen, Razvan Pascanu and Soham De, [How do language models learn facts? Dynamics, curricula and hallucinations](https://arxiv.org/abs/2503.21676v2), arXiv:2503.21676 (2025).
- Lingxiao Zhao, Xueying Ding, Lijun Yu and Leman Akoglu, [Unified Discrete Diffusion for Categorical Data](https://arxiv.org/abs/2402.03701v2), arXiv:2402.03701 (2024).
- Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre, Bernardo P. de Almeida, Alexander Rush, Thomas Pierrot and Volodymyr Kuleshov, [Simple Guidance Mechanisms for Discrete Diffusion Models](https://arxiv.org/abs/2412.10193v3), arXiv:2412.10193 (2024).
