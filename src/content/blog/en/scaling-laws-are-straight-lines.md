---
title: "Scaling Laws Are Straight Lines on a Log-Log Plot"
description: "A power law becomes a straight line under a log-log transform, and that linearity is the whole reason scaling laws are useful for planning training runs."
pubDatetime: "2023-08-13T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Scaling Laws
  - Training
  - Evaluation
categories:
  - AI
---

The first time I plotted a training curve on log-log axes, I felt mildly cheated by how machine learning had been explained to me. On linear axes a loss curve looks like disappointment: a steep drop, then a long flat tail that suggests you have run out of ideas. On log-log axes the same numbers often fall on a straight line. That line is what people mean by "scaling law," and most practical claims about training budgets rest on it.

What follows is not a survey. It is what I wish someone had put in front of me before I started reasoning about how large a run to fund.

## Why a power law is a straight line

A power law has the form

$$L = a \, N^{-b}$$

where $L$ is loss, $N$ is some measure of scale - parameters, tokens, compute - and $b > 0$ is the exponent. Taking logarithms of both sides gives

$$\log L = \log a - b \log N$$

which is the equation of a straight line in $\log N$ and $\log L$, with intercept $\log a$ and slope $-b$. No neural network is required to see this; it is algebra. What is empirical is that language model loss actually behaves this way across several orders of magnitude of $N$. Kaplan and coauthors made that case for parameter count, dataset size, and compute, reporting smooth power-law behaviour in each and arguing that architecture details mattered less than scale over the range they studied (Kaplan et al., 2020).

The industrial appeal follows directly: fit a line to the small runs you can afford, extrapolate to the large run you cannot. You do not need to train the model to know roughly what it will score.

Loss usually depends on parameters $N$ and data $D$ jointly in something like

$$L(N, D) \approx \left( \frac{N_c}{N} \right)^{\alpha_N} + \left( \frac{D_c}{D} \right)^{\alpha_D} + L_\infty$$

where $L_\infty$ is an irreducible floor. Each term is a straight line on log-log axes, but their sum is not. If one term dominates you see a clean slope; when both are comparable you are looking at a curve flattening towards the larger term. I have watched people fit a single line to data sitting in that transition region and then wonder why the extrapolation missed. The line is real, but it is straight in one term, not necessarily in your experiment.

## The dependent variable is chosen, not given

A scaling law is a model of a measurement, and the measurement depends on the tokeniser, the data mixture, and the evaluation set. Cross-entropy per token on a particular corpus is the usual quantity because it is cheap, continuous, and defined everywhere. But "per token" is doing a lot of work.

Change the tokeniser and the same text becomes a different number of tokens. A tokeniser that compresses more aggressively makes each token carry more information, which raises per-token loss even when the model predicts the text no worse. Per-byte metrics exist to sidestep this, and bring their own comparability problems.

The data mixture is the same story in a different costume. Loss on a distribution you trained on heavily falls faster than loss on one you barely saw. If the share of code in your pretraining mixture differs between the small runs and the big run, you have changed the function, not just its argument. A fit will still produce a slope, but the slope now describes a blend of regimes.

So "the scaling law for this model family" honestly means "a fit to this loss, on this distribution, with this tokeniser." Hold those fixed and the law is a useful predictive tool. Vary them and you are measuring something else.

## One axis at a time, versus a fixed budget

There are two ways people talk about scale, and they answer different questions.

The first is to move one axis and hold the rest: bigger models on the same tokens, or the same model on more tokens. Curves from that kind of sweep are what the early scaling papers plotted, and they are straight.

The second is the budget question. With compute $C$ roughly proportional to parameters times tokens,

$$C \approx 6 \, N \, D$$

you must decide how to split a fixed $C$. Spend it all on parameters and you get a huge model trained on almost nothing; spend it all on tokens and a small model reads the internet several times. Both extremes are bad, and the loss surface along the budget-constrained slice has a minimum.

This is where the Chinchilla result lands. Hoffmann and coauthors fitted scaling laws across a large sweep and asked what split minimises loss at fixed compute. Their answer: parameters and tokens should grow in roughly equal proportion, which implied that many contemporary large models were substantially undertrained for their size. A model several times smaller, trained on several times more data for the same compute, would reach lower loss (Hoffmann et al., 2022).

The two framings are easy to conflate. "Loss falls as a power law in parameters" describes the shape of a slice. "At fixed compute, allocate roughly equally" tells you where on the slice to stand. The first is terrain; the second is navigation. A paper can be entirely correct about the first and still motivate a badly allocated run, which is more or less what happened.

Three questions are worth keeping separate: how loss changes when I scale $N$ alone, when I scale $D$ alone, and where the optimum sits on the iso-compute curve. Only the third is an allocation question, and it needs the joint fit rather than either marginal one.

## Exponents carry the information

People argue about constants. They should argue about exponents.

The constant $\log a$ shifts the line up or down. Get it wrong and your absolute predictions are wrong, but your conclusions about relative improvement between two scales are probably intact. The exponent $-b$ controls how fast returns decay, and it decides whether an intervention is worth the money.

Two candidate improvements with exponents $b_1 = 0.05$ and $b_2 = 0.10$ show why. At your current scale the second may look worse because its intercept is higher. Ten times further out, the slope difference dominates and the ranking flips. Which to invest in is a question about the exponent, and its answer changes with the scale you are planning for.

The same logic compares data against parameters. If loss falls faster with tokens than with parameters over the relevant range, an extra unit of compute is better spent on data. That comparison is what the Chinchilla analysis formalised, and why reporting exponents alongside curves beats reporting one extrapolated number.

There is a discipline problem. An exponent fitted to three points spanning half a decade is not well determined: a small change in the data rotates the line, and rotation compounds under extrapolation. If a budget decision rests on an exponent, the fit should span enough range that the slope is not an artefact of two adjacent runs, and someone should have estimated its uncertainty.

## What a scaling law does not tell you

**Data quality.** The law is fitted to a particular distribution. Improve your filtering or mixture and you have changed the function; the fitted line no longer describes your pipeline. Gains from better data can be large, but they are not a term in the standard fit. A team optimising only size and tokens is optimising inside a box that better data would move.

**Downstream task performance.** Loss is a proxy: smooth, cheap, correlated with many capabilities, and not a law. A model can improve on held-out cross-entropy while a benchmark stalls, or the reverse. Quoting a scaling law as a forecast of benchmark accuracy adds a claim the law does not contain.

**Whether the curve bends.** Every fit is local. Two things break the line: a floor, where irreducible loss is reached and returns vanish, and a change of regime, where a different term starts to dominate. A curve straight over three orders of magnitude can bend in the fourth. The responsible claim is "straight over the measured range, with these parameters, and here is where I expect the floor to matter."

## The contrarian case: the law is not about capability

The standard framing is wrong in a specific way: scaling laws are routinely cited as evidence about capability, and they are fits to a scalar loss on a fixed distribution whose fitted exponent usually has an interval wider than the difference people are arguing about.

Look at what such a paper reports: runs, a functional form, a table of constants, a line through points. The line summarises measured loss. The extrapolation claims loss at unmeasured scales, conditional on the form being right. Nowhere in that chain is a claim about what the model can do. The step from "loss follows a power law" to "capabilities arrive on schedule" is an extra premise imported from outside the fit, and it is the premise that fails most often.

The interval point is concrete. Fit a line to a few noisy points and the slope's uncertainty grows as the span shrinks, the noise grows, and the count falls. Published sweeps are expensive, so there are few runs, the range spans a couple of orders of magnitude at best, and seed-to-seed noise is not negligible. The interval on $b$ is not tight. Two groups can fit different exponents to comparable data and both be defensible, which means a dispute over whether returns go as $N^{-0.07}$ or $N^{-0.09}$ may be a dispute inside the noise.

That changes how the result should be used. If someone claims an architectural change "improves the scaling exponent," ask how many runs the claim rests on and how much range the fit covers. Four runs over half a decade will not distinguish an exponent shift from fitting noise, and the claim should be stated as a direction rather than a coefficient.

There is a subtler failure too. Loss is smooth and monotone in scale; capability evaluations are noisy and threshold-like. People compare a smooth curve to a step function and argue about where the step lands. The curve has no step in it. It cannot. Using it to argue about when a capability appears applies a model of the wrong functional form to the question, and the straightness of the line makes the error feel rigorous.

None of this makes scaling laws useless. They are among the few predictive tools for planning expensive runs, and the Chinchilla correction is the framework doing real work: it changed how budgets were allocated, on the strength of a joint fit rather than a hunch. But that authority comes from being narrow and well specified. Widen it into a general theory of progress and the specificity is gone, leaving a straight line carrying an argument it was never fitted for.

## Sources

- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*. https://arxiv.org/abs/2001.08361
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*. https://arxiv.org/abs/2203.15556
