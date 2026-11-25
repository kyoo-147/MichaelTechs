---
title: "Probability Is a Language for Uncertainty, Not a Claim About Truth"
description: "Why cross-entropy and squared error are maximum-likelihood estimates in disguise, and why a good likelihood does not give you calibrated confidence."
pubDatetime: "2023-04-16T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Probability
  - Loss Functions
  - Calibration
categories:
  - Math
---

People often introduce probability as "the chance of an event". That framing is fine for dice and misleading for models. The useful reading is narrower and more practical: a probability distribution is a **bookkeeping device for uncertainty**, and a model's output distribution is a claim about which outcomes are plausible, not a claim about what will happen.

Once you take that seriously, two things follow that most tutorials skip: the loss function you train with is a statement about your noise model, and the number your model outputs is not the probability that it is right.

## Distributions, expectations, estimators

A random variable $X$ with density $p(x)$ has expectation

$$
\mathbb{E}[X] = \int x\,p(x)\,dx
$$

and variance $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2]$. Those are summaries, and summaries lose information: two very different distributions can share a mean and a variance. Goodfellow, Bengio and Courville make this point early in [chapter 3](https://www.deeplearningbook.org/contents/prob.html) of the *Deep Learning* book - the moment you replace a distribution with its mean, you have chosen to ignore everything else about it.

The distinction that matters for training is between a **parameter** (a fixed but unknown quantity, like the true noise level) and an **estimator** (a function of the data that guesses it). Maximum likelihood is one estimator among several, chosen for convenience as much as for principle.

## Maximum likelihood, and where your loss comes from

Given data $\{(\mathbf{x}_i, y_i)\}$ and a model $p_\theta(y \mid \mathbf{x})$, maximum likelihood picks

$$
\hat\theta = \arg\max_\theta \sum_i \log p_\theta(y_i \mid \mathbf{x}_i).
$$

That is the whole recipe. What people rarely notice is that the familiar losses fall out of it, and the assumptions come along for free.

Regression with squared error. Assume $y = f_\theta(\mathbf{x}) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ and $\sigma$ fixed. Then

$$
-\log p_\theta(y \mid \mathbf{x}) = \frac{(y - f_\theta(\mathbf{x}))^2}{2\sigma^2} + \text{const},
$$

so maximising likelihood is minimising squared error. The $\sigma^2$ in the denominator is the part everybody drops, and it is the part that says *every sample has the same noise scale*.

Classification with cross-entropy. Assume $y \sim \mathrm{Bernoulli}(p_\theta(\mathbf{x}))$. Then

$$
-\log p_\theta(y \mid \mathbf{x}) = -\left[y\log p_\theta + (1-y)\log(1 - p_\theta)\right],
$$

which is binary cross-entropy. Softmax plus cross-entropy is the same argument with a categorical distribution.

| Loss | Implicit noise model | What breaks when it is wrong |
| --- | --- | --- |
| Squared error | Gaussian, constant variance | Outliers dominate; heteroscedastic data is mis-weighted |
| Cross-entropy | Bernoulli / categorical | Label noise is treated as truth, not as noise |
| Huber | Gaussian near zero, linear in the tails | Requires tuning the transition point |

## The first reverse angle: the loss is an assumption, not a law

If your residual distribution is heavy-tailed, squared error is not "the natural choice" - it is a Gaussian assumption you did not make deliberately, and it will let a handful of points set your gradient. That is not a hypothetical: it is the everyday behaviour of a regression model on data with a few mislabelled rows.

The fix is not "use a better optimizer". It is to write down the noise model you actually believe, and derive the loss from it. Huber exists precisely because someone did that and decided the tails were not Gaussian. If you cannot state the assumption behind your loss, you cannot say when it is the wrong loss.

## The second reverse angle: likelihood is not calibration

This is the one that costs people money. A model can achieve excellent negative log-likelihood and still be badly calibrated. Cross-entropy rewards *ranking* the correct class above the others; it does not force the score to mean what it looks like.

Concretely: a binary classifier that outputs 0.9 on a thousand examples and is correct on 700 of them has good accuracy and terrible calibration - its confidence overstates its hit rate by 20 points. Expected calibration error measures exactly this gap by binning predictions and comparing mean confidence to observed frequency:

$$
\mathrm{ECE} = \sum_{b} \frac{n_b}{N}\,\left|\,\mathrm{acc}(b) - \mathrm{conf}(b)\,\right|.
$$

Modern neural networks are systematically overconfident, and the correction is often embarrassingly simple - temperature scaling divides the logits by a single scalar $T$ fitted on validation data. It changes no prediction, only the confidence attached to it.

So "the softmax output is a probability" is true in the sense that it sums to one, and false in the sense that anyone cares about. It is a score that has been normalised. If a downstream decision depends on the number - routing, abstention, human review - measure the calibration before you trust it.

## Calibration, worked

Suppose a classifier makes 1,000 predictions with the confidence bins below. "Confidence" is the mean predicted probability in the bin; "accuracy" is the observed hit rate.

| confidence bin | n | mean confidence | accuracy | gap × weight |
| --- | --- | --- | --- | --- |
| 0.0-0.2 | 100 | 0.08 | 0.15 | $0.07 \times 0.1 = 0.007$ |
| 0.2-0.4 | 150 | 0.30 | 0.25 | $0.05 \times 0.15 = 0.0075$ |
| 0.4-0.6 | 200 | 0.50 | 0.48 | $0.02 \times 0.2 = 0.004$ |
| 0.6-0.8 | 250 | 0.70 | 0.62 | $0.08 \times 0.25 = 0.02$ |
| 0.8-1.0 | 300 | 0.92 | 0.80 | $0.12 \times 0.3 = 0.036$ |

Overall accuracy is $0.544$, and

$$
\mathrm{ECE} = 0.007 + 0.0075 + 0.004 + 0.02 + 0.036 = 0.0745 .
$$

Read the table rather than the scalar. The model is under-confident in the lowest bin and badly over-confident in the highest: when it says 0.92 it is right 0.80 of the time. A downstream policy that routes on "confidence above 0.9" is therefore firing on a population that is wrong one time in five, while a threshold at 0.5 would be almost exactly calibrated. No single number would have told me that; the table does, and the ECE only summarises it.

## What I actually do

- Write the noise model next to the loss, in a comment, once. If I cannot, I stop and think.
- Keep a calibration plot for anything that outputs a confidence a human will read.
- Prefer a loss derived from the assumption over a loss chosen because it is in a tutorial.
- Remember that $\mathbb{E}[X]$ is a summary, not the distribution; when the decision depends on the tail, look at the tail.

Probability is useful here, but not as decoration. It is the only language in which "how wrong, and in what way" can be written down precisely enough to be checked.

## Sources

- Ian Goodfellow, Yoshua Bengio, Aaron Courville, [Deep Learning, ch. 3: Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html), MIT Press, 2016.
- Kevin P. Murphy, [Probabilistic Machine Learning](https://probml.github.io/pml-book/), MIT Press (free online).
