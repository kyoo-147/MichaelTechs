---
title: "DPO: Preferences Without Reinforcement Learning"
description: "DPO folds the reward model into the policy and trains on pairs, but the implicit reward is only identified up to a function of the prompt."
pubDatetime: "2024-04-20T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - DPO
  - Preference Optimisation
categories:
  - Math
---

A team has ten thousand pairs of model responses. A human marked one better than the other. The goal is a model that prefers the better one on new prompts.

The obvious route is the two-stage one: fit a reward model to the pairs with a Bradley-Terry likelihood, then run a policy optimiser against that reward while a KL penalty holds the policy near a frozen reference. It works, and it costs four things to keep in sync - a reward model, a policy, a reference policy, a value head - plus a sampler inside the training loop that produces fresh completions every step so the reward model scores the policy it is currently training.

Direct Preference Optimization removes the sampler and the separate reward model, and it does so with algebra rather than approximation. What follows is the derivation, what it hides, and what it still demands from you.

## The Bradley-Terry Model Assumes More Than It Says

The model makes one assumption and everything downstream inherits it. Each response has a scalar latent score, and the probability that one response is preferred over another is a logistic function of the score difference:

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x,y_w) - r(x,y_l)\right), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

This buys a likelihood over pairs that never requires the score to mean anything specific. It also asserts transitivity in probability, one number per response, and no representation of which attribute drove the judgement. Two annotators who disagree because one weighs helpfulness and the other weighs brevity are, under this model, indistinguishable from noise. That is not a small caveat, and I will come back to it.

## The Closed Form That Deletes the Sampler

Start from the objective you actually want:

$$\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D}}\left[\mathbb{E}_{y\sim\pi(\cdot\mid x)}\, r(x,y) - \beta\, \mathbb{D}_{\mathrm{KL}}\left(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)\right]$$

Maximising it in closed form gives a tilted reference distribution:

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right), \qquad Z(x)=\sum_{y}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right)$$

Invert that and the reward comes back as a function of the policy:

$$r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)$$

Substitute it into the Bradley-Terry likelihood. The two responses share the prompt, so $\beta\log Z(x)$ appears on both sides of the difference and cancels:

$$P(y_w \succ y_l \mid x) = \sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)$$

The negative log-likelihood of that expression is the loss:

$$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]$$

Every term is computable from two forward passes plus two frozen forward passes. Training skips the sampler, the separate reward model, and the value function.

## The Implicit Reward Is Only Defined Up to a Function of the Prompt

Define the implicit reward:

$$\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$$

The loss depends only on differences of $\hat r_\theta$ between responses to the same prompt. Add any function $c(x)$ to every score at a fixed $x$ and nothing changes. So the implicit reward is identified only up to an additive term that depends on the prompt alone, not up to a global constant.

That distinction matters more than it sounds. A global constant would be harmless bookkeeping. A prompt-dependent one means the model can never tell you that prompt $x_1$ admits better answers than prompt $x_2$. Every cross-prompt statement about reward is meaningless, including averages over a test set and any absolute threshold you might want to set. The only well-defined quantity is an ordering within a prompt.

The same cancellation answers the partition function question. $Z(x)$ sums over all possible sequences and is intractable, which is usually fatal for energy-based formulations. Here it is never needed, because it is constant at fixed $x$ and only differences survive. The price is that you learn a ratio and never the normaliser.

There is a further consequence, and it is the one that bites when you try to compare runs. The scale is fixed by $\beta$ together with the logistic link, not by the data. You cannot rescale the implicit reward after training and compare it to a reward model trained separately. The units are nats of log-probability ratio per unit of $\beta$, and they do not transfer.

## The Contrarian Reading: The Reward Model Was Renamed, Not Removed

The standard account says DPO eliminated the reward model. Read the loss again. $\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ occupies exactly the position the reward used to occupy. Under the Bradley-Terry assumption the optimal policy and the reward are two coordinates of a single object, related by the closed form above. Eliminating the explicit reward model means the policy network is now the reward model, queried only at the responses that appear in the data.

So the failure modes of reward modelling do not disappear. They move.

Misspecification does not care which parameterisation you choose. If Bradley-Terry is the wrong model for your preferences, the loss still converges, to the policy whose implicit reward best explains the pairs under a wrong likelihood. You removed a place to intervene, not the error.

Distribution shift gets worse rather than better. The pairs were generated by some behaviour policy, and $\pi_\theta$ moves away from it with every step. You are fitting a binary classifier on an offline distribution while the object you care about is the policy at convergence. The reference bounds how far you can travel, which is why $\beta$ carries so much weight.

Observability is the real loss. With a separate reward model you can score held-out responses, plot the distribution, notice that scores track length, and fix the data. With DPO the reward is a log-ratio recomputed on the fly, and the honest instrument you have is preference accuracy on pairs, which is the same quantity you are optimising.

What actually disappeared is narrower than the slogan: the on-policy sampler, the value network, and the reward model as a separately trainable artefact. The optimisation problem is still KL-constrained policy optimisation. It is solved by supervised learning on pairs because the constraint and the preference model together reduce it to classification.

## What the Gradient Pushes, and Why the Margin Is Not Quality

Differentiate the loss:

$$\nabla_\theta \mathcal{L} = -\,\beta\,\sigma\left(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\right)\left[\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\right]$$

Two things follow directly. The scalar weight is the probability the model currently assigns to getting the pair wrong, so pairs already ordered correctly contribute almost nothing and stop teaching. And the update is a difference of two log-probability gradients. Nothing in the objective protects the absolute likelihood of the chosen response, only the gap between the pair. A step that lowers both log-probabilities while widening the gap is legal descent, and the reference term is what keeps that from running away. This follows from the algebra and is not a claim about any particular run.

A minimal implementation, working in sequence-level log-probabilities:

```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    # each argument is a summed log-probability per sequence, shape (batch,)
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward.detach(), rejected_reward.detach()
```

The returned values are the implicit rewards, and their mean difference is what gets reported as reward margin. It is a diagnostic, not a quality score.

## What DPO Still Needs From the Data

The pairs are the entire supervision signal, so their structure matters as much as their count.

Sequence-level labels are coarse. Zeng et al. make this argument in "Token-level Direct Preference Optimization": preference is judged on whole answers, but generation happens token by token, so a sequence-level Bradley-Terry reward compresses a token-level process into one number. They apply a forward KL constraint per token and keep the Bradley-Terry form at token granularity without explicit reward modelling, reporting a better balance between alignment and generation diversity than DPO on controlled sentiment generation and single-turn dialogue. Whether their fix generalises is open. The diagnosis is sound: the unit of supervision is a design choice, not a given.

Scalar rewards compress a population. Wang et al. argue in "Arithmetic Control of LLMs for Diverse User Preferences" that reliance on a scalar reward limits the ability to capture diverse preferences, and build a framework with multi-objective reward modelling and preferences represented as directions in reward space, comparing against DPO as a strong baseline. My reading is that this is the deeper data problem. If your annotators are trading helpfulness against verbosity, one scalar per response is lossy compression applied before training starts, and extra pairs do not repair it.

The checklist I would apply before blaming the optimiser: are prompts drawn from the distribution you will be evaluated on, are pairs length-matched or is length confounded with preference, do annotators agree with themselves on a held-out relabel, and is the reference checkpoint the same model the pairs were sampled from. The last one is routinely violated.

## What DPO Still Needs From the Hyperparameters

| Knob | What it controls | Failure mode when mis-set |
| --- | --- | --- |
| $\beta$ | KL leash strength and the unit of the implicit reward | Too small: policy drifts, margin inflates, outputs degrade. Too large: weights saturate, little movement |
| Learning rate | Step size on a classification loss with a moving target | Too high: chosen log-probability collapses. Too low: no separation after one pass |
| Epochs | How hard you fit the offline pairs | Repeated passes over a fixed pair set overfit the labels and widen the reference gap |
| Batch size | Variance of the per-pair weights | Small batches give noisy weights and unstable margins |
| Reference model | The anchor and the zero point of $\hat r_\theta$ | A reference other than the SFT initialisation starts training with a non-zero bias |
| Pair construction | What the loss can actually see | Length-confounded pairs teach length, because the loss has no length term |

Two of these are coupled in a way that is easy to miss. $\beta$ multiplies both the loss and the reward definition, so changing it also changes the effective step size. Tuning $\beta$ and the learning rate as if they were independent knobs is a common source of confusion: a run that appears to need a larger learning rate at small $\beta$ is often just compensating for the loss scale.

The rest is ordinary care. Freeze the reference and do not update it, because a moving reference breaks the closed form the loss was derived from. Keep the pair set fixed per epoch and read the margin alongside held-out preference accuracy, since margin on its own will keep rising while quality does not.

## Sources

- Haoxiang Wang, Yong Lin, Wei Xiong, Rui Yang, Shizhe Diao, Shuang Qiu, Han Zhao, Tong Zhang. "Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference Alignment with Multi-Objective Rewards." [arXiv:2402.18571](https://arxiv.org/abs/2402.18571), 2024. https://arxiv.org/abs/2402.18571
- Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, Jun Wang. "Token-level Direct Preference Optimization." [arXiv:2404.11999](https://arxiv.org/abs/2404.11999), 2024. https://arxiv.org/abs/2404.11999
