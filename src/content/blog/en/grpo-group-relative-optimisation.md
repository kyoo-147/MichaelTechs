---
title: "GRPO and the Arithmetic of a Group"
description: "GRPO drops the critic and divides by a group's own spread. The estimator is simple, but its scale and its signal both depend on group size."
pubDatetime: "2025-03-22T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - GRPO
  - Policy Optimisation
categories:
  - Math
---

## Eight answers, three correct, one number

You sample eight completions for one prompt. A rule-based checker marks three correct. The update needs one number per completion, and the sign of that number decides whether the tokens inside it are reinforced or suppressed. With a learned critic you would ask a value network what each prefix was worth and compare the realised return against that prediction. GRPO does something blunter. It takes the mean of the eight rewards and subtracts it.

That is the whole mechanism, and it is worth being precise about how little it assumes. The baseline is computed from the same samples that produce the gradient. Nothing is learned to produce it. There is no second network, no value head, no separate optimiser state for a critic, no bootstrap. What you need is a reward you can compute and a model you can sample from more than once.

DeepSeek-AI report that a pure reinforcement learning recipe, with no supervised cold start on reasoning traces, produced self-reflection, verification and dynamic strategy adaptation on verifiable tasks. That result is what made the estimator the default across the field. The estimator is not new. What changed in January 2025 was that a large lab published a training run in which a critic-free group baseline carried the entire load.

## What the subtraction actually is

For completion $i$ in a group of size $G$, the advantage is

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \qquad \mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j, \qquad \sigma_G^{2} = \frac{1}{G}\sum_{j=1}^{G}\left(r_j - \mu_G\right)^{2},$$

and $\hat{A}_i$ is broadcast to every token of completion $i$. The loss wrapped around it is the clipped ratio objective with a KL penalty to a reference model. The group supplies the advantage and nothing else.

Subtracting a constant from the return leaves the policy gradient unbiased, provided the constant does not depend on the sampled action. The group mean does depend on the samples, but it enters detached, and each sample contributes exactly $1/G$ of its own baseline. Expanding the subtraction shows the price:

$$r_i - \mu_G = \frac{G-1}{G}\left(r_i - \mu_{-i}\right),$$

where $\mu_{-i}$ is the leave-one-out mean. The group-mean advantage is the leave-one-out advantage scaled by $(G-1)/G$, which is $0.875$ at $G = 8$. Group size is therefore not a neutral setting. It sets the size of the step, before the division by $\sigma_G$ sets it again.

That division is the part I would flag to anyone adopting this. Standardising makes the update invariant to the scale of the reward, which is useful when the reward is a score in arbitrary units. It also lifts a group whose rewards barely differ up to the same magnitude as a group where the difference is large. The estimator cannot tell a real preference from a rounding artefact.

## The variance of a small group

Two separate problems hide behind the phrase "the group gives you a baseline".

The first is that $\sigma_G$ is itself estimated from $G$ numbers. For roughly normal rewards the sample standard deviation carries relative noise of about $1/\sqrt{2(G-1)}$, so at $G = 8$ the denominator of every advantage is uncertain by something like 27 percent. The advantage is a ratio, and the denominator is a random variable.

The second is discreteness. With a binary reward and pass rate $p$, a fraction $p^{G} + (1-p)^{G}$ of groups has zero spread. Those groups produce zero advantage and exactly zero gradient. The rollout budget was spent and nothing was learned. Li, Liu and Yang name zero-variance advantage estimation among the deficiencies that affect training stability and inference efficiency.

```python
degenerate = p**G + (1 - p)**G         # groups that carry no gradient
noise      = 1 / math.sqrt(2*(G - 1))  # relative noise in the sample std
```

Both expressions are arithmetic rather than measurement. Evaluating them for a binary reward at $p = 0.8$ gives the table below.

| $G$ | Relative noise in $\sigma_G$ | Groups with no gradient |
| --- | --- | --- |
| 2 | 0.71 | 0.68 |
| 4 | 0.41 | 0.41 |
| 8 | 0.27 | 0.17 |
| 16 | 0.18 | 0.03 |
| 32 | 0.13 | 0.001 |

Read the row for eight and the trade becomes concrete. At a pass rate of four in five, one group in six is thrown away. Sixteen samples bring the waste under three percent and cost double the decoding.

A third effect shows up in neither column. For a binary reward the magnitude of the advantage depends on how many successes landed in the group. At $G = 8$ with one success, that completion receives an advantage about seven times the magnitude of each failure, and about 2.6 times what a success receives in a group split four and four. A rare success is strong evidence, so the weighting is defensible. It is also a difficulty curriculum applied to every prompt in the batch, and no one chose it.

## The contrarian read: the group mean is a filter

The usual summary is that GRPO removes the critic, that removing the critic removes memory cost and value-function bias, and that the result is PPO's stability without PPO's machinery. I think that gets the accounting wrong in two places.

The first is cost. The critic was rarely the expensive object in a rollout-heavy loop. Generation is. Going from a learned value function to a group of eight multiplies the rollouts per prompt by eight. The method does not remove cost. It moves cost from the update step to the sampling loop. Whether that is a good trade depends on which side you were already bound by, and that is a property of your hardware and your sequence lengths, not of the algorithm.

The second is variance. The claim that a group baseline reduces variance is true only conditionally, and the condition matters more than the claim. On prompts the model always gets right, or always gets wrong, the baseline subtracts nothing, the advantage is zero, and the gradient is zero. The variance reduction is active only on the slice of the batch near the model's current decision boundary. Which is why I think the group mean is better described as a filter than as a baseline. It decides which prompts receive gradient at all, and the division by $\sigma_G$ then makes every surviving prompt count equally regardless of how uncertain it was.

This reading is testable, which is why I prefer it. If the group mean is a filter, the effective batch is much smaller than the nominal batch, and $1 - p^{G} - (1-p)^{G}$ is a number to log rather than assume. It also predicts that prompt difficulty scheduling matters more than group size once $G$ is past the point where the degenerate fraction is small.

DAPO is consistent with that. The authors report 50 points on AIME 2024 with a Qwen2.5-32B base model, and they describe four key techniques that made large-scale RL succeed at that scale. My reading is that the group-relative estimator on its own was not the recipe.

## What the value function was buying

Dropping the critic is not free, and the cost is not variance.

A critic scores prefixes, so it produces an advantage per token. A token in the middle of a correct derivation can be reinforced while a wrong turn earlier in the same trajectory is suppressed. GRPO produces one scalar per sequence. In a failed four-thousand-token attempt, all four thousand tokens receive the same negative advantage, including the three thousand nine hundred that were fine. That is not noise. It is systematic misattribution, and it gets worse as sequences lengthen, which is the regime reasoning training lives in.

A critic also supports bootstrapping and intermediate rewards. A group relative advantage uses the terminal reward and nothing else. If the reward is sparse and arrives at the end, every token is judged by an outcome it may have had little to do with.

Against that, a critic's error is a bias that more samples do not remove, and it is hardest to fit exactly where the policy is changing fastest. The group baseline's error is variance, and the variance falls as $G$ grows. Trading a bias you cannot control for a variance you can is a reasonable engineering decision, and it is the decision GRPO makes.

What I think drove adoption is less statistical than that. Removing a model removes a class of operational failure: value loss weighting, GAE lambda, critic divergence, a second optimiser state, and the awkward question of when to checkpoint a critic you intend to throw away. A method with one fewer moving part is easier to run, and on a long training job operational simplicity compounds.

## Group size is a learning rate

Two consequences of the algebra are worth stating as rules.

If you change $G$, you have changed the effective step size twice, once through $(G-1)/G$ and again through the scale of $\sigma_G$. Halving the group to save decoding is a change to the optimisation, not a change to the bill. A team that lowers $G$ without touching the learning rate is running a different experiment, and the result will look like instability rather than like the change in step size that it is.

If you change the reward shape, you have also changed the step size, because standardising removes the scale of the reward but not its structure. Shaped rewards put small differences in the numerator, and a small $\sigma_G$ in the denominator turns them into full-size updates.

The algebra also suggests that the baseline is not the load-bearing stabiliser. The clipped ratio and the KL anchor are. The image captioning work makes the case for the mean in its plainest form: the argument there is not that a critic is expensive, it is that self-critical sequence training's single greedy baseline has high variance, and that one comparison is a weak estimate of what a completion was worth. Eight comparisons estimate it better. That is a statement about estimation rather than about architecture, and it is the strongest defence the group mean has.

## What I would measure first

This part is opinion, and it is cheap to act on.

Log the fraction of groups with zero reward spread. If it is above a third, most of the nominal batch is producing nothing and the fix is prompt difficulty rather than group size. Log the distribution of reward spread, not only mean reward: a mean that rises while the spread collapses means you are walking into the degenerate regime, and the curve will look healthy until the gradient stops. Log the effective advantage scale per prompt, because that is the quantity your learning rate is actually multiplying.

Then treat $G$ as a parameter with two effects instead of one. It buys a better estimate of the baseline at a rate near $1/\sqrt{G}$ and a better estimate of the spread at $1/\sqrt{2(G-1)}$, and it buys a smaller degenerate fraction at a rate that depends on how good the model already is. Those are different curves. Somewhere they cross, and past that point you are paying for decode that changes nothing.

The arithmetic of a group is the arithmetic of a sample. Everything GRPO gains, and everything it gives up without saying so, follows from treating eight numbers as a population and letting that population decide the sign of the gradient.

## Sources

- DeepSeek-AI, Daya Guo, Dejian Yang and others, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (2025), arXiv:2501.12948.
- Xu Liang, [Group Relative Policy Optimization for Image Captioning](https://arxiv.org/abs/2503.01333) (2025), arXiv:2503.01333.
- Qiying Yu, Zheng Zhang, Ruofei Zhu and others, [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) (2025), arXiv:2503.14476.
- Chen Li, Nazhou Liu and Kai Yang, [Adaptive Group Policy Optimization: Towards Stable Training and Token-Efficient Reasoning](https://arxiv.org/abs/2503.15952) (2025), arXiv:2503.15952.
