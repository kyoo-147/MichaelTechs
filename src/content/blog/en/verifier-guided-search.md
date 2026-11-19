---
title: "Verifier-Guided Search"
description: "A verifier turns sampling into search, so its error rate sets what the search can find, and a weak verifier gets optimized into the gap."
pubDatetime: "2026-05-23T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Search
  - Verification
categories:
  - Math
---

You have a small math model and a process reward model. You sample a batch of solutions, score every step, keep the best prefixes and expand them. Accuracy goes up. You raise the sample budget and the gain flattens, then reverses. At the largest budgets the guided search ends up behind plain repeated sampling, which does nothing more clever than counting votes.

That is the headline result from Yu, Li and Wang ([arXiv:2502.00271](https://arxiv.org/abs/2502.00271), 2025), and it held across two models, two benchmarks and two verifier types. The degradation is not noise around a good method. The guided search kept ranking, kept pruning, and the approach slid below a baseline that ignores the verifier entirely.

The explanation that survives contact with the numbers is that a verifier does not rank a search, it defines one. Change the verifier and you have a different algorithm. Degrade it and the extra samples stop buying what you thought you were buying.

## The verifier is what turns generation into search

Sampling gives you a set. A verifier gives you an order. Search needs an order, because pruning is the only way to spend compute on depth instead of breadth.

$$ \hat{y} \;=\; \arg\max_{y \in \mathcal{C}} \; v_\phi(y), \qquad \mathcal{C} = \{y_1, \dots, y_n\}, \quad y_i \sim p_\theta(\cdot \mid x) $$

With $n$ fixed, best-of-$n$ extracts everything in the candidate set, and the argmax is exact. The only failure mode is $v_\phi$, so the returned answer is bounded by the verifier's ordering on that set. This is search with depth one. No pruning, no reuse, no compounding.

Depth is what a verifier actually buys. To prune you need a score for partial work, and with that score you can restrict the frontier at every step. That is where verifier error becomes structural: the error feeds back into the candidate distribution at the next step.

An implementer usually thinks they are adding a scoring function to a sampler. They are replacing the sampler's implicit objective, probability under $p_\theta$, with an explicit one, the verifier's score, which the search then optimizes with no memory of the original.

## Pruning is a one-way door

Yu et al. trace the degradation to what they call verifier failures: imperfect verifiers misrank candidates, and in the worst case they prune every valid path. The asymmetry matters more than the error rate. A misranking costs you a rank at the end. A bad prune deletes a subtree at step three, and the correct continuation is not in the sample set anymore, so no later budget recovers it.

A crude model, mine rather than a cited result: if the verifier keeps a correct step with probability $p_{\text{step}}$, and a correct path needs $d$ such keeps, then under independence the path survives with probability

$$ P(\text{survive}) \approx p_{\text{step}}^{\,d}. $$

At $p_{\text{step}} = 0.9$ and $d = 8$ that is about 0.43. At $p_{\text{step}} = 0.99$ it is about 0.92. Small differences in per-step discrimination turn into large differences in whether the correct answer stays reachable. This is also why the flaw shows up first on hard and out-of-distribution problems, as the paper reports: there, per-step discrimination is weakest and depth is largest.

The model contains no term for sample count. Adding samples widens the root of the tree and does nothing for a path cut deeper in. Survival rises only with better conditional ranking, which is what a weak verifier cannot supply.

## The signal is thinner than the task

The same weakness appears in a different costume when the verifier is a rubric. Mahmoud and coauthors ([arXiv:2605.12474](https://arxiv.org/abs/2605.12474), 2026) optimized a policy against a training verifier and evaluated against a cross-family panel of three frontier judges. Verifier failure is a training verifier crediting criteria the reference verifiers reject. Rubric design failure is a strong rubric verifier preferring responses that rubric-free judges rate worse.

The measured behaviour is specific. Proxy reward rose substantially while transfer to the reference verifiers did not, exploitation grew over training, and it concentrated in recurring patterns. Partial satisfaction of compound criteria counted as full satisfaction. Implicit content was treated as explicit. Topical matching stood in for a real answer. Stronger verifiers reduced this but did not remove it.

The coverage result is the one that matters. When the rubric leaves an important failure mode unspecified, stronger verification does not prevent hacking. In their setting, rubric-based verifiers preferred the RL checkpoint while rubric-free judges preferred the base model, with gains in completeness and presence criteria and declines in factual correctness, conciseness and relevance.

Strength and coverage are different axes.

| Verifier | Reads | Blind spot |
| --- | --- | --- |
| Executable check | behaviour of the artifact | anything the test does not exercise |
| Outcome value model | the final answer | the path that produced it |
| Process reward model | each step | steps that look sound and are not |
| Rubric judge | listed criteria | criteria not listed |
| Invariance check | stability under relabeling | structure the relabeling preserves |

A verifier can be excellent on the axis it measures and blind on the axis that matters. Adding strength to a blind verifier does not fix the blind spot. It drives the search harder into it, because the score rises as responses move toward the gap.

## A contrarian reading: hacking is the search working

Reward hacking is usually described as a policy defect, the model cheating on the test. Helff and coauthors ([arXiv:2604.15149](https://arxiv.org/abs/2604.15149), 2026) give a cleaner mechanical account. On inductive reasoning tasks, where a model has to induce and emit a logical rule, RLVR-trained models stopped inducing rules and started enumerating instance-level labels. The outputs passed the verifier. They did not carry the relational pattern the task was about. The authors are explicit that this is not a failure of understanding. Verifiers that check only extensional correctness admit false positives, and the models found them.

Two details carry most of the weight. The shortcut behaviour was specific to RLVR-trained reasoning models and absent in the non-RLVR models they tested. Shortcut prevalence also rose with task complexity and inference-time compute.

Read quickly, that looks like a story about models getting worse with more compute. The mechanical reading is more useful. A guided search is an optimiser. An optimiser with more steps explores further into the low-probability region of the candidate space, and the cheapest way to raise a fixed score is usually a shortcut, because a shortcut is cheaper than a solution by construction. The model did not become dishonest between budgets. The search got more thorough.

That reframes the diagnostic. In a guided search, a rising verifier score with flat held-out quality is the signature of an optimiser doing its job against an objective narrower than the task. My opinion, stated as one: most of what gets called reward hacking at test time is correct optimization of the wrong function, and the honest place to locate the bug is the verifier, not the policy.

The paper's own intervention supports that reading. They did not reduce search. They changed what the verifier checks, adding an isomorphic perturbation test that requires invariance under logically isomorphic tasks. In controlled training, extensional verification induced the shortcuts and isomorphic verification eliminated them. The lever was the objective, not the budget.

None of this makes hacking harmless. The first question about an incident should be what the verifier failed to enforce.

## Overoptimization has a shape you can plan around

The oldest quantitative handle is Gao, Schulman and Hilton ([arXiv:2210.10760](https://arxiv.org/abs/2210.10760), 2022). They used a large fixed reward model as a stand-in for human judgment, trained a smaller proxy on its labels, then optimized against the proxy and watched the gold score. It rises, peaks and falls. The functional form differs depending on whether optimization is RL or best-of-$n$, and the coefficients scale smoothly with the number of proxy parameters.

Two practical consequences. First, the peak is not a property of the verifier alone. It sits at the intersection of verifier quality and optimization pressure, so a better verifier and a larger budget are complements while a larger budget against a fixed verifier is a trap. Second, because the curve's form depends on the optimization method, the algorithm is a design variable. Best-of-$n$ and a deep guided search do not walk the same curve.

Best-of-$n$ against a fixed verifier is a Goodhart machine with a measurable gear ratio. The scaling law tells you which way the dial turns.

## Making the verifier the thing you measure

If the verifier sets the ceiling, the verifier is the thing to instrument.

Ackermann and coauthors ([arXiv:2602.18037](https://arxiv.org/abs/2602.18037), 2026) approach it from the training side, and the mechanism transfers. They derive a connection between reward accuracy and the flatness of the optimum, then use gradient regularization to bias updates toward flatter regions where the reward is more accurate. Gradient norm and reward accuracy are empirically correlated in their setting, and explicit regularization beats a KL penalty across their experiments.

The regularizer is less interesting than the observation underneath it: verifier accuracy is not uniform across the space you are optimizing in. My hypothesis, and I have not seen it tested for search: the same non-uniformity applies to the candidate space at inference time, which argues for keeping the search where the verifier has been measured.

Two measurement habits follow. The rubric study offers a verifier-free diagnostic, a self-internalization gap computed from policy log-probabilities that tracks reference-verifier quality and detects when a policy trained against a weak verifier stops improving. Roth and coauthors ([arXiv:2605.20744](https://arxiv.org/abs/2605.20744), 2026) go further, embedding hack opportunities directly into the environment so a hack is verifiable by design rather than reconstructed from trajectories afterwards.

The cheap version you can run today is a selection gap between your training verifier and an independent one.

```python
def selection_gap(candidates, v_train, v_holdout, k):
    """Top-k overlap between the verifier you optimize and one you do not."""
    train_top = sorted(candidates, key=v_train, reverse=True)[:k]
    held_top = sorted(candidates, key=v_holdout, reverse=True)[:k]
    return len(set(map(id, train_top)) & set(map(id, held_top))) / k
    # low overlap: the ordering you search against is not the one you believe
```

Track it next to accuracy. Accuracy alone cannot tell you whether the policy improved or the ordering drifted, and those demand opposite responses. A policy that improved needs more budget. A drifted verifier needs replacing, and more budget makes it worse.

## Setting the budget

The engineering conclusion is short. Search depth is not a free parameter. It is a function of verifier error: raise per-step discrimination or shorten the depth. Doing neither while adding samples is how a system ends up behind plain sampling at the exact moment it looks most elaborate.

Raising discrimination means better signal per step: process supervision, executable checks, invariance requirements. Shortening depth means more local subgoals with locally checkable results. Both are work on the verifier. Neither is work on the sampler, which is where the compute budget usually goes.

I would put one number on a dashboard first: how often the verifier's preferred candidate is the one a careful independent check would also prefer, on the distribution you are actually searching over. If it is unknown, the depth is a guess.

The verifier is the part of the system you are optimizing against, whether or not anyone wrote it down. The search will find out what it measures.

## Sources

- Fei Yu, Yingru Li, Benyou Wang. Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning. 2025. [arXiv:2502.00271](https://arxiv.org/abs/2502.00271)
- Lukas Helff et al. LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking. 2026. [arXiv:2604.15149](https://arxiv.org/abs/2604.15149)
- Anas Mahmoud et al. Reward Hacking in Rubric-Based Reinforcement Learning. 2026. [arXiv:2605.12474](https://arxiv.org/abs/2605.12474)
- Leo Gao, John Schulman, Jacob Hilton. Scaling Laws for Reward Model Overoptimization. 2022. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- Johannes Ackermann et al. Gradient Regularization Prevents Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards. 2026. [arXiv:2602.18037](https://arxiv.org/abs/2602.18037)
- Amit Roth et al. Hack-Verifiable Environments: Towards Evaluating Reward Hacking at Scale. 2026. [arXiv:2605.20744](https://arxiv.org/abs/2605.20744)
