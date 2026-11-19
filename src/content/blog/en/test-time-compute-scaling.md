---
title: "Test-Time Compute Is a Budget, Not a Button"
description: "More inference compute only helps where a task admits search. Sampling, longer thinking and verification are different purchases with different decay rates."
pubDatetime: "2025-01-18T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Test-Time Compute
  - Reasoning
categories:
  - Math
---

## The dial that survives every postmortem

Training is finished by the time the request arrives. The weights are fixed. The reasoning budget is not, and that asymmetry explains why "just let it think longer" has become the default answer to a bad failure rate. It is the only knob left in the room.

A knob is not a policy. Consider two tasks that both look like reasoning on a dashboard. The first is a single-hop question over material the model has partly memorised. The second is a competition problem with a checkable answer and a wide space of wrong approaches. On the first, multiplying the token budget by ten usually buys noise, because there is no intermediate artifact to search over; the model either lands the fact or it does not. On the second, the same multiplication can move accuracy by tens of points, and the move is uneven, because some problems fall at four samples and some are still unsolved at two hundred.

Averages over a mixed benchmark hide both effects. The paper number is an average over a task distribution you do not have. The endpoint number is what you pay for.

I want to make the case for treating inference compute as a budget line with a marginal price, allocated per request, rather than a global setting applied with a slider. That framing changes what you build. It moves the question from "how high should we turn it up" to "where is the return".

## Three different purchases share one label

People say "more test-time compute" and mean at least three different mechanisms, with different costs and different failure modes.

Sampling more draws $N$ independent attempts and aggregates them, usually by majority vote or by scoring. It buys coverage. If a single attempt succeeded with probability $p$ and the attempts were genuinely independent, coverage would be $1-(1-p)^N$. The independence assumption is the weak part: samples from one model at one temperature fail in correlated ways, so measured coverage sits below that prediction. Sampling remains the cheapest search primitive, because it needs no verifier and no tree.

Thinking longer runs one trajectory for more tokens, with self-checking inside the trace. This is the behaviour of the o1 family and QwQ-32B-Preview. It buys depth on problems whose difficulty is sequential, where each step constrains the next. It buys nothing when the first step is a wrong framing, because the trace elaborates that framing instead of questioning it. That limitation is real, and it is why long traces sometimes fail more confidently rather than merely more slowly.

Verifying harder scores candidates, prunes them and backtracks. It buys precision, and it requires an external signal: a test case, a symbolic checker, a process reward model. rStar-Math is the cleanest published example I know of this being the mechanism rather than the marketing. The authors run Monte Carlo tree search over a policy model guided by a process reward model, and report MATH scores moving from 58.8 to 90.0 for Qwen2.5-Math-7B and from 41.4 to 86.4 for Phi3-mini-3.8B, with a 53.3 percent average on AIME. Those are search results. The tree does the work, not the trace length.

The three are not substitutes. Verification with a weak sampler has nothing worth selecting. Sampling without a verifier cannot tell a correct minority from a confident wrong majority. Longer thinking without either is a single trajectory with no exit.

| Purchase | What it buys | Cost shape | Fails when |
|---|---|---|---|
| Sample $N$ times | coverage of the solution space | linear in $N$, parallelisable | failures are correlated |
| Longer trace | sequential depth | linear in tokens, serial | the first framing is wrong |
| Verified search | precision under a scorer | superlinear, needs a checker | the scorer is biased |

## The contrarian part: search is the precondition, not the axis

The standard framing puts test-time compute on the same shelf as parameters: another scaling axis, another curve to extrapolate. The survey work published in this window organises the field that way, with repeated sampling, self-correction and tree search as the deliberate mechanisms layered on top of intuitive inference. I think the framing is incomplete in a way that costs money.

The axis only exists where the task admits search. I mean three properties at once. There is a space of partial solutions. Partial solutions can be scored better than chance. Scoring is cheaper than generating.

Competition mathematics satisfies all three. A step can be checked, and a bad branch can be dropped. Verified code generation satisfies all three, which is why unit tests are such an effective compute multiplier. Retrieval over a fixed corpus satisfies the first weakly and the third sometimes, depending on whether the retriever is any good. A single-hop fact question satisfies none of them, so extra compute just resamples the same ignorance.

EMMA is the useful negative result. The authors built a multimodal benchmark that requires genuinely cross-modal, multi-step work, and report that chain-of-thought prompting and test-time compute scaling both underperformed on it. I read that as a boundary marker rather than a refutation. Extra compute did not create a search space the task did not have.

So the classification step comes before the budget step. Ask whether a wrong first step can be detected and discarded for less than the cost of generating it. If it can, extra compute has somewhere to go. If it cannot, you are buying more of the same failure with better latency numbers on the invoice.

## Where the marginal return goes

Two things decay, and they decay for different reasons.

Coverage decays geometrically. Differentiating the coverage expression gives

$$\frac{d}{dN}\left[1-(1-p)^N\right] = (1-p)^N \ln\frac{1}{1-p},$$

which shrinks by a constant factor per additional sample. Going from one sample to eight is often a large gain. Eight to sixty-four is usually a small one. Sixty-four to five hundred is usually inside the confidence interval. The shape is not a law of nature; it is what independence plus a fixed $p$ implies, and correlated samples make it worse.

Verification decays against a ceiling, which is the part people miss. If the selector picks the correct candidate with precision $\gamma$ when it declares a winner, accuracy plateaus near $\gamma$ no matter how many candidates you feed it. Sampling past that point only gives the selector more chances to be confidently wrong. That is why a good verifier is worth more than a larger sampling budget, and why a mediocre verifier is worse than plain majority vote at large $N$.

The ceiling is not hypothetical. The authors of ReARTeR list early-step bias in process reward model scores among the problems they had to design around, next to biased training data and a mismatch between scalar rewards and natural language explanations. If the scorer is least reliable exactly where the search branches most, then the pruning happens at the worst moment in the tree. That is structural, not a tuning problem.

There is a second ceiling that has nothing to do with the model. Serial thinking costs latency. Parallel sampling costs throughput. On a 20 second p99 budget, sixteen parallel samples and a vote can fit while a sixteen thousand token trace cannot. Those two budgets are not interchangeable at the serving layer, and treating them as one number is how a capacity plan becomes an incident.

## Cheap first pass, targeted second pass

Uniform allocation is the default and the worst option. Difficulty is heavy-tailed, so a fixed per-query budget overpays on the easy majority and underpays on the tail that determines your failure rate.

The shape I would build has two passes. Answer cheaply first: one attempt, or a small number at low temperature. Then gate. If the cheap attempts agree, ship the answer. If they scatter, escalate in order of cost: more samples, then a verifier pass, then a longer trace.

The gate should be a disagreement detector rather than a difficulty classifier. Disagreement is free, it is observable at request time, and it correlates with the outcome you care about. A difficulty classifier needs labels and drifts as the distribution moves.

Sketch of the policy:

```
ans = sample(k=2, temp=0.2)
if agree(ans): return ans[0]
cands = sample(k=16, temp=0.8) + ans
scored = verifier.score(cands)
if max(scored) > tau: return argmax(scored)
return longest_trace(cands[0])
```

Three properties make this pay. The easy majority exits on the first branch, so average cost stays near the cheap path. The expensive branch is funded by those savings rather than by a separate budget negotiation. And escalation is conditioned on evidence from this item, so one bad batch cannot quietly consume the allocation for the rest.

Here I should separate what is observed from what is not. The components are documented individually: sampling raises coverage, process reward models generalise better than outcome-only scoring on multi-step tool use, and search over a guided policy beats a single pass on competition math. The composition, meaning a cheap pass, a disagreement gate and targeted escalation, is my hypothesis about the right engineering shape. I have not seen a public benchmark that isolates the gate's contribution from the extra samples it triggers. That experiment is cheap to run: hold total budget constant, compare uniform against gated, and put cost per solved item on the vertical axis.

One supporting observation from outside the math setting. The medical reasoning replication work reports that inference-time scaling improved their results and that task complexity tracked the required chain length. That is consistent with the allocation argument. If the demand for length varies by item, a fixed length is mispriced by construction.

## What I would measure before turning the dial

Four numbers, and none of them is average accuracy on the eval set.

Accuracy against budget, sliced by task family. The curve is the artifact. A single point measured at the maximum budget tells you what the maximum budget does, which is not the decision in front of you.

Verifier precision, measured directly. Take a sample of items, generate candidates, and count how often the selector's pick is actually correct when it claims to be. That number caps everything downstream, and it is usually lower than teams assume before they check.

Cost per solved item, not cost per query. The denominator matters. If escalation converts a fifth of the failures into successes, cost per query rises while cost per solved item can fall. Optimising the first metric is how a team talks itself out of a system that works.

Gate quality. Precision and recall of the disagreement signal against the ground truth question of whether escalation would have helped. Without that, you cannot tell whether the router is routing or just adding variance.

The dial is real. There is genuine headroom in spending more compute at inference time and the recent results make that hard to argue against. The headroom is not the same as a return, and a return is not the same as a return on your distribution. Treating inference compute as a budget is the only framing I have found that survives contact with a production curve.

## Sources

- Yixin Ji et al. A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning. [arXiv:2501.02497](https://arxiv.org/abs/2501.02497), 2025. https://arxiv.org/abs/2501.02497
- Fengli Xu et al. Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. [arXiv:2501.09686](https://arxiv.org/abs/2501.09686), 2025. https://arxiv.org/abs/2501.09686
- Xinyu Guan et al. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking. [arXiv:2501.04519](https://arxiv.org/abs/2501.04519), 2025. https://arxiv.org/abs/2501.04519
- Zhongxiang Sun et al. ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding. [arXiv:2501.07861](https://arxiv.org/abs/2501.07861), 2025. https://arxiv.org/abs/2501.07861
- Yunzhuo Hao et al. Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark. [arXiv:2501.05444](https://arxiv.org/abs/2501.05444), 2025. https://arxiv.org/abs/2501.05444
- Zhongzhen Huang et al. O1 Replication Journey, Part 3: Inference-time Scaling for Medical Reasoning. [arXiv:2501.06458](https://arxiv.org/abs/2501.06458), 2025. https://arxiv.org/abs/2501.06458
- Vaskar Nath et al. ToolComp: A Multi-Tool Reasoning and Process Supervision Benchmark. [arXiv:2501.01290](https://arxiv.org/abs/2501.01290), 2025. https://arxiv.org/abs/2501.01290
