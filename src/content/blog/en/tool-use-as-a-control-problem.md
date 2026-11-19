---
title: "Tool Use as a Control Problem"
description: "Selecting a tool is an action under uncertainty: the policy must price the call, and permissions and reversibility belong in the action space."
pubDatetime: "2026-02-07T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Tool Use
  - Agents
categories:
  - Math
---

A request arrives that a single subtraction would settle. The agent has a catalog of tools in front of it. It calls search, then fetch, then fetch again on a page the first call already covered, and answers correctly. No single step was wrong. The trajectory was.

Most writing on tool use treats that as a prompting problem or a training problem. I read it as a control problem, because control problems come with quantities you can write down. If you cannot name the quantity the policy estimates, you cannot tell whether a fix worked.

## The call is the action, and actions have prices

Strip the interface away and an agent at step $k$ is a controller. It observes a state $s_k$ holding the task, the conversation, and everything the tools have returned. It picks an action

$$a_k = (t, \theta, \kappa),$$

where $t$ is a tool or the no-op, $\theta$ are the arguments, and $\kappa$ is the permission scope the call runs under. Then the environment returns an observation and the state moves. That is a control loop. The part worth staring at is that the action has a price even when it works.

Let $p$ be the probability the call improves the outcome, $g$ the gain when it does, $h$ the loss when it does not, $c$ the cost of making it, and $\rho \in [0,1]$ the fraction of the effect that cannot be undone. With $\lambda$ as the price of irreversibility, the decision margin for one candidate is

$$\Delta(s,a) = p\,g - (1-p)\,h - c - \lambda\rho .$$

Call when $\Delta > 0$, and among the candidates that clear zero, take the largest margin. Solving for $p$ gives the confidence the policy needs:

$$p^{\star} = \frac{c + \lambda\rho + h}{g + h}.$$

Cost and irreversibility raise the bar directly. Asymmetry raises it faster, because $h$ sits in numerator and denominator with opposite signs. Set $h = 10g$ and ignore cost: $p^{\star} = 10/11$, so the call must be right about 91 times in a hundred to be worth making. Set $h = 100g$ and it is $100/101$. The relevant target is accuracy sufficient for the payoff in front of it, which is a higher bar than general accuracy.

| Downside relative to upside | Confidence required | What it implies |
| --- | --- | --- |
| $h = g$ | 0.50 | Any signal better than a coin is worth using |
| $h = 10g$ | 0.91 | Only near-certain calls are rational |
| $h = 100g$ | 0.99 | The call should usually not be made |

Drop $h$ and $\lambda\rho$ and you get a selector that grades competence and nothing else. Drop $c$ and you get an agent that calls everything, because a free action with positive expected gain should always be taken. Each term is a design decision, and omitting one is a decision too.

## Estimating the value before the call

The margin is only as good as $p$. Nothing in the formula says how to obtain it, and here tool use stops being a routing problem and becomes an estimation problem.

Li and coauthors, writing in February 2026, report a signal worth taking seriously. In long trajectories they observe a strong positive correlation between entropy reduction and high quality tool calls, and use entropy reduction as a supervisory signal. Their measured results are specific: sparse outcome rewards cut tool calls by 72.07 percent against the baseline average, and dense process rewards improved performance by 22.27 percent. [arXiv:2602.02050](https://arxiv.org/abs/2602.02050), Zeping Li et al., 2026.

That paper reports two fixes for two diseases. Excess calls are an efficiency problem and the sparse reward attacks it. Poor calls are a quality problem and the dense reward attacks that. The correlation is their observation. The reading that entropy reduction tracks "this call will resolve something" is their hypothesis. My opinion is that it holds for a narrower reason than the framing suggests: entropy drop measures how far the model's own distribution moved, and a call that moves nothing has probably returned redundant information. That is a claim about informational content, and utility is a separate quantity. The gap is where a confident agent calls a tool, receives a well formed page, and did not need it.

There is a structural problem underneath. You want $\hat p$ calibrated on the states you actually visit, and those states are selected by the policy. An agent that over-calls search generates a distribution heavy in search outcomes and thin in everything else, so its estimate for the tools it rarely picks is fitted on almost no data. This is off-policy evaluation in different clothes, and it explains how an agent can look calibrated in aggregate while being badly miscalibrated on the one action that matters.

## Permissions are a coordinate in the action space

Most deployments treat permissions as a filter applied after the policy produces a call. The policy names a tool, a separate layer checks it, and the call proceeds or is blocked. Betser and coauthors describe the failure this produces as unbalanced tool-driven agency. An agent can retain permissions it does not need, which they call excessive agency, and it can fail to invoke tools it does need, which they call insufficient agency. Their framework enforces per-step least-privilege access with adaptive filtering and status-aware validation, and on AgentDojo it reduces attack success while holding task performance. [arXiv:2601.12449](https://arxiv.org/abs/2601.12449), Roy Betser et al., 2026.

The structural argument is stronger than the result. If permission scope is part of the action, then $(t, \theta, \text{read})$ and $(t, \theta, \text{write})$ are distinct actions with distinct values, and the policy should rank between them. If permission is applied afterward, the policy assigns probability mass to actions it cannot take, and its ordering over the reachable ones is computed under a distribution that still contains the unreachable ones. That is a control policy with a projection bolted on, and projections do not preserve preference order. Constrain the action space before the policy sees it and the policy optimizes over what is reachable. Constrain after and you get a filtered sample from a distribution nobody optimized.

## Reversibility changes what a mistake costs

The $\rho$ term is the one usually left out, and it changes the shape of everything else. Read a file and a mistake costs a wasted call. Send an email, drop a table, or push to a protected branch and the mistake is in the world. You cannot average over retries, because the retry starts from a state you already damaged.

Mou and coauthors put the intervention before execution, which is the right place. Their guardrail model reasons over the interaction history, assesses request harmfulness and action-attack correlation, and returns a judgment plus feedback before the call runs. Their reported numbers: harmful tool invocations of ReAct-style agents drop by 65 percent on average, and benign task completion improves by roughly 10 percent under prompt injection. [arXiv:2601.10156](https://arxiv.org/abs/2601.10156), Yutao Mou et al., 2026.

The benign improvement is the number I would underline. A detector that only suppresses harmful calls can always be improved by blocking more, and eventually it blocks the task. Moving both numbers the right way means the guardrail separates two populations rather than sliding a threshold. Post-execution detection cannot do that, because the observation arrives after the state changed and there is nothing left to protect.

Reversibility is also what makes the threshold asymmetric in practice rather than only on paper. The table above says a reversible action with symmetric payoff should be taken on a coin flip. The same action made irreversible should not be taken at 90 percent confidence. Same model, same tool, only $\rho$ changed. A policy with no $\rho$ in its objective cannot express that, so it applies one threshold to both regimes.

## The contrarian case: overlap, not intelligence, is the failure mode

My position, and this is an argument rather than a finding. The default response to bad tool use is more tools, better descriptions, and more training on selection. I think much of it is a specification bug that training cannot fix.

Consider two tools with similar descriptions whose effects partially overlap. After the call the state is nearly the same either way, so $\Delta$ is nearly equal for both, and the difference is small against the noise in $\hat p$. A policy cannot be well calibrated on a distinction that does not matter, because nothing in the outcome teaches it. The signal about which of two near-identical tools was correct is close to zero, and more data does not make it sharper. Adding compute to a tie spends money without moving the quantity you care about.

The rule that follows is narrow. If two tools have the same effect and the same permissions, merge them. If they have the same effect and different permissions, keep both and let permission decide, since it is now the only real difference. If they have different effects, keep both and write descriptions that separate effects rather than implementations.

Insufficient agency is the counterexample that keeps this honest. Collapsing overlap too aggressively deletes capability, and an agent that cannot reach the tool it needs fails in a way that permission clarity cannot repair. The rule is fewer tools that do the same thing, not fewer tools overall.

## What to instrument

The margin is a specification, so it prescribes what to log. For every call, record the estimate the policy held, the realized effect, the cost, and the permission scope it ran under. Those four make the calibration curve computable per tool, and the per-tool breakdown is where the diagnosis lives. Aggregate calibration hides the pattern that matters, which is usually one tool over-called and one under-called.

Separate three failure types before changing anything. Wrong tool chosen, right tool with wrong arguments, and right call that harmed. They have different repairs. The first is an action-space problem, the second a schema problem, the third a $\rho$ problem. A single "tool use accuracy" number blends them into something you cannot act on.

```python
def should_call(p, gain, harm, cost, rho, lam):
    """One candidate action. Returns the margin and the p it needed."""
    margin = p * gain - (1.0 - p) * harm - cost - lam * rho
    return margin, (cost + lam * rho + harm) / (gain + harm)
```

The return line carries the point. It says the bar is a property of the payoff and not of the model, which reframes where effort goes. If the bar is 0.99 because the action is irreversible, more prompting will not get you there, and the right move is to make the action reversible or split it into a reversible proposal and a separately confirmed commit. If the bar is 0.50 because the tool only reads, call it and stop deliberating.

An agent that reasons about whether to act is doing what a feedback controller does. It estimates a state, prices an intervention, and acts where the margin clears the cost. Tools are the actuators, permissions bound the reachable set, and reversibility sets the price of being wrong. None of that is new in control, and most of it is missing when a tool-using agent gets built.

## Sources

- Zeping Li, Hongru Wang, Yiwen Zhao, Guanchu Chen, Yixia Li, Keyang Chen, Yixin Cao, Guangnan Ye, Hongfeng Chai and Zhenfei Yin, [Rethinking the Role of Entropy in Optimizing Tool-Use Behaviors for Large Language Model Agents](https://arxiv.org/abs/2602.02050) (2026).
- Roy Betser, Amit Giloni, Shamik Bose, Sindhu Padakandla, Chiara Picardi, Lidor Erez and Roman Vainshtein, [AgenTRIM: Tool Risk Mitigation for Agentic AI](https://arxiv.org/abs/2601.12449) (2026).
- Yutao Mou, Zhangchi Xue, Lijun Li, Peiyang Liu, Shikun Zhang, Wei Ye and Jing Shao, [ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback](https://arxiv.org/abs/2601.10156) (2026).
