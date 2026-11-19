---
title: "Agentic Evaluation Is Trajectory Evaluation"
description: "An agent that reaches the right answer by a broken path has failed silently. Score the intermediate steps, not only the answer."
pubDatetime: "2026-01-17T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Agents
  - Evaluation
categories:
  - Math
---

Consider a refund agent that closes a support ticket correctly. It reached the right customer by the wrong path. It opened an account record that shared the customer's surname, read a ledger that happened to contain the right purchase, and issued a refund for the right amount by coincidence. Every visible output is correct. The path underneath is a chain of near misses.

Give the same agent a customer whose surname is unique. It opens the wrong record, reads an empty ledger, and either refunds nothing or refunds the wrong person. The final-answer score drops, the dashboard shows a regression, and someone files a ticket against the retrieval tool. The score reported luck, not capability.

This failure mode is not exotic. It appears whenever a task admits more than one route to the same string, which is most tasks worth automating.

## The Answer Is a Coarse Function of the Trajectory

Let a run be a trajectory $\tau = (s_0, a_1, o_1, \ldots, a_T, o_T)$, where each $a_t$ is a tool call or a message and each $o_t$ is the observation returned. The final answer is $f(\tau)$ for some function $f$ that the runtime defines, usually a parse of the last assistant message.

Final-answer accuracy over $N$ tasks is

$$A_{\text{final}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\, f(\tau_i) = y_i^{*} \right].$$

The function is doing its job. The trouble is that it is many-to-one. For a fixed answer $y$, the preimage $\mathcal{T}(y) = \{\tau : f(\tau) = y\}$ contains the clean run, the lucky run, the run that made nine redundant calls, and the run that wrote to a table it should not have touched. $A_{\text{final}}$ gives every member of that set the same value. A statistic that is constant on a set carries no information about which member you are looking at, and the members differ in exactly the properties that decide whether the agent generalizes.

So the accurate version of the complaint is narrower than "the answer does not matter". The answer is a single sample from a distribution the agent will meet again. The trajectory is the process that generates the distribution. If you only score the sample, you are fitting to a draw.

## Three Ways a Correct Answer Hides a Broken Run

Most silent successes fall into three shapes, plus a fourth that is a capability rather than a defect.

**Lucky shortcut.** The agent resolves a referent by coincidence: the wrong record with a matching key, the wrong file with a matching name, the wrong endpoint that returns a superset. Each action is well formed and each subsequent step is correct given what came back. The run is valid conditional on a false premise, and the premise never gets checked because the answer matches.

**Compensating waste.** The agent takes a path several times longer than the task needs. It retries a failing call with the same arguments, loops through an empty search, then stumbles onto the result. The answer is right, and the cost, the latency tail and the rate-limit exposure are all wrong. None of that appears in the accuracy term.

**Unsafe side effects.** The task completes and the environment is worse: a scratch file left in the output directory, a probe row committed, a notification sent twice, a cache invalidated for a tenant that was not the target. Task success and system integrity are different questions that happen to share an answer.

**Recovery.** An error arrives and the next action responds to it - different arguments, different tool, narrower query. A run that recovers demonstrates a skill that a run which never errs cannot show. A run that "recovers" by retrying blindly until something works is indistinguishable in the answer column and is a different agent.

AgentRewardBench made the three-shape problem concrete by asking annotators to label trajectories along several axes instead of one. Expert reviewers covered 1,302 trajectories drawn from 5 benchmarks and 4 models, scoring success, side effects and repetitiveness separately ([arXiv:2504.08942](https://arxiv.org/abs/2504.08942), Lù et al., 2025). The axes exist because a single success bit does not describe a run.

## What a Trajectory Score Has to Cover

Six dimensions recur across agent tasks. The table is the smallest version I would defend.

| Dimension | What it reads | What answer accuracy says about it |
|---|---|---|
| Referent grounding | whether each call resolved to the intended entity | nothing |
| Argument support | whether each parameter traces to a prior observation | nothing, unless the error changed the answer |
| Policy compliance | whether order and constraints held | nothing |
| Recovery | whether the step after an error addresses that error | nothing |
| Efficiency | steps, tokens and wall clock against a budget | nothing |
| Side effects | what changed in the environment beyond the task | nothing |

None of these is free. Policy compliance is the most expensive to build honestly, and SOPBench is the clearest example of what it costs: 167 tools across 7 customer service domains, more than 900 generated test cases, and rule-based verifiers derived from the original SOP source rather than from a model's opinion ([arXiv:2503.08669](https://arxiv.org/abs/2503.08669), Li et al., 2025). The verifiers work because the procedure already existed as code. Across the 18 models tested, even strong models sat in the 30% to 50% band on adherence, and the authors report that agents could be talked out of the procedure by a user turn. That result is legible only because compliance was checked at each step. An answer-only score would have recorded those runs as ordinary successes or ordinary failures and explained neither.

## Against the Golden Trajectory

The obvious response is to record one correct trajectory per task and diff against it. That is the first thing most teams build, and it is the wrong instrument.

A golden path encodes one implementation's habits. An agent that finishes in fewer steps with a different tool, or takes a legitimate detour around a flaky endpoint, gets penalized for being better. What you end up measuring is conformity to a reference, and the reference is not the task.

The annotation cost is also not a detail. Comparing a trajectory against a ground-truth trajectory is, in the words of the TRACE authors, fundamentally limited because annotating every valid trajectory is prohibitively expensive. The number of valid paths grows with task length, and the labels you would need grow with it.

There is a second-order trap: the gate reward. If only the annotated path earns credit, the gradient points at that path's inefficiencies. You teach the agent to reproduce a reference rather than to solve a problem.

The alternative is to score properties instead of paths. TRACE drops the reference and scores a trajectory on efficiency, hallucination and adaptivity, using an evidence bank that accumulates what earlier steps established, so each step is judged against what the run had already earned the right to believe ([arXiv:2510.02837](https://arxiv.org/abs/2510.02837), Kim et al., 2025). Reference-free scoring trades one kind of precision for coverage: it does not compare against a specific route, so it will not object to a path that a reference would have rejected.

One more piece of the contrarian case: do not assume an LLM judge rescues you. AgentRewardBench evaluated 12 judges over those 1,302 trajectories and found that no single judge was best across all five benchmarks, and that the rule-based evaluation common in web benchmarks tends to underreport success. An automatic judge is a component with its own error profile. It needs its own validation set, and its agreement with humans belongs next to the agent's score, not in an appendix.

## Logging Is the Hard Part

You cannot score a property you did not record, and most agent logs record the final message and the tool names. The minimum useful record is per step: the tool, the arguments as the model emitted them, the arguments as your parser used them, the response status, a digest of the observation, the step index, wall clock, and which earlier step this one retries.

The distinction between raw and parsed arguments matters more than it looks. A model can emit a name that your resolver normalizes to a different entity, and that rewrite is the exact place where a grounding failure hides. Keep both.

```json
{"run":"r_8812","step":4,"tool":"crm.find_customer","args_raw":"{\"name\":\"Nakamura\"}","args_used":{"name":"Nakamura","normalized_from":"nakamura"},"status":"ambiguous","n_matches":3,"chosen":"c_4471","obs_digest":"sha256:1f9c...","started_s":0.812,"duration_ms":214,"retry_of":null}
```

"ambiguous" with three matches and a chosen record is the lucky-shortcut signature in one line. You do not need a judge to see it; you need the field.

## Aggregate with a Gate, Not a Mean

If you average the dimensions, a strong answer term hides a policy violation. Process failures are not additive quantities. A policy violation does not become acceptable because nine other checks passed. So make the score a product of the answer indicator and the weakest dimension:

$$S(\tau) = \mathbb{1}\left[\,f(\tau) = y^{*}\right] \cdot \min_{d \in D} s_d(\tau).$$

The minimum has a real cost: one noisy verifier can zero a good run. The fix is not to abandon the gate but to split $D$. Policy violations, grounding failures and out-of-scope side effects are gates. Efficiency and recovery quality are graded terms. Report the gates as pass rates and the graded terms as distributions, and keep them separate.

My view is that a single composite number is a reporting convenience, not a measurement, and it should never be the only number in a table. What I would report for a release: the gated pass rate, the fraction of passes that used a non-canonical path, the recovery rate after a tool error, and the cost per successful run. Those four move independently, and that independence is the information.

## What Changes When You Measure This Way

Scores drop the first time you switch. Runs that used to pass now fail on a gate. That drop is the old metric's blind spot becoming visible, not a regression, and anyone reading the dashboard needs to hear it before the number moves.

The training consequence is the part I care about most, and it is inference from the metric's shape, not something I measured here. If the score reads only the last message, the cheapest way to raise it is to make the agent luckier, not better: shorter paths that skip verification, retries that terminate on any answer, tool calls that find the right entity often enough. A reward built on the final answer cannot distinguish a policy that generalizes from one that gets the sample right, so optimization is free to choose the second.

Three costs are real. Trajectory scoring is slower and noisier than answer matching. Verifiers rot as the environment changes, so they need the same maintenance as the agent. And some state you want is simply not observable from inside the process, which means part of every score is an estimate, and the honest report says which part.

Keep final-answer accuracy. It is cheap, it catches gross regressions, and it is the number users experience. Just stop making it the dependent variable. The refund agent that closed the right ticket for the wrong reason was never healthy. The metric that said it was had only read the last line.

## Sources

- "AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories", [arXiv:2504.08942](https://arxiv.org/abs/2504.08942) - Lù, X. H. et al. (2025). AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories.
- "SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints", [arXiv:2503.08669](https://arxiv.org/abs/2503.08669) - Li, Z. et al. (2025). SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints.
- "Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents", [arXiv:2510.02837](https://arxiv.org/abs/2510.02837) - Kim, W. et al. (2025). Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents.
