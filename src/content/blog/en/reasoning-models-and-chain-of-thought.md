---
title: "Reasoning Models and the Shape of a Thought"
description: "Chain of thought is extra tokens used as working memory, and reinforcement learning on verifiable answers never checks whether the trace is faithful."
pubDatetime: "2025-02-15T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Reasoning
  - Chain of Thought
categories:
  - Math
---

A reasoning model answering a math problem spends most of its output on text that is not the answer. The final line may be four characters. The two thousand tokens before it are intermediate work, and every one of them is decoded at the same speed and billed at the same rate. On a hosted endpoint that text is the dominant term in both latency and price.

Those tokens often help. The useful question is what they are doing when they help, because that decides what you should measure, where you can cut, and when a confident-looking trace deserves suspicion.

## The extra tokens are working memory

A transformer does a fixed amount of serial computation per token. Depth gives it a stack of layers, but the route through them is fixed, so the number of sequential transformations available for one token's representation is bounded by the layer count. That bound does not loosen when the problem gets harder.

Chain of thought moves the work into the sequence. If the model emits $T$ tokens before committing to an answer and the stack has $L$ layers, the longest dependency path from input to final answer spans up to $L \cdot T$ sequential transformations instead of $L$. Each token is generated conditioned on everything before it, so a partial result written early becomes an input to every later step. The context is the only writable memory the model has during a forward pass, and a trace is a way of using it as a scratchpad.

That framing predicts something specific. The value should come from intermediate results being written down and read back, not from length as such. A trace that restates the problem four times at increasing verbosity adds tokens without adding state, and it changes nothing downstream except the position index. A short trace that records a partial result the model would otherwise recompute is doing the work. Length is the price of the mechanism, not the mechanism.

## What the reward actually touches

The DeepSeek-R1 result is the clearest public example of where the training signal lands. DeepSeek-R1-Zero was trained with large-scale reinforcement learning and no supervised fine-tuning, and the reasoning behavior came out of that process rather than out of demonstrations. In that setting the reward comes from checking the final answer against a target, which is only possible on problems whose answers can be checked mechanically.

So the gradient sees the trace entirely through the answer. For a policy $\pi_\theta$ that emits a trace $\tau$ and an answer $\hat{y}$ against target $y$,

$$\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\Big[\, r(\hat{y}, y)\,\nabla_\theta \log \pi_\theta(\tau) \,\Big].$$

The reward sits outside the trace. It multiplies the log-probability of the whole sampled sequence, so every token shares one credit assignment and the only condition the objective imposes is that the sampled trace ended at a correct answer. A token that could be deleted without changing the answer is not penalised for being there and not rewarded for being absent. The shape of the trace is left free.

R1's published observations fit that reading. The pure-RL model developed reasoning behavior that emerged from the optimization rather than from demonstrations, and it also developed poor readability and language mixing, which the team addressed in R1 by adding cold-start data and multi-stage training before the RL stage. Unreadable traces are not a failure of the reward. They are what the reward permits, because nothing in the objective asked for readable ones.

| Signal | Constrains | Leaves free |
|---|---|---|
| Final answer match | Whether the trace ends at a correct answer | How the trace gets there |
| Format reward | Surface structure such as delimiters | Whether structure reflects computation |
| Length budget | Tokens spent | Which tokens are dropped |
| Process reward | Plausibility of individual steps | Whether steps are load-bearing |

## The contrarian case: length is a symptom, not a control

The standard framing treats reasoning as a quantity you can buy. Longer traces score better on hard benchmarks, so the story runs, thinking longer is the mechanism, and the work is to scale test-time compute. That framing is incomplete in a way that sends engineering effort to the wrong place.

Part of the length-accuracy correlation is a property of the data. Hard problems produce longer traces and lower accuracy, easy problems produce short traces and high accuracy, and pooling them makes length look predictive when it is partly carrying difficulty. Within a narrow difficulty band the relationship is much weaker. My reading is that most public discussion quotes the pooled correlation and then treats it as a causal lever, which is the step that does not hold.

The cleaner counter-evidence comes from work that treats length as a cost to reduce rather than a resource to grow. Arora and Zanette train reasoning models with RL to allocate inference-time compute according to task complexity, rewarding the model for shedding unnecessary overhead while holding accuracy, and report substantial cost reductions with most of the accuracy preserved across two open-weight models, with one hyperparameter sweeping the tradeoff. If a large share of trace tokens were load-bearing, that tradeoff would not exist in that form. Its existence is evidence that a large share of the extra text was slack.

There is also a mechanical reason length cannot pay forever. Take each step in a trace as correct with probability $q$. If errors compounded independently, the chance a trace of $T$ steps survives would be $q^T$:

$$\Pr[\text{correct}] \;\approx\; \prod_{t=1}^{T} q_t \;=\; q^{T}.$$

This is a toy model and real traces are not independent step chains. A model can notice an earlier mistake and repair it, which is arguably what the long traces do. The model is still worth keeping because it names the two forces in tension. More tokens buy more sequential computation and also more chances to commit to a wrong intermediate result. Whether the net is positive depends on the model and the task, and no general statement about length settles it.

## A trace can be fluent and not be reasoning

The trace comes from the same next-token distribution that produces prose, so it inherits prose's surface properties. It reads as an explanation. Fluency is not evidence that the computation behind the answer happened in the text.

Three shapes are worth separating, and these are my categories rather than settled ones. The first is post-hoc rationalisation, where the model commits early and the remaining tokens defend the commitment. The second is local validity with a global error, where each step is defensible alone and the composition fails because a quantity was defined wrongly at the start. The third is omission, where the trace is coherent and the answer is right and the step that actually decided it is absent, because the model used a cue it never reported. Only the second tends to look wrong while you read it.

A step is load-bearing if removing it changes the answer, which gives an operational test. For a trace with steps $s_1, \dots, s_T$,

$$\Pr\big[\hat{y} \mid \tau\big] \;\ne\; \Pr\big[\hat{y} \mid \tau \setminus s_i\big].$$

The size of the gap is the weight that step can claim. Run it over a sample of steps and you get a map of where the answer came from, which tells you more than reading the trace and judging whether it sounds right.

The Meta-CoT work by Xiang and coauthors is useful framing for this. They separate the chain of thought a model writes from the reasoning process that would produce it, argue that current traces do not represent the underlying search, and propose process supervision, synthetic data and search as routes to traces that do. Their claim is structural. It is a hypothesis about mechanism, not a measurement of how unfaithful existing traces are, and the deletion test above is how you would check it on a particular model.

## What the benchmarks do and do not show

All three of these results come from tasks with mechanically checkable answers, and they inherit the limits of that setup.

For a pass rate $p$ over $n$ problems, the standard error of the estimate is roughly

$$\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}.$$

At $p = 0.8$ and $n = 500$ that is about 1.8 percentage points. Two checkpoints separated by two points on a set like that are not distinguishable from the marginals. A paired comparison on the same problems, scoring the discordant cases rather than the two totals, does better, and most careful papers do it that way. Most informal comparisons do not.

R1 reports results comparable to a named reference model on reasoning benchmarks, under that specific evaluation. The efficient-reasoning result covers two open-weight models and reports cost reduction with most of the accuracy intact, not all of it. Meta-CoT offers a framework, evidence consistent with in-context search, and a proposed training pipeline. None of the three demonstrates that a trace is faithful, and none separates reasoning from a good imitation of it. They show that answer-only reward produces capable models and that trace length can be compressed at modest cost. Those are narrower claims than the popular summary, and they are the ones the evidence supports.

## Evaluating a reasoning model without fooling yourself

Report accuracy at a fixed token budget. A model that wins by spending ten times the tokens has not won on the number you care about in production, and averaging over unconstrained lengths hides exactly that difference.

```python
for problem in benchmark:                       # same problems for both models
    out = model.generate(problem, max_tokens=BUDGET)
    ans = extract_answer(out)                   # parse, do not trust the last line
    record(problem.id, ans == problem.target, tokens=out.usage.completion)

p    = sum(correct) / n
se   = sqrt(p * (1 - p) / n)
disc = sum(a != b for a, b in zip(run_a, run_b))   # discordant pairs
```

Then read the traces that produced wrong answers, and check whether the wrong step is early or late. An error in the first third is a modelling or comprehension failure. An error in the last third, after a run of correct intermediate results, is usually a bookkeeping failure, and those respond to format constraints rather than to more tokens.

Two diagnostics are worth the compute. The deletion test on a sample of steps tells you whether the trace carries state or decorates an answer the model already had. A length sensitivity check tells you where the floor is: cap the model at a third of its natural length and see what breaks. If accuracy collapses, the tokens were carrying real work. If accuracy holds, they were not, and you have bought yourself a cheaper endpoint.

A long trace is a hypothesis about where the answer came from. It is not proof that the model reasoned, and it is not proof that it did not. The reward only ever checked the answer.

## Sources

- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), arXiv:2501.12948 (2025).
- Daman Arora and Andrea Zanette, [Training Language Models to Reason Efficiently](https://arxiv.org/abs/2502.04463), arXiv:2502.04463 (2025).
- Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden et al., [Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought](https://arxiv.org/abs/2501.04682), arXiv:2501.04682 (2025).
