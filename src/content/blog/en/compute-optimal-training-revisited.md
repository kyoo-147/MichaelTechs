---
title: "Compute-Optimal Training, Revisited"
description: "The twenty-tokens-per-parameter rule is a fitted summary, not a constant; once serving cost enters the objective, the optimum moves."
pubDatetime: "2026-11-07T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Scaling
  - Training
categories:
  - Math
---

The plan usually starts with a parameter count. A team picks the largest model that fits the serving hardware, multiplies by twenty, and writes the token budget. The slide says compute-optimal, and the number is not invented: it comes from a careful fit. It is also a fit to a different compute budget, a different tokenizer, and a different data mix, with the cost of serving the model left outside the objective.

The ratio is doing a lot of work in that decision and almost none of it is causal. It summarizes where the minimum of a fitted loss surface sat for one family of models, under one tokenizer, across one range of compute. Change the deployment and the minimum moves. Change the tokenizer and the unit underneath the ratio changes. Treat the number as a law and the budget lands in the wrong place, usually by buying parameters that get paid for again on every token served. The useful question is what allocation your loss surface and your serving bill imply.

## What the Fit Actually Says

Kaplan et al. (2020) fitted power laws for loss against parameters, data, and compute, and concluded that larger models are more sample-efficient, so a fixed compute budget should go to very large models trained on a relatively modest amount of data ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)). Hoffmann et al. (2022) ran a wider sweep, more than 400 models from 70M to 16B parameters and 5B to 500B tokens, and concluded the opposite about allocation: model size and training tokens should grow together, roughly in equal proportion ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)). Their 70B model, trained with four times more data than Gopher at the same training compute, outperformed the 280B Gopher across a range of tasks. The paper reports an average MMLU score of 67.5%, more than seven points above Gopher.

The fitted surface had the form

$$L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

with fitted exponents close to 0.34 for parameters and 0.28 for data. The familiar ratio of about 20 tokens per parameter is what that surface implies at the scale they studied. It is a derived number, not an input.

Two careful groups, two different answers about allocation. The difference was the shape of the sweep and the assumptions baked into the protocol. That disagreement is the first clue that the ratio is a fitted quantity. A physical constant does not change when the experimental design improves. A regression coefficient does.

## Tokens Are Not a Unit of Information

The token is a property of the tokenizer, not of the text. Limisiewicz et al. (2026) make this concrete. They trained 988 latent-tokenized models from 50M to 7B parameters at different compression rates, measured in average bytes per token, and went well beyond the 4.57 bytes per token of a common BPE tokenizer ([arXiv:2605.01188](https://arxiv.org/abs/2605.01188)). Their central finding is that in compute-optimal configurations, parameter counts scale proportionally with data measured in bytes, not in tokens. They also find that the optimal compression rate is not the BPE rate and falls as compute rises. The results hold for subword tokenizers and for languages other than English.

Read those two findings together and the ratio stops looking like a constant. If parameters track bytes, and the optimal bytes per token fall with compute, then tokens per parameter rises with compute even when the information budget per parameter is stable. The ratio is not merely fit-dependent; it drifts along the frontier as the tokenizer adapts.

This is why ratios do not travel between teams. A model at 20 tokens per parameter with a coarse tokenizer sees fewer bytes per parameter than one at the same ratio with a finer tokenizer. The runs are not comparable. Bytes per parameter is the portable statistic; tokens per parameter is a property of the pipeline.

The fit was not wrong. The unit was local.

## Putting Inference in the Objective

Now add the term the original fit leaves out. Let $N$ be parameters, $D$ training tokens, and $T$ tokens served over the model's life. With the usual accounting of about six floating-point operations per parameter per training token and about two per parameter per served token, the total is

$$C(N,D,T) = 6ND + 2NT.$$

Both terms scale with $N$, but only the first is paid once. For a long-lived service with heavy traffic, $T$ can be the larger number.

To see which way the optimum moves, fix a loss target $\ell$ and minimize cost subject to $A N^{-\alpha} + B D^{-\beta} = \ell - E = c$. The Lagrangian first-order conditions reduce to

$$1 + \frac{T}{3D} = \frac{\alpha A}{\beta B} N^{-\alpha} D^{\beta}.$$

With $T = 0$ the left side is 1, and the condition gives $D \propto N^{\alpha/\beta}$, which is the near-equal scaling they reported. With $T > 0$ the left side is larger, so the solution moves along the loss contour toward smaller $N$ and larger $D$. The direction does not depend on the exact exponents.

The intuition is accounting. A parameter is a fixed cost plus a per-token tax. Training pays once; serving pays forever. As the tax base grows, the optimum substitutes tokens for parameters.

This is why models are now trained well past the point that minimizes training loss per unit of compute. They are not ignoring the earlier result. They are minimizing a cost function that includes the deployment. The label "overtrained" is accurate only relative to the training-only objective. Relative to the bill, it may be the correct choice.

## The Optimum Moves With Deployment

The serving term is not a single number. It depends on batch size, sequence length, quantization, memory bandwidth, and how long the model lives.

| Deployment regime | Dominant cost | Where the optimum moves | Why |
|---|---|---|---|
| One-shot batch scoring | training | close to the training-only balance | the serving term is small next to the training run |
| High-volume, long-lived service | serving | smaller model, more training tokens | the per-token term scales with parameters |
| Latency-bound edge | memory and per-token latency | smallest model that clears the quality bar | weights must fit, and every token reads them |

Two details matter more than the table. At small batch sizes, token generation is limited by reading weights from memory rather than by arithmetic. The effective per-token cost then scales with weight bytes, which is worse than the $2N$ term suggests. Long contexts add attention cost that grows with sequence length, so the serving coefficient is a property of the workload, not a constant.

Liu and Liu (2026) make the same point from the other side. Their method parallelizes token-by-token weight updates during training by predicting activation gradients with an auxiliary network and supervising it with a consistency loss, with exactness for the sequential learner ([arXiv:2609.34174](https://arxiv.org/abs/2609.34174)). At deployment the predictor is discarded and the model updates natively, one token at a time, with forward and backward passes. If that is the plan, the per-token cost includes adaptation and scales with the parameter count. The boundary between training and inference moved, so the objective must move with it.

## The Contrarian Case: The Ratio Is a Symptom

The common practice is to treat 20 tokens per parameter as a target and tune the run until it lands there. I think that is backwards.

The ratio is an output of an allocation problem. Its inputs are the data distribution, the tokenizer, the architecture, and the serving plan. When you fix the ratio and vary everything else, you are not holding the important variable constant. You are holding a residual constant and letting the inputs drift. Two runs at the same ratio can land at different losses because the constants $A$ and $B$ depend on the data, not on the ratio. The number is a symptom of a well-specified allocation, not a cause of a good model.

"Compute-optimal" usually means optimal against a training budget and a training loss. A product needs a quality threshold at a total cost. Those are different constrained problems with different solutions. A team that reports a compute-optimal training run and then serves it at high volume has optimized the first problem and is paying for the second.

The review literature points at this gap. Dwyer (2026) surveys the move from scale maximization toward deliberate allocation of parameters, tokens, compute, and hardware, and flags that it remains unresolved whether scaling principles fitted on enterprise infrastructure transfer to smaller models and constrained environments ([arXiv:2610.06387](https://arxiv.org/abs/2610.06387)). That is the honest state of the evidence. The laws are empirical, the transfer is not established, and the deployment is part of the experiment.

## A Protocol That Survives Deployment

The alternative to importing a ratio is to measure a small version of your own.

First, train at three or more small scales on your own data and tokenizer, and fit the five constants. That is cheap next to the run you are planning, and it gives you a surface instead of a number. Second, write down $T$ before you choose $N$, counting retries, tool loops, and evaluation traffic. An agent workload can multiply $T$ by a large factor, which pushes the optimum toward smaller models more than most teams expect. Third, set the loss target from the product requirement, not from the lowest loss the budget can buy, and treat latency and memory as hard caps on $N$. Fourth, re-fit when the tokenizer, data mix, context length, or serving hardware changes. Those are the inputs; the ratio is downstream.

A small cost model is enough to make the trade visible:

```python
# Toy allocation model. A, B, alpha, beta, E come from your own pilot sweep.
def tokens_for_loss(n, target, A, B, alpha, beta, E):
    room = target - E - A / n**alpha
    return (B / room) ** (1 / beta)

def total_cost(n, served_tokens, target, A, B, alpha, beta, E):
    d = tokens_for_loss(n, target, A, B, alpha, beta, E)
    return 6 * n * d + 2 * n * served_tokens
```

Sweep $n$ over a plausible range, keep the points that satisfy the latency and memory caps, and take the minimum. The result is your allocation, given your assumptions. If it disagrees with 20 to 1, the disagreement is information.

## What I Take From This

The earlier work did not hand us a constant. It handed us a better coordinate system: a fitted surface, a procedure for finding its minimum, and a demonstration that the minimum is not where intuition put it. The coordinate system is still useful. The constant is not.

Inference cost belongs in the objective, not in a handoff to the deployment team. For a long-lived service it dominates. Once it is there, the optimum moves toward smaller models trained on more tokens, and it keeps moving as traffic grows. Test-time adaptation moves it again, because it changes what a token costs.

Pick the loss you need, cost the deployment you will actually run, and solve for the allocation. The ratio will fall out of the solution. If it lands near 20, the number was right for your case. If it lands at 200, that is not a mistake. It is a different bill.

## Sources

- [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) - Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models.
- [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) - Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models.
- [arXiv:2605.01188](https://arxiv.org/abs/2605.01188) - Limisiewicz, T. et al. (2026). Compute Optimal Tokenization.
- [arXiv:2609.34174](https://arxiv.org/abs/2609.34174) - Liu, B. and Liu, Q. (2026). GradLev: Token-Parallel Test-Time Training Via Costate Prediction.
- [arXiv:2610.06387](https://arxiv.org/abs/2610.06387) - Dwyer, J. (2026). Scaling Down the Scaling Laws: Parameter Efficiency and Compute-Optimal Training in Resource-Constrained Large Language Models.
