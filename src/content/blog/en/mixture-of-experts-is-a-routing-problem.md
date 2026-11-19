---
title: "Mixture of Experts Is a Routing Problem"
description: "Sparse activation looks like a compute trick, but top-k selection, capacity limits and balancing losses make it an assignment problem first."
pubDatetime: "2024-06-08T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - MoE
  - Routing
categories:
  - Math
---

The first thing that breaks when a sparse mixture-of-experts model reaches production is a budget mismatch. Capacity planning asks two questions. How much accelerator memory do the weights need, and how much compute does one token cost. For a dense model both answers come from the same number. For a sparse MoE they come from different numbers, and the distance between them is the entire design.

Memory is set by total parameters, because any token might need any expert and every expert has to be resident. Compute per token is set by active parameters, because only $k$ experts run. That gives a sparse model a degree of freedom a dense model does not have. You can add parameters without adding FLOPs. You cannot add them without adding memory, and on a single device memory is the budget that runs out first. Size a cluster from the active parameter count and the checkpoint will not load.

## Two Budgets, Not One

Write it down. With $E$ experts of width $N_e$ and a shared stack of size $N_s$:

$$N_{\text{total}} = E\,N_e + N_s, \qquad N_{\text{active}} = k\,N_e + N_s$$

The ratio $E/k$ is the memory multiplier at fixed compute. It is the number that gets reported as "sparsity", and it is a ratio of two things you pay for separately. Training makes the split worse, not better. Optimizer state exists for every expert, not just the activated ones, so the memory term is a multiple of $N_{\text{total}}$ while the compute term stays at $N_{\text{active}}$. A configuration that looks efficient at inference time looks considerably less efficient in a training job.

These are not variations on one budget. They scale with different variables and they fail in different ways.

| Budget | Scales with | Fixed by | Fails as |
| --- | --- | --- | --- |
| Weight memory | $E\,N_e + N_s$ | expert count $E$ | batch size, device count |
| Compute per token | $k\,N_e + N_s$ | active count $k$ | throughput, latency |
| All-to-all traffic | $k$ routes per token per layer | $k$, $E$, parallelism | interconnect bandwidth |
| Slot waste | $c$ against realized load | capacity factor | effective FLOPs |

The column that matters most is the last one, and it is the one nobody puts on a slide. Memory and compute are what you specified. Slot waste is what you actually got.

## What Top-k Actually Computes

A router is a classifier over experts. It produces $E$ scores per token, turns them into a distribution, then discards most of that distribution:

$$\mathcal{S}(x) = \operatorname{TopK}\big(\operatorname{softmax}(W_r x),\, k\big)$$

$$y = \sum_{i \in \mathcal{S}(x)} g_i(x)\, E_i(x), \qquad g_i(x) = \frac{\exp(s_i)}{\sum_{j \in \mathcal{S}(x)} \exp(s_j)}$$

Three operations sit in that line, and they have three different characters. The softmax is smooth. The top-k is a set-valued map. The renormalization is smooth again, but only over the surviving terms. The model's real decision is the middle one, and it is the only part that is not a continuous function of anything.

This is why "the router learns which expert to use" is a loose description. The router emits preferences. Something else, usually an argmax, turns preferences into an assignment. Those are distinct objects and they are trained differently.

## Where the Gradient Stops

Take the derivative. Membership in $\mathcal{S}(x)$ is piecewise constant in $W_r$, so it contributes nothing:

$$\frac{\partial y}{\partial W_r} = \sum_{i \in \mathcal{S}(x)} \frac{\partial g_i}{\partial W_r}\, E_i(x)$$

There is no $\partial \mathcal{S} / \partial W_r$ term because that derivative is zero almost everywhere and undefined on the boundaries. What the gradient does is sharpen the weighting among experts the router already chose. It does not, by itself, teach the router to choose differently. Reassignment happens as a side effect: example $A$ nudges the weights, example $B$ sits near a decision boundary, and $B$ flips. The flipping is real learning, but it arrives through an indirect path.

Pham et al. take this seriously rather than papering over it. In CompeteSMoE the assignment comes from a competition rule that routes to the experts with the highest neural response, and a separate lightweight router is trained to predict the competition outcome. The decision rule and the predictor are different objects. Under their assumptions the competition rule keeps the convergence rate of an optimal estimator, and the router only has to imitate it. That is a clean way to resolve the non-differentiability: stop asking the router to be the decision.

## The Capacity Factor Is a Queue With Overflow

Every expert gets a fixed number of slots per batch group:

$$C = \left\lceil \frac{T \cdot k}{E} \cdot c \right\rceil$$

with $T$ tokens in the group and $c$ the capacity factor. Total supply is $E \cdot C$. Total demand is $T \cdot k$. At $c = 1$ and a perfectly uniform router the two match exactly, which is a knife edge. Any imbalance means some expert overflows while others idle.

Tokens that overflow are dropped. Their expert branch is skipped and the token continues through the residual stream unchanged. That is the part people underestimate. Dropping makes the computation lossy as well as sparse, and where it loses information depends on which other tokens happened to share the group. At $c < 1$ you cap latency and accept loss. At $c > 1$ you protect quality and pay for padding on the experts that came in under capacity. Neither setting is neutral.

```python
# one MoE layer, shared expert omitted
scores = router(x)                       # (T, E) logits
probs  = scores.softmax(dim=-1)
topk_p, topk_i = probs.topk(k, dim=-1)   # selection: no gradient here
gates  = topk_p / topk_p.sum(-1, keepdim=True)

# capacity: slots are finite, overflow is dropped
keep  = within_capacity(topk_i, capacity=C)
gates = gates * keep                     # dropped tokens contribute zero

# in practice this is a grouped GEMM over dispatched tokens
y = sum(gates[..., j] * experts[topk_i[..., j]](x) for j in range(k))
```

The cross-token coupling has a consequence beyond quality. Hayes shows that routing strategies with cross-batch dependencies are vulnerable to crafted queries: a malicious input can change the model's output on a benign input when both land in the same batch. Their demonstration is a proof of concept in a toy setting, not a deployed exploit, and I would not read more into it than that. The structural point stands on its own. Capacity converts a per-token function into a batch function, and a batch function has a batch-shaped attack surface.

## Load Balancing Is a Tax, Not a Constraint

Left alone, routing collapses. The experts that receive more tokens early receive better gradients, become better, and get selected more often. Their peers starve. The usual remedy is an auxiliary loss of the product form:

$$\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{E} f_i \, p_i$$

where $f_i$ is the fraction of tokens dispatched to expert $i$ and $p_i$ is its mean router probability. Minimizing that sum pushes the two distributions toward each other and both toward uniform.

Notice what is being optimized. You care about likelihood. Uniformity is not likelihood. It is a proxy chosen because a collapsed router wastes the parameter budget you already paid for, and the loss is the cheapest available way to prevent that. So $\alpha$ works as an exchange rate between specialization and utilization rather than as a hyperparameter in the ordinary sense, and no value of it buys both.

There is a second wrinkle. The quantity that fills capacity is $f_i$, the discrete count. The quantity the gradient can touch is $p_i$, the continuous mean probability. A uniform $p$ does not imply uniform $f$ at small batch sizes, which is exactly when variance in the counts is largest. The proxy is weakest where you need it most.

## The Contrarian Case: The Router Is Probably Not Specializing the Way You Think

The standard story says the router learns to send each token to the experts that specialize in it, and the whole system improves because specialization improves. I think that story is backwards about which part does the work.

Consider what HyperRouter reports. Do et al. start from the finding that fixing the routers, including keeping them random, can achieve competitive performance, because a fixed assignment avoids the collapse dynamic. Their critique of freezing is that a random policy is sub-optimal and that frozen routers still cost full training and evaluation resources. Both of those are cost arguments. Neither is a claim that a learned router finds materially better specialization than a fixed partition.

Sit with that. If a random fixed assignment is competitive, then performance is coming from somewhere other than semantic matching between token and expert. The candidate is stability: each expert sees a consistent subset of inputs and adapts its weights to that subset. The partition can be arbitrary and the model still specializes, because the experts do the specializing and the router only has to keep the partition from churning.

The other two results fit that reading. Pham et al. frame the central failure of sparse training as representation collapse, which is a story about what experts learn, not about assignment quality. Nguyen et al. show that the router's parameterization changes the statistical behavior of the entire model: with a cosine router, estimation rates for both experts and model parameters can degrade to $\mathcal{O}(1/\log^{\tau} n)$, and the degradation is repaired by adding noise to the $\ell_2$ norms. A component that only selected experts would not be able to slow the estimation rate of the experts themselves. The router is inside the model's statistical structure, not bolted onto it.

My reading, and it is a reading rather than a measured result, is that in most deployed systems the router behaves closer to a learned partition keyed on surface properties of the token than to a topic classifier. The testable version of that claim is a hypothesis: if it holds, ablating experts by measured utilization should cost far less than ablating them by nominal parameter share. Lu et al. report something in that direction, having pruned and skipped experts in MoE language models while maintaining satisfactory performance on their benchmark suite. That is consistent with the hypothesis. It is not proof of it, and their evaluation is task-specific by construction.

## What Follows for Design

Decide which budget you are optimizing before you pick $E$ and $k$. A memory-bound deployment wants few, wide experts. A throughput-bound deployment wants many experts with small $k$, and it should expect the all-to-all to become the bottleneck before the arithmetic does. Choosing $E$ and $k$ first and discovering the binding constraint afterwards is the common order and the wrong one.

Treat the capacity factor as a latency target rather than a default. Measure the drop rate on your real batch distribution, not on a uniform one. The distribution of tokens across experts at serving time is not the distribution you saw in training, and the difference lands entirely in the overflow term.

Do not tune $\alpha$ in isolation. The balance you want is the balance at the batch sizes you serve, and the auxiliary loss gets weaker as groups get smaller. If you serve single requests, you are running a different balancing regime than the one you validated.

Expect the router to be the least interpretable component and the most consequential one. The router weights will not tell you much. Watch per-expert utilization over time, the overflow rate, and the variance of both across batch groups. Those three numbers tell you whether the model you paid for is the model that is running.

And if you serve multiple tenants through shared batches, know that the batch is now part of your trust boundary. That is a consequence of the queue, not of the model's intent.

## Sources

Pham, Q. et al. *CompeteSMoE: Effective Training of Sparse Mixture of Experts via Competition.* [arXiv:2402.02526](https://arxiv.org/abs/2402.02526), 2024. https://arxiv.org/abs/2402.02526

Do, G. et al. *HyperRouter: Towards Efficient Training and Inference of Sparse Mixture of Experts.* [arXiv:2312.07035](https://arxiv.org/abs/2312.07035), 2023. https://arxiv.org/abs/2312.07035

Nguyen, H. et al. *Statistical Advantages of Perturbing Cosine Router in Mixture of Experts.* [arXiv:2405.14131](https://arxiv.org/abs/2405.14131), 2024. https://arxiv.org/abs/2405.14131

Lu, X. et al. *Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models.* [arXiv:2402.14800](https://arxiv.org/abs/2402.14800), 2024. https://arxiv.org/abs/2402.14800

Hayes, J. *Buffer Overflow in Mixture of Experts.* [arXiv:2402.05526](https://arxiv.org/abs/2402.05526), 2024. https://arxiv.org/abs/2402.05526
