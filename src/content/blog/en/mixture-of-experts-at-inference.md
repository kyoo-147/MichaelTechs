---
title: "Mixture of Experts at Inference"
description: "Sparse activation cuts FLOPs per token, but residency, all-to-all traffic and routing skew decide what a sparse model actually costs to run."
pubDatetime: "2025-05-24T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - MoE
  - Inference
categories:
  - Math
---

A capacity plan for a sparse model usually starts with the active parameter count and ends with a surprise. You take the number from the launch post, multiply it by bytes per parameter, divide by the memory of one accelerator, and buy that many devices. Then the checkpoint does not fit, the interconnect saturates before the arithmetic units do, and the throughput you projected from a dense model of the same active size never arrives.

None of that is a mistake in the papers. It is what happens when a training-time efficiency claim gets read as a deployment cost claim. Sparsity buys something specific. At inference it buys that thing in one regime only, and the rest of the system sends you the bill for the difference.

## The Saving Is Real, and It Misses the Bottleneck

Autoregressive decode is a memory-bandwidth problem before it is a compute problem. To emit one token you have to read the weights that take part in it. For a dense model that is $N$ parameters. For a top-$k$ sparse model it is $k$ experts plus the shared stack:

$$N_{\text{active}} = k\,N_e + N_s$$

What one decode step reads is not $k$ experts, though. It is the union of experts touched by every sequence in the batch. Call that union $U(B)$, with $U(B) \le E$. Weight bytes per emitted token are then

$$\frac{U(B)\,N_e\,\beta_w}{B}$$

where $\beta_w$ is bytes per parameter. Two consequences follow, and they pull in opposite directions. Peak residency is governed by $E$, because over enough tokens $U(B)$ reaches every expert. The per-token weight bill does fall with $B$, but only as fast as $U$ grows relative to $B$, and when routing is diverse $U$ saturates near $E$ quickly.

FLOPs per token stay fixed at $k N_e + N_s$. That is the quantity sparsity reduces. It dominates cost only once the weights are amortised, which is a statement about your batch, not about the architecture.

| Regime | Binding constraint | What sparsity buys |
| --- | --- | --- |
| Small batch, all resident | Weight bytes per step | Lower residency if the expert set fits |
| Offloaded experts | I/O per layer against compute | Bubbles unless the schedule hides the load |
| Large batch, compute bound | FLOPs and collective volume | The real FLOP reduction, minus skew |

That table is most of the argument in three lines. A sparse model is cheap where the compute term is the bottleneck and the weights are already paid for. Everywhere else it is a memory and network story with a compute discount attached.

## Memory Does Not Get Sparse

Sparsity is a conditional computation claim. It is not a conditional storage claim. Any token may select any expert, so at serving time every expert has to be reachable, resident in HBM or one predictable step away on a host.

$$M_{\text{weights}} = (E\,N_e + N_s)\,\beta_w, \qquad M_{\text{kv}} = 2\,L\,H\,d_h\,T\,B\,\beta_{kv}$$

The first term does not move when you reduce $k$. Not by one byte. Take $k$ from 8 to 2 and you cut FLOPs by three quarters while the residency line stays exactly where it was. The second term is the one that grows across a long generation, and it competes with the first for the same pool. On a single device this is the arithmetic that fails first, and no routing improvement touches it.

Lu et al. take the direct route and delete experts after training. Their post-training pruning and skipping methods reduce model size and raise inference speed while holding task performance across a range of tasks, without special hardware. That is the honest shape of the fix. When residency is the binding constraint, you have to stop keeping the experts.

Offloading is the other valve, and it turns a capacity problem into a scheduling problem. Fang et al. describe the regime plainly in Klotski: for MoE the compute for each expert is often less than the I/O to fetch it, so a naive pipeline spends most of its time waiting on memory. Their fix is scheduling, ordering work by heterogeneous compute and I/O needs so the load of the next layer overlaps the compute of the current one. Worth being precise about what offloading means. You did not make the model fit. You made not fitting cheaper.

## The All-to-All Is a Second Bill

Under expert parallelism the experts are sharded across devices and the tokens are not. Each MoE layer moves token activations to wherever the selected expert lives, then moves the results back. That is two collectives per layer in the usual implementation, and both sit on the critical path between layers.

$$V_{\text{layer}} \approx 2\,k\,B\,T\,d\,\beta_a$$

The factor $k$ is the sting. A dense layer moves one activation per token per layer. A sparse layer moves $k$ copies out and $k$ partial results back, so communication volume scales with the very quantity that was supposed to make the model cheap. Bandwidth is only half of it. An all-to-all is a barrier, so the layer cannot finish until the slowest participating link finishes, and latency tracks the tail rather than the mean.

ExFlow goes after this with inter-layer expert affinity. If the tokens that pick expert $i$ in one layer tend to pick a predictable set in the next, then experts can be placed so that routing stays local, and two all-to-alls collapse into one. Yao et al. report up to 67 percent lower cross-GPU routing latency and up to 2.2x throughput across 8 to 64 experts, with no fine-tuning and no accuracy loss. Those are their measurements on their configurations. The transferable idea is that communication cost is a function of expert placement, and placement is a decision you are allowed to make.

## Batching Meets Routing Skew

Dense batching amortises weight reads over more tokens. Sparse batching does that too, then hands you a second cost, because a batch does not spread evenly over experts.

Layer latency under expert parallelism is set by the busiest expert:

$$\tau_{\text{layer}} \approx \max_{e} L_e, \qquad \rho = \frac{\frac{1}{E}\sum_e L_e}{\max_e L_e}$$

Read $\rho$ as the fraction of the hardware you actually used. If one expert takes three times the mean load, you paid for $E$ experts and got the throughput of roughly a third of them, because everything else waits at the barrier.

He et al. name this the straggler effect and measure its shape. Underloaded experts finish early and idle, and the most burdened expert sets the step time. Their capacity-aware token drop enforces a per-expert cap, and they report a 30 percent speedup for 0.9 percent degradation on OLMoE, plus 1.85x speedup and a 0.2 percent average improvement from expanded drop on Mixtral-8x7B-Instruct. The numbers are theirs. The mechanism is the part that matters: you can recover latency by capping load per expert, and you pay with tokens that never reach their first choice.

There is a second-order effect I find more interesting. The Klotski authors note that adding batches can activate more distinct experts, which lengthens loading and adds bubbles rather than removing them. So the reflex that more batching means better amortisation can relocate a bottleneck instead of shrinking it. My reading is that for offloaded or heavily sharded sparse models, batch size is not a free knob. It changes the working set.

## The Standard Framing Gets Routing Backwards

The usual claim is that a sparse model gives you the capacity of a large model at something near the cost of a small one. The missing clause is that this holds only when routing is balanced, the batch is large enough to be compute bound, every expert is resident, and the interconnect is fast. Four conditions, and in production serving they rarely hold at once.

The stronger objection is about where routing comes from. Routing is trained. The token-to-expert distribution is a property of the router and of the input distribution, both fixed before your service exists. At inference you cannot retrain it, so you have three moves and none of them is free. You reschedule, which is what Klotski and ExFlow do: same function, better order, and the cost moves into engineering complexity. You add capacity and accept idle experts, which is waste you budgeted for. Or you drop tokens, which is what capacity-aware inference does, and which changes the function the model computes. That last move is a correctness change wearing an optimisation label.

Calling routing an inference problem is therefore incomplete. Routing is a training artefact whose serving symptom is skew. Every pure-serving fix relocates the cost: into placement, into idle silicon, or into dropped tokens. Anyone quoting a sparse throughput number should say which of the three they spent to get it.

I would add a second correction. Gating is a statistical object, not a scheduler. Nguyen et al. analyse softmax gating and show the estimation difficulty depends on the expert class. A two-layer feedforward expert needs polynomially many samples under their strong identifiability condition, while a linear expert needs exponentially many because of parameter interaction. That is a claim about what the gate can learn. It is also why I am sceptical of describing the router as a load balancer that happens to be accurate. It is an estimator first, balance is a constraint imposed on it afterwards, and constraints of that kind cost accuracy somewhere.

## Cheaper Per Token, More Expensive Per Deployment

Here the two claims stop contradicting each other. Cost per token is a fixed charge spread over volume, plus a marginal charge:

$$\frac{\text{cost}}{\text{token}} = \frac{C_{\text{fixed}}}{U} + c_{\text{marg}}$$

Sparsity pushes the two terms in opposite directions. $C_{\text{fixed}}$ rises, because residency is set by $E$ and you buy interconnect to carry $k$ routes. $c_{\text{marg}}$ falls, because each token computes $k N_e$ instead of $N$. Comparing a sparse configuration $s$ against a dense one $d$, the sparse deployment wins only above

$$U^{*} = \frac{C_s - C_d}{c_d - c_s}$$

tokens served over the life of the deployment. Below $U^{*}$ the dense model is the cheaper thing to run even though it costs more per token. This is the ordinary fixed-versus-marginal structure of any capital purchase, and it explains why a per-token figure in a launch post says almost nothing without a utilisation figure beside it.

A short calculation shows the shape without pretending to benchmark anything.

```python
def break_even(c_fixed_sparse, c_fixed_dense, c_marg_dense, c_marg_sparse):
    """Tokens needed before the sparse deployment is cheaper overall."""
    if c_marg_sparse >= c_marg_dense:
        return float("inf")  # no volume rescues it
    return (c_fixed_sparse - c_fixed_dense) / (c_marg_dense - c_marg_sparse)

def effective_rate(peak_tokens_per_sec, balance):
    """Straggler loss: the busiest expert sets the step time."""
    return peak_tokens_per_sec * balance
```

Both functions are accounting, not measurement. Feed them your own prices and your own measured balance. If $U^{*}$ lands above the traffic you can plausibly route to that model before you replace it, the sparse deployment is the worse buy on cost alone, whatever the per-token number says.

Surveys of this space, Liu et al. among them, tend to organise the work by stack level: model, then system, then hardware. That taxonomy is correct and it is also a warning. A win at one level is usually a bill at another, which is why compression papers and scheduling papers can both be right while disagreeing about what to optimise.

## What I Would Size First

Three quantities, in this order. Resident bytes: every expert plus the KV cache at the target context length and batch. Collective volume per layer: $2kBTd$ times bytes per element, set against real interconnect bandwidth and the tail latency you can tolerate. Then load balance on realistic prompts, because the routing distribution on a benchmark suite is flatter than the routing distribution on production traffic.

If residency fails, the answer is pruning, quantisation or offload, and a better router will not save you. If collective volume fails, the answer is expert placement and sequence parallelism. If balance fails, you are choosing between wasted slots and dropped tokens, and that is a product decision about how much accuracy you will trade for a step time.

A sparse model can be cheaper per token and more expensive per deployment because those are two budgets with two different scaling variables. Per-token cost follows $k$. Deployment cost follows $E$ and the network. Optimise the first and the second grows. No configuration escapes the trade. Some configurations just place it where your traffic pattern can pay for it.

## Sources

- Huy Nguyen, Nhat Ho, Alessandro Rinaldo. "Convergence Rates for Softmax Gating Mixture of Experts." [arXiv:2503.03213](https://arxiv.org/abs/2503.03213)v1, 2025. http://arxiv.org/abs/2503.03213v1
- Shwai He et al. "Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts." [arXiv:2503.05066](https://arxiv.org/abs/2503.05066)v5, 2025. http://arxiv.org/abs/2503.05066v5
- Zhiyuan Fang et al. "Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline." [arXiv:2502.06888](https://arxiv.org/abs/2502.06888)v1, 2025. http://arxiv.org/abs/2502.06888v1
- Jiacheng Liu et al. "A Survey on Inference Optimization Techniques for Mixture of Experts Models." [arXiv:2412.14219](https://arxiv.org/abs/2412.14219)v2, 2024. http://arxiv.org/abs/2412.14219v2
- Jinghan Yao et al. "Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference." [arXiv:2401.08383](https://arxiv.org/abs/2401.08383)v2, 2024. http://arxiv.org/abs/2401.08383v2
- Xudong Lu et al. "Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models." [arXiv:2402.14800](https://arxiv.org/abs/2402.14800)v2, 2024. http://arxiv.org/abs/2402.14800v2
