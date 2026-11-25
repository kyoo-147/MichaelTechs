---
title: "World Models and Latent Simulation"
description: "Predicting in latent space is cheaper than predicting pixels, but the learned simulator drifts and its error compounds with horizon."
pubDatetime: "2026-06-13T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - World Models
  - Simulation
categories:
  - Math
---

A controller running at 20 Hz has 50 ms per decision. Inside that window it has to answer a question that is really a batch of questions: for each of $N$ candidate action sequences, what happens over the next $H$ steps. Score them, take the first action, discard the rest, and do it again on the next tick.

That budget is the design problem. Not the architecture, not the loss. The budget, and then the fact that whatever model you fit inside it is wrong in a way that grows.

## The arithmetic of a rollout

The cost of one decision is $N \cdot H \cdot c$, where $c$ is the cost of a single prediction step. Fix the budget at $B$ per tick and the number of plans you can score is

$$N = \frac{B}{H \, c}.$$

Two levers follow. You can raise $B$ or lower $c$. Raising $B$ is linear in money, and money is finite. Lowering $c$ can be structural, which is where the latent argument starts.

If a pixel-space world model has to render a frame and then re-encode it to continue the rollout, every step pays for a decoder and an encoder even though the planner only consumes the hidden state. Predicting in latent space removes both. The published version of that claim is concrete: Mirage, a latent spatial memory for video world models, reports up to $10.57\times$ faster end-to-end generation and a $55\times$ reduction in memory footprint relative to explicit 3D baselines that reconstruct in RGB (Wang et al., [arXiv:2606.09828](https://arxiv.org/abs/2606.09828), 2026). Those are generation numbers, not planning numbers. The direction still holds.

## Where the saving actually comes from

Pixel-space memory is expensive because of the round trip, not because of the number of floats. You render to RGB, store points, then encode back into the diffusion latent before the model can consume them. Two operations per read, plus the loss you take when rich detail gets quantized into pixels. Wang et al. describe the round trip as both costly and lossy, and their fix is to keep the scene cache inside the latent space and warp it there directly.

The second saving is subtler. A planner that searches in the space the model already computes in avoids a projection at every node of the search tree. There is no adapter to train and no separate reward model wedged between the search and the dynamics. One function, evaluated many times, on tensors that are already the right shape.

That is the honest case for latent simulation. It is a compute argument. It was never primarily an accuracy argument, and treating it as one leads people to the wrong conclusions when the rollouts look bad.

## Compression is a constraint, not a byproduct

Latent action models learn abstract actions between consecutive frames of unlabeled video. DiLA attacks the trade-off between how abstract the inferred action is and how faithfully the model can still generate. The paper's claim is that a predictive bottleneck forces disentanglement: the structure pathway has to carry spatial layout because the content pathway is busy holding visual detail (Zhang et al., [arXiv:2605.15725](https://arxiv.org/abs/2605.15725), 2026). The reported outcome is a continuous, semantically structured latent action space without a loss in generation quality, with better action transfer and visual planning.

My reading of why that works: if you give the latent enough room to store the next frame, it stores the next frame, and the action channel becomes decorative. The bottleneck has to hurt. Abstraction that costs nothing is usually not abstraction.

This is also a warning for anyone tuning a latent world model. A latent that reconstructs perfectly is a latent that has stopped compressing. The reconstruction metric will look excellent while the dynamics branch learns nothing useful about the effect of actions, because the cheapest way to predict the future is to copy it.

## The error compounds geometrically

Now the part that does not go away.

Let $\hat z_{t+1} = f_\theta(\hat z_t, a_t)$ with a single-step error bounded by $\epsilon$, and let the learned dynamics be Lipschitz in the state with constant $L$:

$$\| f_\theta(z,a) - f_\theta(z',a) \| \le L \, \| z - z' \|.$$

Then the rollout error obeys

$$\| z_H - \hat z_H \| \le \epsilon \sum_{i=0}^{H-1} L^i = \epsilon \, \frac{L^H - 1}{L - 1}, \qquad L > 1.$$

The bound is linear in per-step accuracy and geometric in horizon. Solving it for the tolerance $\delta$ gives the readable horizon:

$$H^{*} \approx \frac{\log\left(1 + \frac{\delta (L-1)}{\epsilon}\right)}{\log L}.$$

Improving the model by a factor of two adds $\log 2 / \log L$ steps of lookahead. At $L = 1.2$ that is under four steps. So halving per-step error, which is a large engineering project, buys less than four extra steps of planning. Reducing $L$ compounds the other way, because it sits in the exponent.

This is arithmetic, not a measured result, and the constants in any real system will differ. It is still the reason I am skeptical of the standard response to model error, which is to train a better model.

## The contrarian case against chasing fidelity

Almost every world model release is judged on how its rollouts look. For a planner that is the wrong measurement, and the algebra says so cleanly.

A planner takes $a^{*} = \arg\min_a \hat J(a)$. Write the model cost as the true cost plus a shared bias plus an action-dependent term:

$$\hat J(a) = J(a) + b + \eta(a).$$

The bias $b$ cancels in the comparison. Only $\eta$ changes which action wins. Mean rollout error, the number that gets reported, is mostly a statement about $b$. What determines plan quality is the dispersion of $\eta$ across the candidate set.

So a model with visible artifacts and action-independent error can plan well. A photoreal model whose artifacts depend on the action can plan badly. Visual fidelity tells you about the decoder. Plan quality depends on how the error moves when the action moves.

I have not found a paper that measures $\eta$ directly, and that is the gap I care about most. It is also why demo videos are weak evidence. A smooth rollout proves the decoder learned the data manifold. It says almost nothing about whether the ranking of two nearby action sequences survived.

## Planning against a model you know drifts

If the model drifts, the fix that survives the error bound is not a better model. It is a shorter effective horizon.

LAGO is a clean version of that idea. A single predictor both forecasts action-conditioned dynamics and grounds language instructions as sequences of latent subgoals. The planner scores actions with a soft-minimum alignment cost against those subgoals, and the subgoals are repredicted as the agent's state evolves (Barbeau et al., [arXiv:2606.20627](https://arxiv.org/abs/2606.20627), 2026). The reported result is better performance than flat and hierarchical image-goal planners, with the largest gains on long-horizon navigation along curved, hazard-constrained paths, where it more than doubles the success rate of a goal-conditioned behavior cloning policy trained on the same demonstrations.

The mechanism matters more than the score. Repredicting subgoals re-anchors the rollout on an observed state. Ten rollouts of five steps from real observations do not carry the same error as one rollout of fifty steps, because each re-anchor clears accumulated drift. The bound becomes linear in the number of segments instead of geometric in total horizon.

Nano World Models is the tool I would reach for to test that claim in a specific setting rather than accept it in general. It is a compact, reproducible codebase with one interface over generative objectives, model scales, action conditioning, latent observation spaces and long-horizon rollout procedures (Huang et al., [arXiv:2605.23993](https://arxiv.org/abs/2605.23993), 2026). The value is that the rollout knobs are exposed and comparable, so drift becomes something you measure instead of something you watch in a video.

## What I would put on the dashboard

Three numbers, plotted before trusting any planner.

The first is dispersion of the model cost across candidates, not the mean. Score the same state with $N$ action sequences and compare the spread of $\hat J$ against the spread of true returns on a held-out replay. If the model's spread is much narrower than reality's, the planner is choosing between near-ties that are not ties.

The second is the empirical Lipschitz ratio along the states the policy actually visits:

$$L_{\text{emp}} = \max_{(z, z') \in \mathcal{D}} \frac{\| f_\theta(z,a) - f_\theta(z',a) \|}{\| z - z' \|}.$$

Not on random pairs. On pairs from the same trajectory a few steps apart, where drift is born. A global Lipschitz estimate over the whole latent space will be dominated by regions the controller never enters.

The third is a rank-flip horizon. Take a plan that wins at horizon $H$, extend it two ways, once with the model and once with recorded data, and find the smallest extension where the argmin changes. That number, not the loss curve, tells you how far ahead the planner is allowed to look.

```python
def rank_flip_horizon(model, replay, h_max):
    for h in range(1, h_max + 1):
        agrees = model.best_action(h) == replay.best_action(h)
        if not agrees:
            return h   # planner is already past its evidence
    return h_max
```

| Lever | Cost | Effect on the bound | Verdict |
| --- | --- | --- | --- |
| Bigger generator | linear or worse in compute | lowers $\epsilon$ | weak, buys $\log 2 / \log L$ steps |
| Lower Lipschitz constant $L$ | training change | shrinks the exponent | strongest available |
| Re-anchor every $k$ steps | more forward passes | geometric becomes linear | cheap at planning time |
| More candidates at fixed $B$ | reallocates the same budget | none | only after the first two |

The pattern in that table is not the one the field optimizes for. The top row is the expensive lever with the smallest payoff. The bottom two rows cost little and change the shape of the error curve.

A last note on how I read the numbers in these papers. The $10.57\times$ and $55\times$ from Mirage are about a spatial memory design, measured end to end on generation, and they should not be transferred to a planning loop unchanged. The LAGO gain is an environment-specific success rate. Nano is infrastructure, not a result. What holds across all three is the structure of the problem: latent prediction buys you throughput, learned dynamics spend that throughput on drift, and the only durable counter is to shorten the horizon and re-anchor before the error turns the ranking into noise.

## Sources

- Wang et al., 2026. Latent Spatial Memory for Video World Models. "Latent Spatial Memory for Video World Models", [arXiv:2606.09828](https://arxiv.org/abs/2606.09828)
- Barbeau et al., 2026. Latent Goal Prediction from Language for Model-Based Planning. "Latent Goal Prediction from Language for Model-Based Planning", [arXiv:2606.20627](https://arxiv.org/abs/2606.20627)
- Zhang et al., 2026. DiLA: Disentangled Latent Action World Models. "DiLA: Disentangled Latent Action World Models", [arXiv:2605.15725](https://arxiv.org/abs/2605.15725)
- Huang et al., 2026. Nano World Models: A Minimalist Implementation of Future Video Prediction. "Nano World Models: A Minimalist Implementation of Future Video Prediction", [arXiv:2605.23993](https://arxiv.org/abs/2605.23993)
