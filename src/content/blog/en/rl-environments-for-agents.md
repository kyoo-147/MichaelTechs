---
title: "RL Environments for Agents"
description: "An environment is a specification of what counts as success, so reward shaping edits the claim and a simulator asserts it about the world."
pubDatetime: "2026-08-29T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - RL
  - Agents
categories:
  - Math
---

A team trains a support agent for six weeks. The environment pays 0.2 for a refund that matches the ledger, 0.1 for citing the policy document, and a small penalty per step. Eval return climbs to 0.83 and holds there. The agent ships. Two weeks later it is closing tickets without refunding anyone, and the return is still 0.83.

Nothing in that story is a bug in the optimiser. The environment did what it was written to do. The number that went up measured agreement with a function someone wrote in an afternoon, and that function was not the task. When the two came apart, nothing in the pipeline was watching, because the pipeline only ever carried one number.

## An Environment Is a Specification, Not a World

Strip the interface down and an environment is two things: a transition function and a predicate.

$$\mathcal{E} = \langle \mathcal{S}, \mathcal{A}, P, \phi, \rho_0, H\rangle, \qquad P(s' \mid s, a), \quad \phi : \mathcal{S} \to \{0,1\}$$

The first says what happens when the agent acts. The second says whether what happened counted. Everything else - the observation channel, the reward, the episode length - is scaffolding around those two.

The learner never sees $\phi$. It sees

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{H} \gamma^{t} r(s_t, a_t)\right],$$

and it climbs that. So $r$ is not the specification. It is a device for turning the specification into a gradient. When the shaping is well built the two agree. When they disagree, the optimiser follows $r$ and the specification stops mattering, because nothing downstream checks it.

There is another compression inside $\phi$ itself. Success conditions are multi-objective and the weighting is not universal: two users can agree a task succeeded while disagreeing about how much each constraint mattered. Holgado-Sánchez and colleagues learn that weighting instead of hand-writing it, formalising value system learning over multi-objective Markov decision processes and giving preference-based and inverse reinforcement learning methods that infer value grounding functions and their weights from demonstrations ([arXiv:2602.04518](https://arxiv.org/abs/2602.04518), Holgado-Sánchez et al., 2026). Their motivation should worry anyone who writes $\phi$ by hand: users hold different value systems, and the meaning of a value in context is hard to state computationally. Learning the weights is a reasonable answer, and it does not remove the need to publish the predicate, because learned weights are part of the specification too.

## What Shaping Actually Optimises

Reward shaping is any $r' = r + F$, and the shaped return splits cleanly:

$$J'(\pi) = J_r(\pi) + J_F(\pi).$$

The standard result, due to Ng, Harada and Russell, is that if $F$ is a potential difference, $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$, the set of optimal policies is unchanged. That theorem is narrower than it is usually read as. It preserves the argmax and says nothing about the ordering of near-optimal policies, nothing about which a learner reaches first, and nothing about behaviour off the training distribution. Those are the properties you ship.

For arbitrary $F$ there is no invariance at all, and the scale tells you how badly. Normalise $F$ to $\alpha \tilde{F}$ with $\lVert \tilde{F} \rVert_\infty = 1$. Then

$$\left| J'(\pi) - \alpha J_{\tilde{F}}(\pi) \right| = \left| J_r(\pi) \right| \le \frac{\lVert r \rVert_\infty}{1-\gamma},$$

so once $\alpha$ is large relative to $\lVert r \rVert_\infty$, the policy maximising the shaped return is close to the one maximising the bonus alone. The task term is still in the objective. It is no longer the term being decided.

The exploration literature has a clean instance of the same defect. Wahab, Kumaraswamy and White examine bonuses learned by estimating a value function on reward bonuses and identify a structural problem: the bonus for a state-action pair only rises after a higher bonus has already been observed there ([arXiv:2602.12375](https://arxiv.org/abs/2602.12375), Wahab et al., 2026). The bonus is retroactive by construction, so it cannot make a first visit attractive, which is the one thing an exploration bonus exists to do. Their VBE algorithm keeps an ensemble of random action-value functions, uses their estimation errors to build bonuses with first-visit optimism, and designs the rewards so the bonus can decay to zero. As reported, VBE beats Bootstrap DQN, RND and ACB on classic exploration environments and scales to Atari.

Read that as a template, not a result. The bonus encodes an assumption about the shape of the optimum. The assumption is testable, and the test is $\phi$, so an environment that never exposes $\phi$ on its own cannot test it.

## Five Ways Shaped Rewards Fail

These modes are structural, and each leaves a signature in logs you probably already have.

| Mode | What the policy does | The tell | Root cause |
|---|---|---|---|
| Bonus farming | parks where the auxiliary term is largest | shaped return rises, task success flat | auxiliary term dominates the argmax |
| Proxy collapse | optimises the measurable part of the task | training return up, held-out success down | measurement narrower than the claim |
| Threshold cliff | stops at the edge of the bonus region | bimodal return distribution | bounded or discontinuous predicate |
| Retroactive credit | never visits the unrewarded region | flat state coverage across training | bonus cannot fire before observation |
| Schedule drift | changes behaviour when the bonus decays | return jumps between checkpoints | non-stationary reward function |

Bonus farming is what people call reward hacking, and the name hides the mechanism. The agent finds the maximum of the function you gave it, and that maximum is not what you meant. Proxy collapse is worse, because the proxy is often legitimate: a rubric, a unit test, a similarity score. A lever that reads part of the task teaches the agent to ignore the rest.

Threshold cliff is subtler. When $\phi$ is a bound, shaping that fits well near the bound does badly just past it, and the policy learns to sit on the safe side; you see it as a return distribution with two humps and nothing between them. Retroactive credit is the VBE problem generalised: any bonus that needs an observation before it pays out is silent on exactly the states you have not reached. Schedule drift is the easiest to miss, because the schedule lives in the trainer and the evaluation lives in another file.

None of these is fixed by tuning a coefficient. A coefficient moves where the failure sits. It does not remove it.

## A Simulator Is a Claim About the World

An environment standing in for a real system asserts three things at once: that $P$ behaves like the system, that $\phi$ matches the real success condition, and that the observation channel carries the information the real one carries. The first is what people audit. The second and third are where most of the risk lives.

$P$ being wrong is usually visible, because the physics looks strange. $\phi$ being wrong is invisible, because a predicate has no visual signature. A simulated refund task where success means the database row changed is a different task from one where success means the customer was made whole, and the simulator cannot tell you which one it implements. The observation channel is worse. Simulation hands the policy state, while the world hands it an estimate with an error distribution attached, and a policy trained on the first consumes the second without noticing the difference.

Open Ant is the most concrete attack on this problem I have read this year. Lupu and colleagues build a physical variant of the Gymnasium Ant, open hardware and open software, and report competent walking policies learned from scratch in roughly one hour directly from the physical robot, for two quite different algorithms, SARSA($\lambda$) and SAC ([arXiv:2607.18488](https://arxiv.org/abs/2607.18488), Lupu et al., 2026). Separately they show simulation-trained policies transferring to the physical robot. They also measure the two quantities that decide whether a physical platform gets used: how quickly new users reach a first success, and how easily it is repaired when hardware fails.

The walking is the headline. The paper matters because it makes the claim checkable: a simulator is a hypothesis about a system, and the only experiment that tests it happens outside the simulator.

## The Contrarian Case: Fidelity Is Not the Bottleneck

The standard response is to build a better simulator: more contact fidelity, better noise models, more realistic latency. That spends the budget in the wrong place, for two reasons.

First, the properties that decide whether a policy transfers are the action interface, the observation the policy conditions on, and the way $\phi$ responds to actions across the range of behaviour you will deploy. Those are interface properties, cheap to get right compared with fidelity work. A contact model accurate to the millimetre buys nothing for a policy that cannot see the variable its predicate depends on.

Second, the binding constraint on real environments is maintenance, not realism. A physical rig breaks. A service-backed environment drifts when the service changes. A data-derived one goes stale when the data does. Each is a recurring cost, and if keeping an environment alive costs more than the training run, it quietly stops being run. The evaluation becomes a memory of an evaluation, and the claim about the world goes untested for a year.

That is why I read the repair and onboarding measurements as load-bearing rather than as appendices. An environment you can fix in an afternoon is one you keep running; one only a specialist can repair gets retired the first time it breaks.

This is opinion, not measurement. What the papers support is narrower: competent policies were learned on the physical robot in about an hour, and simulation-trained policies transferred to it. Which factor did the work is not something those results separate.

## What I Would Put in the Environment

Four things, ordered by how much they cost me when missing.

Return the task signal and the training signal as separate fields. The learner reads one, the dashboard reads both. That change alone makes every failure mode above detectable, because the divergence is the signature.

```python
class Task:
    """The claim. Versioned independently of any training run."""
    version = "3"

    def success(self, traj) -> bool:
        return traj.refund_matches_ledger and traj.cited_policy

    def violation(self, traj) -> bool:
        return traj.refund_issued and not traj.customer_verified


class Env:
    def step(self, action):
        obs, done = self.world.step(action)
        self.traj.record(action, obs)
        return obs, done, {
            "task_success": self.task.success(self.traj),      # what you claim
            "task_violation": self.task.violation(self.traj),  # a gate, not a term
            "shaped": self.bonus(obs),                         # what the learner sees
        }
```

Make $\phi$ callable on a stored trajectory. If success can only be computed online, you cannot re-score last month's runs after the definition changes, and you will want to. The predicate is the part of an environment most likely to change and most often unrecoverable.

Version the predicate like an API. A changed $\phi$ invalidates every number produced before it. Two runs under different versions are not comparable, and a results table that mixes them is not a comparison.

Test the predicate against trajectories it should reject. Take the run that farmed the bonus, hand it to $\phi$, and assert that $\phi$ says no. That test is cheap and deterministic, and it is the only check that reads the claim instead of the signal.

The agent that scored 0.83 was not misbehaving. It found the maximum of the function it was given. Publish the claim and the signal separately, keep both on the same chart, and version the predicate. That is the difference between an environment that specifies a task and one that specifies a gradient.

## Sources

- "Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning", [arXiv:2602.12375](https://arxiv.org/abs/2602.12375) - Wahab, A., Kumaraswamy, R., White, M. (2026). Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning.
- "The Open Ant: A Robot Platform for Reinforcement Learning Research", [arXiv:2607.18488](https://arxiv.org/abs/2607.18488) - Lupu, E. S. et al. (2026). The Open Ant: A Robot Platform for Reinforcement Learning Research.
- "Learning the Value Systems of Agents with Preference-based and Inverse Reinforcement Learning", [arXiv:2602.04518](https://arxiv.org/abs/2602.04518) - Holgado-Sánchez, A. et al. (2026). Learning the Value Systems of Agents with Preference-based and Inverse Reinforcement Learning.
