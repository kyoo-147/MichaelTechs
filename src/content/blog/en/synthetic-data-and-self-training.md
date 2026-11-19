---
title: "Synthetic Data and the Mathematics of Self-Training"
description: "A model trained on its own outputs converges to whatever your filter, temperature and verifier encode, not to the distribution you started from."
pubDatetime: "2025-07-12T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Synthetic Data
  - Self-Training
categories:
  - Math
---

Consider a 7B model on a task where human labels are expensive. You sample 200k completions from the current checkpoint, score them with a reward model, keep the top half, fine-tune, and repeat. Round two improves. Round three is flat. Round four produces answers that are shorter, more confident and more uniform in structure than round one, and a manual read finds four argument shapes recycled across questions that used to get eight.

Nothing in that loop is a bug. Every step is a defensible engineering choice, and the composition still walks the model somewhere you did not choose. The question worth doing mathematics about is not whether self-training works, but which distribution the loop converges to and whether you would have picked it.

## The loop is an operator

Let $p_\star$ be the data distribution and $q_t$ the model after round $t$. A round samples from $q_t$, transforms the samples somehow, and refits. Write the round as

$$q_{t+1} \;=\; \Pi_\theta\left(\lambda\, p_\star \;+\; (1-\lambda)\,\mathcal{F}(q_t)\right),$$

with three free pieces. $\mathcal{F}$ is what you do to the samples: nothing, filter, temper, gate. $\lambda \in [0,1]$ is the real-data fraction, so $\lambda = 0$ is the replace regime and $\lambda > 0$ is accumulation. $\Pi_\theta$ is the projection onto what the model class can represent from the samples it received.

Take the ideal case first: $\mathcal{F}$ is the identity, samples are unlimited, and $\Pi$ is a linear projection so $\Pi^2 = \Pi$. The fixed point satisfies $q = \Pi(\lambda p_\star + (1-\lambda) q)$. Applying $\Pi$ to both sides and using idempotence gives $\Pi q = q$, so the estimate is already a fixed point of the projection. Substituting that back gives $\lambda q = \lambda \Pi p_\star$, hence $q_\infty = \Pi p_\star$ for any $\lambda > 0$. Self-training with no filtering and a constant real-data fraction converges to the best representable approximation of the truth. Mixing, not collapsing. Fu and coauthors prove a version of this for self-consuming loops, where a constant proportion of real data is enough for convergence.

So the interesting mathematics is not in the fixed point of the clean map. It lives in the three places where the clean map is wrong: when $\mathcal{F}$ is not the identity, when $\Pi$ is not a projection, and when the samples are finite. Each one moves the fixed point or destroys it.

## Filtering is a tilt, not a cleanup

Rejection feels like error removal. Mathematically it is a reweighting. Keep a sample $x$ with weight $w(x)$ and refit to what you kept, and the training target becomes

$$\tilde q(x) \;=\; \frac{q(x)\,w(x)}{Z}, \qquad Z = \mathbb{E}_{q}\left[w\right].$$

A fixed point of the pure self-loop needs $q = \tilde q$, which needs $w$ constant on the support of $q$. A hard filter that rejects a set of positive mass cannot satisfy that. Mass leaves the rejected region every round, and the limit lives on the set the filter accepts. You do not recover $p_\star$ with its mistakes deleted, because the operation that deletes mistakes also deletes whatever else the filter dislikes, including parts of the true distribution that score low.

This is the precise sense in which the usual warning is right. The fixed point of a filtered loop is not the true distribution. It is a tilted distribution, and the tilt is the filter written in distribution space.

## Temperature fixes the entropy, not the shape

Tempering is the same story with a cleaner parameter. Sampling at temperature $T$ gives $q_T \propto q^{1/T}$. Write $q = e^{-E}$ so the tempered law is Gibbs with energy $E$ at temperature $T$. Its entropy obeys

$$\frac{dH}{dT} \;=\; \frac{\operatorname{Var}_T[E]}{T^2} \;\ge\; 0,$$

which is exact, not an approximation. Sampling below $T = 1$ lowers the entropy of the pool, and refitting carries that lower entropy into the next model. Do it every round and $H$ decreases strictly each round. The limit is a point mass on the mode. Above $T = 1$ the argument reverses and the limit drifts toward uniform.

Both are wrong, and wrong in opposite directions. Only $T = 1$ leaves the target alone. Temperature is not a dial you can leave off centre and correct for later; it selects a different fixed point.

| What the round does | $\mathcal{F}(q_t)$ | Where the loop settles | Early sign |
|---|---|---|---|
| Nothing | $q_t$ | $\Pi p_\star$ under a true projection, otherwise drifts | no visible drift |
| Mix a constant real fraction | $(1-\lambda)q_t + \lambda p_\star$ | $p_\star$, with bounded error | diversity stable |
| Sharpen, $T<1$ | $q_t^{1/T}/Z$ | concentration on the mode | entropy down each round |
| Flatten, $T>1$ | $q_t^{1/T}/Z$ | drift toward uniform | entropy up, accuracy down |
| Reject below a threshold | $q_t \mathbb{1}[s \ge \tau]/Z$ | the accepted set, minus its tail | tail coverage down |
| Hard verifier gate | $q_t a(x)/Z$ | the verifier's preferred set | verifier score up, held-out flat |

## Finite samples move the fixed point

Even with $\mathcal{F}$ the identity, real training is not a projection. It is a noisy estimate. Write the parameter update as a walk,

$$\hat\theta_{t+1} = \hat\theta_t + b_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}\left(0, \sigma^2/n_t\right),$$

with per-round bias $b_t$ and $n_t$ samples. Without a contraction term the accumulated variance after $T$ rounds is $\sum_{t \le T} \sigma^2/n_t$. Bounding it as $T \to \infty$ requires $\sum_t n_t^{-1} < \infty$, which needs $n_t$ to grow faster than any linear rate, roughly like $t^{1+\epsilon}$. Xu, He and Cheng derive this shape and show the required growth accelerates when the estimator carries bias.

Real data supplies the missing contraction. With constant $\lambda > 0$ and constant batch size $n$, the stationary variance is bounded,

$$\operatorname{Var}_\infty \;\lesssim\; \frac{\sigma^2}{n\left(1 - (1-\lambda)^2\right)},$$

which is finite for any $\lambda > 0$ and blows up as $\lambda \to 0$. This two-line recursion is not a theorem about transformers and I am not claiming it is. It does reproduce the qualitative split the empirical papers report: replacing real data with synthetic data degrades, accumulating both does not, and Gerstgrasser and coauthors prove a finite upper bound on test error for the accumulating case.

## The contrarian case: collapse is a label doing two jobs

The standard framing says a model trained on its own output amplifies its own errors, so the loop is dangerous and should be avoided or heavily diluted. That framing is incomplete in a specific way. It fuses two failures with different causes and different repairs.

The first is distributional narrowing. The support shrinks, entropy falls, the model becomes confident and repetitive. Its fix is an explicit diversity term or a larger real-data fraction, and it is measurable before any downstream evaluation.

The second is capability regression. Held-out accuracy on real data gets worse. Its cause is usually noise accumulation or a biased estimator, and its fix is more samples or a better estimator, not more diversity.

A loop can show the first without the second, and often does. Hu, Rostami and Thomason study multimodal loops and report something awkward for the standard framing: vision-language alignment improved across rounds even as caption variance rose. Growing the decoding budget, adding distinct models to the loop, and relabeling with a frozen model all reduced the damage. A loop that improves one measurable property while widening its output variance is not collapsing. It is doing what the objective asked.

Zucchet and coauthors add a third mechanism unrelated to synthetic data. In their factual recall setup, hallucinations appear at the same time as the knowledge rather than after it, and fine-tuning new facts into a model quickly corrupts the parametric memories already there. Some of what gets blamed on the loop is the ordinary cost of fine-tuning anything.

Schaeffer, Kazdan, Arulandu and Koyejo make the terminology problem explicit: the literature carries eight distinct definitions of model collapse, several prominent predictions rest on assumptions that do not match how models are trained, and some of the scariest scenarios are avoidable with ordinary care. My own view is narrower and points somewhere they do not go. The dangerous loop is not the hypothetical pretraining recursion. It is the small fine-tuning loop teams actually run, on a task with a few thousand real examples, where $\lambda$ drifts toward zero because synthetic data is free and annotation is not.

## Measuring diversity collapse honestly

Vocabulary statistics are cheap: distinct-1 and distinct-2 over a fixed prompt set, plus type-token ratio at fixed sample size. They catch repetition early.

They miss the case that matters. Zhu and coauthors analyse synthetic text and find both distributional shift and over-concentration of $n$-gram features, which is the sharper version of the same signal. It is not that the vocabulary is small. It is that particular $n$-grams carry far more mass than the real corpus gives them.

The measurement I trust more is tail coverage. Fix a held-out real set and a threshold $\epsilon$, then measure what fraction of real examples receive model probability above $\epsilon$. A model can hold its top-1 accuracy while losing half its tail, and the tail is what generalises.

Two guardrails when reading any of these numbers. An average can hide a bimodal split where most prompts improved and a minority fell into a loop; report the distribution, not the mean. And non-collapse is a weak guarantee. Kanabar and Gastpar show that even when collapse is deliberately averted by adding fresh real samples, the ratio between the self-consuming minimax loss and an oracle-assisted loss can grow without bound as batch size grows. Bounded is not equal. A loop can be perfectly stable and still strictly worse than one that can tell real from synthetic.

## What to instrument

Four numbers per round, logged together: held-out loss on real data, held-out loss on the previous round's synthetic data, distinct-2 on a fixed prompt set, and tail coverage at a fixed $\epsilon$. The gap between the first two is the tilt. The last two are the narrowing.

Decide $\lambda$ in advance and write it down, because the moment synthetic data becomes cheaper than annotation, $\lambda$ decays on its own. Log $n_t$ next to every metric, since every variance term above scales as $n^{-1}$ and a metric that moved because the sample count moved is not a metric about the model.

Then decide what the filter is for before trusting an improvement. The filter is the thing that chooses the fixed point. If you cannot state which distribution your loop converges to, you have not finished designing it.

The model does not decide where self-training ends up. The operator does. Most teams never write the operator down, which is why the round-four behaviour is a surprise.

## Sources

- Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian and Dacheng Tao, [A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](https://arxiv.org/abs/2502.18865), arXiv:2502.18865 (2025).
- Shirong Xu, Hengzhi He and Guang Cheng, [A Probabilistic Perspective on Model Collapse](https://arxiv.org/abs/2505.13947), arXiv:2505.13947 (2025).
- Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho and Sanmi Koyejo, [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413), arXiv:2404.01413 (2024).
- Rylan Schaeffer, Joshua Kazdan, Alvan Caleb Arulandu and Sanmi Koyejo, [Position: Model Collapse Does Not Mean What You Think](https://arxiv.org/abs/2503.03150), arXiv:2503.03150 (2025).
- Zizhao Hu, Mohammad Rostami and Jesse Thomason, [Multi-modal Synthetic Data Training and Model Collapse: Insights from VLMs and Diffusion Models](https://arxiv.org/abs/2505.08803), arXiv:2505.08803 (2025).
- Millen Kanabar and Michael Gastpar, [Model non-collapse: Minimax bounds for recursive discrete distribution estimation](https://arxiv.org/abs/2501.19273), arXiv:2501.19273 (2025).
- Xuekai Zhu, Daixuan Cheng, Hengli Li, Kaiyan Zhang, Ermo Hua, Xingtai Lv, Ning Ding, Zhouhan Lin, Zilong Zheng and Bowen Zhou, [How to Synthesize Text Data without Model Collapse?](https://arxiv.org/abs/2412.14689), arXiv:2412.14689 (2024).
- Nicolas Zucchet, Jörg Bornschein, Stephanie Chan, Andrew Lampinen, Razvan Pascanu and Soham De, [How do language models learn facts? Dynamics, curricula and hallucinations](https://arxiv.org/abs/2503.21676), arXiv:2503.21676 (2025).
