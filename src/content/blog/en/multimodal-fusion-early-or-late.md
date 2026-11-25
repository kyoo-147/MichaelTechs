---
title: "Multimodal Fusion, Early or Late"
description: "Early fusion shares one representation and pays for alignment; late fusion keeps separate heads and pays for correlated errors at the join."
pubDatetime: "2026-07-04T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Multimodal
  - Fusion
categories:
  - Math
---

A tile encoder returns a grid of patch vectors. A text encoder returns one vector per caption, or one per token. Before the first training run, someone decides where the two streams meet: concatenated into one vector and passed to a classifier, or kept apart until two heads emit two scores. That choice is usually made in one sentence and treated as settled.

It is not settled. The fusion point fixes what the loss can see, how much of each modality's geometry has to be shared, and which modality ends up carrying the decision.

## Fusion is where the loss becomes joint

Write them down.

Early fusion builds $z = [\phi_v(x_v);\,\phi_t(x_t)]$ and trains a single head $g$ on $L(y, g(z))$. One loss reaches both towers.

Late fusion trains heads $h_v$ and $h_t$ on their own inputs and combines scores afterwards. Each head has to be predictive alone, because nothing in its objective refers to the other modality.

The layer index is not the real distinction; what matters is whether the training objective is joint. A model can concatenate at layer nine and still be late fusion if the branches are trained separately. A model can combine at the last layer and still be early fusion if one loss backpropagates through both towers.

Joint training can produce a representation in which neither half is useful alone. Late fusion cannot, by construction.

Late fusion also has an exact justification. If the modalities are conditionally independent given the label, then

$$p(y \mid x_v, x_t) \propto p(y)\, p(x_v \mid y)\, p(x_t \mid y),$$

so the log posterior is a constant plus a sum of per-modality terms. Averaging two log-likelihood scores is not a compromise under that assumption; it is the posterior. The assumption is the whole question.

## The interaction term decides more than the architecture

Define the gap between joint information and the sum of marginals:

$$\Delta = I(Y; X_v, X_t) - I(Y; X_v) - I(Y; X_t) = I(X_v; X_t \mid Y) - I(X_v; X_t).$$

Positive $\Delta$ is interaction: the pair says more than the parts. Negative $\Delta$ is redundancy.

Let $Y = X_v \oplus X_t$ with independent uniform bits. Each marginal carries no information about the label, so a unimodal head is at chance, and any average or product of chance-level heads is still at chance. Only a function of the pair separates the classes. Now let $X_v = X_t = Y$. Then $\Delta = -H(Y)$, the second modality adds no bits, and a late-fusion average loses nothing.

This is why the debate stalls. Each side is correct on a different task. Early fusion is the only option that can exploit $\Delta > 0$, and it can waste capacity chasing interactions that are not there. Late fusion is exact when $\Delta \le 0$ and blind when $\Delta > 0$, regardless of the combiner's size, unless the combiner is itself trained jointly on both inputs; then it is early fusion with extra steps.

The practical reading is to estimate $\Delta$ or bound its sign. If the label is a conjunction of image content and text content, no late-fusion engineering recovers it.

## A shared space is a loan against alignment

Concatenation is an assertion. It says dimension $i$ of the image vector and dimension $i+d$ of the text vector are commensurable, because the first linear layer mixes them through one weight matrix with no per-modality scaling. Nothing in the data guarantees it.

The cleanest way to price the assertion is a linear alignment. Centre both embedding sets, then solve

$$W^\star = \arg\min_{W^\top W = I} \mathbb{E}\,\| W \phi_v(x_v) - \phi_t(x_t) \|^2 .$$

With cross-covariance $M = \mathbb{E}[\phi_v \phi_t^\top] = U\Sigma V^\top$, the optimum is $W^\star = VU^\top$ and the residual is $\mathbb{E}\|\phi_v\|^2 + \mathbb{E}\|\phi_t\|^2 - 2\sum_i \sigma_i$. Normalised by $\mathbb{E}\|\phi_t\|^2$, that is the distortion the best orthogonal map leaves behind, which caps what rotation-based alignment can recover. What remains, the joint loss must learn as a nonlinearity, charged against the task's capacity.

Observed: Lee, Chang, Tsai, Diao and Lee argue that conventional pipelines flatten image embeddings and lose hierarchical structure, then align a vision encoder to a language model through layered cross-attention under a three-stage schedule ([arXiv:2604.00086](https://arxiv.org/abs/2604.00086), 2026). My reading is that the schedule exists because a cold joint loss on badly aligned towers is unstable. It is the alignment bill, paid in epochs instead of parameters.

A second cost is dimensional. A shared space of width $d$ must hold both modalities, so if $d$ was sized for the longer vectors, the other arrives compressed. My heuristic is to size the width as the sum of the two intrinsic dimensions minus what is shared, then check the residual above. It is a heuristic, not a theorem.

## What late fusion buys, and what it hides

The buys are real. Heads can train on unpaired data, a modality can be added without retraining, each head is debuggable alone, and a missing sensor costs one term.

What it hides is arithmetic. Averaging probabilities is a mixture. Averaging logits is a product of experts:

$$\sum_m w_m \log p_m(y \mid x_m) \quad \text{versus} \quad \log \sum_m w_m\, p_m(y \mid x_m).$$

The left is a geometric mean and sharp: one expert assigning near-zero probability drags the score to near-zero. The right is forgiving and can be carried by one overconfident, wrong head. Neither is the posterior unless the heads are calibrated and independent, and heads trained on the same labels with the same preprocessing are neither.

Observed: Swati, Roy and Ntoutsi compare the two on a recruitment task built on FairCVdb and report that late fusion produced highly generalised mean scores with higher MAEs, while early fusion tracked the ground truth more closely for both demographic groups ([arXiv:2407.16892](https://arxiv.org/abs/2407.16892), 2024). One dataset, one task, a fairness framing rather than an architecture benchmark. Still, the modular option lost on accuracy, and their explanation is the one above: late fusion smoothed away per-modality detail.

## The contrarian case: intermediate fusion is mostly a capacity story

The framing of early, intermediate and late treats cross-attention as a separate answer. I think most reported wins for the middle do not isolate the fusion point at all.

Take the paper above. Its comparison moves three things at once: where the towers meet, how much capacity the joint model has, and the training schedule. The reported gains on MME, GQA, OK-VQA and ScienceQA are real and attributed to hierarchical feature integration. But the experiment cannot tell you whether the gain came from the junction or from the capacity attached to it. That is not a criticism of the authors: matching parameters across fusion points is expensive and rarely rewarded.

The ablation that would settle it is narrow: hold total parameters, data, schedule and token budget fixed, and move only the junction. A learned alignment is not a third mechanism; it is early fusion with the alignment trained explicitly.

Early fusion is not free either. Ch'ng and Chua concatenate demographic metadata with tokenised essays in a DistilBERT scorer and report that the naive early fusion degraded accuracy, with quadratic weighted kappa falling from 0.727 to 0.656, validation loss rising from 1.25 to 1.29, and score parity dropping from 15 to 12 of 19 tests ([arXiv:2606.21066](https://arxiv.org/abs/2606.21066), 2026). The added dimension was construct-irrelevant and the shared space had no way to discard it. A joint loss uses whatever is available, including a feature that only correlates with the label in the training split.

So my working rule is unattractive but cheap. If you cannot run that ablation, treat intermediate fusion as early fusion with a larger bill and a harder optimisation. If you can, run it: the answer is task-specific, and architecture fashion is no substitute for measurement.

## When one modality dominates

A joint loss is a competition for gradient. Early in training, the tower that produces larger and more consistent gradients captures the shared parameters, and the other is not trained so much as regularised.

Two measurements show the imbalance. The first is the ratio of gradient norms on the two encoders, $\rho = \|\nabla_{\theta_v} L\| / \|\nabla_{\theta_t} L\|$. The second is the ratio of Jacobian norms of the fused representation with respect to each input.

```python
def grad_ratio(loss, enc_v, enc_t):
    gv = torch.autograd.grad(loss, list(enc_v.parameters()),
                             retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(loss, list(enc_t.parameters()),
                             retain_graph=True, allow_unused=True)
    nv = torch.sqrt(sum((g ** 2).sum() for g in gv if g is not None))
    nt = torch.sqrt(sum((g ** 2).sum() for g in gt if g is not None))
    return (nv / nt).item()
```

A small and falling ratio has two causes that demand opposite responses. Either the weak modality carries little label information, so dominance is correct and suppressing it damages the model, or it is harder to optimise, so dominance is an artefact of the objective.

The test that separates them is a unimodal head on the weak modality alone. Near chance, and the dominance is honest. Well above chance while the joint model still ignores the input, and the fusion is throwing information away.

A third cause is easy to miss: token count. A long text sequence and a small patch grid contribute very different numbers of vectors to the same attention, so the loss sums over more text terms. My hypothesis is that this alone can produce apparent dominance with no informational cause, which would make per-modality token normalisation cheaper than a new architecture.

Observed context: Alam, Chowdhury and Hoque Prince describe adapter stacks for tri-modal alignment in low-resource, non-English settings, and note that pipelines remain English-centric ([arXiv:2605.17152](https://arxiv.org/abs/2605.17152), 2026). The dominance they describe is a data problem before it is an architecture problem. The modality with more and better data wins the shared space, and no junction choice repairs that.

## Deciding with two measurements

Two numbers do most of the work: the gap between the joint model and the sum of the unimodal models on held-out data, a proxy for the sign of $\Delta$, and the normalised alignment residual from the Procrustes fit on frozen embeddings.

Read them together. A large positive gap says the interaction is real and the shared space must be paid for, so budget alignment deliberately: staged schedules, per-modality normalisation, a width not inherited from the larger encoder. A gap near zero or negative says late fusion is not a compromise but the correct model.

One bias to watch. The gap is estimated with fitted unimodal heads, and weak heads understate the marginals, which flatters fusion. Give the baselines generous capacity, or you measure your baseline budget.

| Question | Embedding-level (early) | Decision-level (late) |
| --- | --- | --- |
| Where the loss is computed | Once, on the joint vector | Once per modality, then combined |
| Can represent $\Delta > 0$ | Yes | Only if the combiner is trained jointly |
| Data it needs | Aligned pairs for the task | Aligned pairs, plus optional unpaired data per head |
| Missing modality at inference | Needs a mask token or imputation | Drop the head and renormalise |
| Adding a modality later | Retrain the joint model | Train one new head |
| Dominant-modality risk | High, the shared space can collapse | Lower, but the combiner can still ignore a head |
| Calibration | One score to calibrate | Needs per-head calibration first |
| Typical failure | Alignment absorbs capacity | Correlated errors behind an averaging rule |

The usual order is to choose the junction first and meet the alignment bill later, when the loss curve is flat and nobody remembers the slide. Naming the bill up front is cheaper. The junction is a claim about whether the two modalities interact given the label, and that claim can be measured before the first training run.

## Sources

- Eugene Lee, Ting-Yu Chang, Jui-Huang Tsai, Jiajie Diao and Chen-Yi Lee, "Hierarchical Pre-Training of Vision Encoders with Large Language Model", [arXiv:2604.00086](https://arxiv.org/abs/2604.00086), 2026.
- Firoj Alam, Shammur Absar Chowdhury and Enamul Hoque Prince, "Multilingual and Multimodal LLMs in the Wild: Building for Low-Resource Languages", [arXiv:2605.17152](https://arxiv.org/abs/2605.17152), 2026.
- Teik Peng Ch'ng and Hui Na Chua, "Demographic Metadata as Construct-Irrelevant Noise in DistilBERT-Based Automated Essay Scoring", [arXiv:2606.21066](https://arxiv.org/abs/2606.21066), 2026.
- Swati Swati, Arjun Roy and Eirini Ntoutsi, "Exploring Fusion Techniques in Multimodal AI-Based Recruitment: Insights from FairCVdb", [arXiv:2407.16892](https://arxiv.org/abs/2407.16892), 2024.
