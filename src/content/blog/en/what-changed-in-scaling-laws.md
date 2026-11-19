---
title: "What Changed in Scaling Laws"
description: "Early scaling laws mapped compute to loss on one distribution. Data quality, repetition and fitting ambiguity broke the clean version."
pubDatetime: "2026-11-14T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Scaling Laws
  - Evaluation
categories:
  - Math
---

You have a fixed compute budget and a corpus. A scaling law lets you write down the loss for a model of size $N$ trained on $D$ tokens, then choose the split. That is the whole promise: turn a budget into a prediction before you spend it. The prediction carries a scope condition that is easy to miss, because it is stated once in the setup and never again. The law describes loss on a distribution you measured, under a recipe you ran. Almost every decision that matters happens outside that sentence.

## The original claim was narrower than its citations

Kaplan and colleagues fit cross-entropy loss against model size, dataset size and training compute, and found power laws spanning more than seven orders of magnitude. Architectural details such as width and depth had minimal effect within a wide range. The practical conclusion was blunt: compute-efficient training means very large models, a modest amount of data, and stopping well before convergence ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361), Kaplan et al., 2020).

Each term of the fitted form is a separate claim.

$$L(N,D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

The first two terms describe how fast you approach a floor. The third is the floor. All three constants are properties of the distribution, not of transformers. Change the corpus and the constants move. Change it badly enough and $L_\infty$ moves first, and that is the one setting the asymptote you were extrapolating toward.

The scope was honest and invisible in practice, because the assumption underneath it held at the time. Fresh tokens were plentiful, so nobody trained four epochs on a filtered subset.

## Chinchilla changed the allocation, not the shape

Two years later, Hoffmann and colleagues trained over 400 models between 70 million and 16 billion parameters on 5 to 500 billion tokens and re-fit the same kind of curve. The allocation came out different: model size and token count should scale together, so doubling parameters means doubling tokens. Their 70B model, trained on four times more data at the same compute as a 280B baseline, beat that baseline and several larger models, reaching 67.5% average on MMLU ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556), Hoffmann et al., 2022).

A power law survived a much larger re-fit and still produced a different recipe. The family of curves had roughly the right shape. The constants did not transfer. The law was never a claim that you know which member of the family you are on.

Both papers also assumed unlimited fresh data. That went unargued because it did not need arguing. It became the interesting variable only when it stopped being true.

## Repetition is not a monotone penalty

The natural first fix is to treat repeated tokens as discounted fresh tokens and keep the law additive. Chudnovsky and colleagues tested that and found something worse than a discount. Earlier controlled studies predated Chinchilla-style laws and could only measure repetition cost indirectly, so they fit a no-repetition law first, then priced repetition in FLOPs as Compute-Equivalent Gain and Loss ([arXiv:2606.24998](https://arxiv.org/abs/2606.24998), Chudnovsky et al., 2026).

Holding compute spent on repeated data constant, evaluation loss peaks at an intermediate repeat count. Repeating a moderate subset a moderate number of times damages the model more than repeating a large subset a few times or a small subset many times. The peak is well fit by a power law in model size, and that fit implies the most damaging repeat count grows faster than compute. The worst case moves away from you as you scale, the opposite of what a plan built on "repetition costs a little" predicts.

The number to keep: when repeated documents took 10% of the FLOPs budget in an exact-document repetition setting, on FineWeb-Edu-Dedup, the most damaging repeat count for a Qwen3-style 344M-parameter model at one token per parameter matched the loss of a no-repetition run using 67% of the FLOPs. You paid for a full run and received two thirds of one.

The structural point: a peak breaks the assumption that $D$ is a sufficient statistic for the corpus. Monotone damage could be absorbed as one more term. A maximum means two corpora with identical token counts and identical unique-document counts can have different losses depending on how repeats are arranged.

## Data quality became a schedule

Zhu and colleagues extended functional scaling laws by letting quality vary over time, and derived jointly optimal quality and batch-size schedules inside a feature-space regression model. Two regimes appear. In the noise-limited regime, a smaller batch converts cleaner data into more signal at comparable noise. In the signal-limited regime, placing clean data late suppresses terminal noise without giving up signal accumulation. That explains a real conflict: conventional decay schedules can work against curriculum pipelines, because one is trying to finish quietly and the other is trying to finish on the best data ([arXiv:2605.25698](https://arxiv.org/abs/2605.25698), Zhu et al., 2026).

Their schedule is Drop-Stable-Rampup: drop the batch size at the quality transition, hold it low to accumulate signal, then ramp up to suppress noise. On a 15B mixture-of-experts model midtrained on 108B tokens of proprietary general-domain data it improved average accuracy by 1.70 points over Warmup-Stable-Decay and 2.98 over cosine decay, including 4.23 on GSM8K and 2.80 on MATH. On a public math-and-code mixture it led all reported STEM, mathematics and code benchmarks, beating the strongest baseline by 3.27 overall on a 600M dense model and 5.25 on the same architecture.

Notice what the law has become. $D$ used to be a scalar. Now the ordering of data is a variable, and the optimum depends on which constraint binds, and when. A schedule is harder to plan against than a quantity, because it must be executed correctly to deliver the predicted number.

## The regime you are in is probably not the regime the law was fit for

The cleanest statement of what changed comes from Qin, Hamidieh and Alvarez-Melis. Classical compute-optimal scaling assumes unbounded fresh data, and pretraining increasingly runs where compute grows faster than high-quality data. Their Compute-Data laws interpolate between compute-optimal scaling, where data scales freely, and data-optimal scaling, where the corpus is fixed and compute grows without bound ([arXiv:2607.25271](https://arxiv.org/abs/2607.25271), Qin et al., 2026).

The mechanism is a token-effectiveness function $\eta$, which prices a derived token against a fresh one:

$$D_{\mathrm{eff}} = D_{\mathrm{fresh}} + \eta \cdot D_{\mathrm{derived}}, \qquad \eta \in [0,1]$$

They fit it for multi-epoch repetition and paraphrasing from 14M to 600M parameters on Dolma-3, and find it far from constant: it depends jointly on model size, the tokens-per-parameter ratio, and how much derived data you have added, and it saturates as the corpus expands. The fitted form implies diminishing returns when substituting compute for data, diminishing faster as either model size or data availability grows.

Training then partitions into three operational regimes: compute-bound, data-bound and model-bound. And classical compute-optimal allocation is suboptimal across most practically relevant settings. The recipe most teams inherit is not the one their situation calls for.

| Law | Treated as fixed | Lets you choose | Holds when |
| --- | --- | --- | --- |
| Kaplan 2020 | Distribution, fresh tokens | Model and data split | Corpus is stationary and large |
| Chinchilla 2022 | Distribution, fresh tokens | Model and data split at scale | Tokens stay scarce relative to compute |
| Chudnovsky 2026 | Nothing about repetition | Repeat structure, subset size | Repeats exist and are unevenly arranged |
| Zhu 2026 | Nothing about ordering | Quality and batch schedule | High-quality data is the scarce input |
| Qin 2026 | Nothing about data supply | How much derived data to accept | Compute outgrows available clean data |

## The contrarian take: the exponent was never the interesting part

The field has spent years arguing about which terms belong in the law: quality, repetition, sparsity, curriculum order. Meanwhile a quieter problem sat in the last stage, treated as routine.

Mlodozeniec and Frei describe that stage. You train across scales, evaluate on a target metric, then fit. The fit minimises a non-convex loss by local optimisation from a grid of initialisations and returns the best solution found. That fails twice: it cannot guarantee a nearly optimal solution, and it hides whether very different parameter choices fit nearly as well. The result is a Rashomon effect, where the sublevel sets of near-minimisers are broad regions whose extrapolations diverge at scale. Their fix is a deterministic branch-and-bound algorithm that guarantees a nearly optimal fit and maps the near-minimisers to the implied range of curves ([arXiv:2610.34514](https://arxiv.org/abs/2610.34514), Mlodozeniec and Frei, 2026).

My contrarian position is this. Whether data quality changed the law is secondary to whether the law's parameters were identified at all. If a hundred vectors fit your grid equally well and disagree by a factor at ten times the scale, the exponent you quote is a selection from a set, made by an optimiser's tie-breaking. Adding a quality term does not fix that. It adds a parameter to an under-determined fit.

```python
best, near = None, []
for theta0 in init_grid:
    theta = local_minimise(fit_loss, theta0)
    if best is None or fit_loss(theta) < fit_loss(best):
        best = theta
    if fit_loss(theta) <= fit_loss(best) * (1 + tol):
        near.append(theta)

# `best` is what gets published. `near` is what predicts.
spread = [extrapolate(t, scale) for t in near]
```

This does not make scaling laws useless. It makes the honest output an interval rather than a point, and that interval is often wide enough to change which experiment you should run. Knowing it before the run beats a slightly better exponent after.

## What a scaling law still cannot promise

It cannot promise a different distribution. The constants, especially the floor term, belong to the corpus you measured. A law fit on web text does not transfer to code except by accident.

It cannot promise your downstream metric. The fitted quantity is cross-entropy on held-out tokens, and nothing in the form connects that to accuracy on a task with a threshold.

It cannot promise the frontier is reachable, or unique. Extrapolation is a claim about form: a human chose it, a local optimiser fitted it, and the Rashomon result shows many vectors fit one grid and disagree at scale. The curve you extended may be the wrong curve, not the right one with error.

It cannot promise the corpus stays still. Quality enters as a time-varying term, repetition interacts non-monotonically with size, and the most damaging repeat count grows faster than compute. A stationary law applied to a moving corpus answers a question it was not fit for.

What I would do instead is unglamorous. Report the near-minimiser interval next to the point estimate. Hold out one scale and check the extrapolation against it. State the tokens-per-parameter ratio and which regime you are in, because that decides whether data or compute is the cheaper lever. Ask whether one more epoch at the current size beats one more month at a larger one: the CD result suggests the answer flips depending on where you already are. And note what none of these laws contain, which is deduplication, filtering or licensing.

A scaling law is still good at one thing. It ranks experiments cheaply, before you pay for them. Treat it as a way to order your next three runs and it earns its keep. Treat it as a promise about the fourth and you are trusting a fitted curve to know things about a corpus it has never seen.

## Sources

- Kaplan, McCandlish, Henighan et al. (2020). *Scaling Laws for Neural Language Models*. "Scaling Laws for Neural Language Models", [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann, Borgeaud, Mensch et al. (2022). *Training Compute-Optimal Large Language Models*. "Training Compute-Optimal Large Language Models", [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Chudnovsky, Kazdan, Levi et al. (2026). *Internal Data Repetition Destroys Language Models*. "Internal Data Repetition Destroys Language Models", [arXiv:2606.24998](https://arxiv.org/abs/2606.24998)
- Zhu, Wang, Wu et al. (2026). *How Should LLMs Consume High-Quality Data? Optimal Data Scheduling via Quality-Aware Functional Scaling Laws*. "How Should LLMs Consume High-Quality Data? Optimal Data Scheduling via Quality-Aware Functional Scaling Laws", [arXiv:2605.25698](https://arxiv.org/abs/2605.25698)
- Qin, Hamidieh, Alvarez-Melis (2026). *Bridging Compute- and Data-Optimal Pretraining*. "Bridging Compute- and Data-Optimal Pretraining", [arXiv:2607.25271](https://arxiv.org/abs/2607.25271)
- Mlodozeniec, Frei (2026). *Ambiguous Fits: The Rashomon Effect and Challenges in Scaling Law Estimation*. "Ambiguous Fits: The Rashomon Effect and Challenges in Scaling Law Estimation", [arXiv:2610.34514](https://arxiv.org/abs/2610.34514)
