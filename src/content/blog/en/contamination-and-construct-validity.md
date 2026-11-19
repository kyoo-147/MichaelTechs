---
title: "Contamination and Construct Validity"
description: "A benchmark score measures the construct your items elicit, not the capability you named it after, and contamination is one of several ways that gap opens."
pubDatetime: "2025-08-09T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Evaluation
  - Contamination
categories:
  - Math
---

Qwen2.5 responded to reinforcement learning on arithmetic with gains that other model families did not reproduce, including under reward signals that were random or simply wrong. The explanation that survived scrutiny is unglamorous: those benchmarks were in the pretraining crawl, and the models that answered nonsense rewards were the models that had already read the questions. The benchmark was meant to measure arithmetic reasoning. On those items, for those models, it measured a mixture of arithmetic and recall of a text distribution.

Contamination is one way the wrong thing gets into a score. Construct validity is the wider question of whether the score was ever about the capability in the name. The first problem has detectors, however imperfect. The second needs a design decision, and design decisions are the part that gets skipped.

## Two ways a score goes wrong

Measurement people have names for the two failures. Construct underrepresentation is when the items cover less than the construct: you named reasoning, and you sampled a hundred word problems from the same three textbooks. Construct-irrelevant variance is when the items let something else into the score: answer position, formatting habits, tokenizer quirks, memorised text.

Write the score as an estimate of a target. Let $\theta$ be the construct and $S_n$ the mean score on $n$ items:

$$S_n = \theta + b + \varepsilon_n, \qquad \mathrm{SE}(\varepsilon_n) = \sqrt{\frac{p(1-p)}{n}}.$$

Validity is the question of whether $b$ is small enough for the claim you want to make. Contamination contributes to $b$. So does a benchmark whose items are all the same shape. The second is not a data hygiene failure and cannot be fixed by better data hygiene, which is why contamination gets the attention: it is the part of $b$ with an audit trail.

The idealisation costs something. Real constructs are not scalar and no benchmark is unbiased for one. Treat the decomposition as bookkeeping: it separates the part of the error that shrinks with $n$ from the part that does not.

## What contamination actually adds

Suppose a fraction $c$ of items appeared in training and are answered correctly with probability near one, while the rest are answered at the model's honest rate $p$:

$$p_{\mathrm{obs}} = c + (1-c)\,p = p + c\,(1-p).$$

The inflation is $c(1-p)$, and the shape matters more than the size. At $p = 0.5$, ten percent contamination buys five points. At $p = 0.2$ it buys eight. Contamination flatters weak models more than strong ones, in absolute points, because there is more headroom left to steal. It also compresses the difficulty range: with $c = 0.4$, a fifteen-point honest gap becomes a nine-point gap. This is arithmetic, not a finding, but it is the arithmetic that makes contaminated leaderboards look tight at the top.

The published evidence runs the same direction. Wu and colleagues traced widely circulated reinforcement-learning results and found the gains concentrated in models whose pretraining data plausibly contained MATH-500, AMC and AIME items. They then built a generator for clean arithmetic problems of arbitrary length and difficulty. On that leakage-free set, only accurate reward signals produced steady improvement past the base model. Random and incorrect rewards did not. The contaminated benchmark had supported a claim about reward design that the clean benchmark reversed.

That reversal is a construct failure, not a score adjustment. The number was never measuring sensitivity to reward. It was measuring sensitivity to reward on items the model had already seen, which is a different quantity with a different answer.

## The contrarian read: this is not a data hygiene problem

The standard framing says contamination is a pipeline defect. Items leaked into the crawl, so refresh the items, and the score becomes trustworthy again. Rewrite the questions, perturb the numbers, generate paraphrases, reorder the options, report the new number.

Sun and colleagues tested that framing directly. Their setup covered ten models, five benchmarks, twenty mitigation strategies and two contamination scenarios, and scored each strategy on two axes: fidelity, meaning the updated item still tests what the original tested, and contamination resistance. The result is the one worth memorising. No strategy improved resistance significantly over the vanilla case, where the benchmark is simply left alone, across all benchmarks, and none balanced fidelity against resistance.

The reason is structural rather than empirical. An item that preserves its difficulty, its answer distribution and its reasoning path is close to the original, so it inherits the original's accessibility. An item far enough away to be unmemorisable is far enough away that it may be measuring something else. Fidelity and resistance pull in opposite directions and the trade has no free corner. You are choosing between a contaminated question that still tests the right thing and a clean question that tests a different thing. Sometimes the second is better. It is not automatically better, and it is not the same benchmark.

So the standard framing is incomplete in a way that matters operationally. It treats contamination as separable from the construct. In practice the two are coupled through the item, and every mitigation spends one to buy the other. A paper that reports a refreshed benchmark without reporting its fidelity has changed the subject and kept the title.

## Why n-gram overlap is the wrong instrument

The usual check computes $k$-gram overlap between benchmark items and a training corpus. It has two gaps, and they are independent.

The first is that it measures the corpus, not the model. A match shows a string was reachable. It does not show the model stored it, and it certainly does not show the model uses it at inference. A miss shows even less: if the item was paraphrased, translated, or re-derived as a formula, there is nothing to match.

The second is that it tests coverage, not similarity. A detector fires when any window of $k$ tokens survives intact. An item of $L$ tokens contains $L - k + 1$ such windows. Take $L = 120$ and $k = 13$: there are 108 windows, and a single substitution destroys at most $k = 13$ of them, leaving 88 percent intact and the flag raised. Space substitutions every $k$ tokens and no window survives at all. A light normalisation pass, or a model that restates the problem before solving it, clears the detector while leaving the semantics untouched. The auditor does not know the edit density, and the detector's sensitivity is a function of exactly that unknown.

The counter-evidence deserves a hearing. Hidayat and colleagues compared detectors under simulated leakage and found the n-gram method had the highest F1 of those tested, then used it to produce cleaned versions of MMLU and HellaSwag. Both results are true. Their leakage was injected by inserting benchmark items into a continual-pretraining corpus, which is the friendly case for a coverage test: verbatim, recent, contiguous. Real leakage is a crawl, and a crawl is not an append. My reading is that the n-gram figure is a good measurement of a simulation whose contamination model was chosen to be detectable.

The honest use of the tool is reporting, not gating. Report the overlap rate, report the direction of the failure, and report that the test cannot see paraphrase. Do not report a pass.

## Item families are the only durable defence

If items can be memorised, the durable move is to not have fixed items. Generate them from a parameterised family, publish the family's distribution, and hold the instance until after the run.

The generator does not have to be clever. It has to be deterministic, seeded, and committed to before anyone sees a score.

```python
import hashlib, random

def item(seed: int, index: int, depth: int = 3):
    rng = random.Random(seed * 1_000_003 + index)
    value = rng.randint(2, 99)
    expr = str(value)
    for _ in range(depth):
        op = rng.choice("+-*")
        rhs = rng.randint(2, 12)
        if op == "*":
            value *= rhs
        elif op == "+":
            value += rhs
        else:
            value -= rhs
        expr = f"({expr}{op}{rhs})"
    return expr, value

def commit(seed: int, texts) -> str:
    h = hashlib.sha256(str(seed).encode())
    for t in sorted(texts):
        h.update(t.encode())
    return h.hexdigest()[:12]
```

The depth parameter is the construct. Depth three and depth eight are different capabilities, and a family lets you say which one you measured. Publish the hash before the run and the generator after it. That ordering is what makes the held-out claim checkable by someone who does not trust you.

The cost is honesty about thinness. A generated family measures a narrow construct precisely. A curated benchmark measures a rich construct vaguely. Neither is wrong, but the report should say which one it is doing, and a paper that swaps one for the other mid-stream has to say so.

## Report the interval or the number means nothing

A benchmark score is a proportion, so it carries a binomial standard error. Items also cluster: five variants of one problem template are not five independent observations. The variance inflates by the design effect $1 + (m-1)\rho$, where $m$ is the cluster size and $\rho$ the intra-cluster correlation.

| Items $n$ | $\mathrm{SE}$ at $p = 0.5$ | 95 percent half-width | Effective $n$, $\rho = 0.3$, $m = 5$ |
|---|---|---|---|
| 100 | 5.0 | 9.8 | 45 |
| 500 | 2.2 | 4.4 | 227 |
| 1000 | 1.6 | 3.1 | 455 |
| 4000 | 0.8 | 1.6 | 1818 |

At 500 items with clustered weighting, a three-point improvement is inside the noise and a two-point regression is not evidence of anything. Most reported benchmark deltas live in that region. This does not make them false. It makes them unresolved, and an unresolved delta should be written as one.

Contamination belongs in the same paragraph. If $c$ cannot be bounded, say so and give the range $[\,p,\ p + c_{\max}(1-p)\,]$. If it can be bounded, the range is the result. A point estimate from a benchmark with unknown contamination is a number with a hidden second term, and the second term is usually larger than the difference being claimed.

## What a report should contain

Six things, in prose rather than a table, because each needs a caveat attached.

The construct sentence: one line naming the capability and what the items actually elicit, with the gap admitted. The provenance of the items, including whether they were generated and what the family parameters were. The contamination evidence, with the detector named and its blind spot stated. The interval, clustered if the items are. The frozen seed or hash, if generation was involved. And the benchmark version, because an instrument that changes between submissions is a moving target rather than a fixed one.

Most of that has no leaderboard payoff. A bench that reports a range and a blind spot looks weaker than one that reports a decimal, even when the decimal is the fiction. That asymmetry is why the practice persists, and better tooling will not fix it.

A benchmark measures the construct its items elicit. The name is a claim about that, and a claim is not a measurement. Contamination is leakage that inflates the number. Construct validity asks whether the number was ever about the thing on the label. The second question is harder, pays worse, and survives a fresh crawl.

## Sources

Yifan Sun, Han Wang, Dongbai Li, Gang Wang, Huan Zhang. "The Emperor's New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination." [arXiv:2503.16402](https://arxiv.org/abs/2503.16402), 2025. https://arxiv.org/abs/2503.16402v1

Naila Shafirni Hidayat, Muhammad Dehan Al Kautsar, Alfan Farizki Wicaksono, Fajri Koto. "Simulating Training Data Leakage in Multiple-Choice Benchmarks for LLM Evaluation." [arXiv:2505.24263](https://arxiv.org/abs/2505.24263), 2025. https://arxiv.org/abs/2505.24263v1

Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Huijie Lv, Ming Zhang, Yanwei Fu, Qin Liu, Songyang Zhang, Qi Zhang. "Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination." [arXiv:2507.10532](https://arxiv.org/abs/2507.10532), 2025. https://arxiv.org/abs/2507.10532v1
