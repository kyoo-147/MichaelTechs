---
title: "Benchmarks After Saturation"
description: "A benchmark that saturates stops separating models; the remaining signal is in the failures, not in harder items or a prettier leaderboard."
pubDatetime: "2026-09-12T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Benchmarks
  - Evaluation
categories:
  - Math
---

Consider a results table. The number in the cell is 97.4. Next to it are 97.1 and 96.9, and the largest of the three is bold. The table has a caption and a footnote about the number of shots. It does not say what a gap of 0.3 points means, and nothing else in the document does either.

The argument that follows a table like that is about whether the 0.3 is real. Nobody in the discussion knows the item count, the samples per item, or whether the three runs shared a prompt template. Two years ago the same benchmark read 61 and 78 and the gap mattered. Now it is producing a number shaped like a measurement.

A saturated benchmark still works. That is the problem. An exhausted instrument keeps returning readings, and a reading in a table looks identical whether or not it resolves anything.

## What Saturation Actually Destroys

What dies first is discrimination, not accuracy.

Take a benchmark of $N$ items and a set of models, and let $q_i$ be the fraction answering item $i$ correctly. An item with $q_i = 1$ is passed by everything. An item with $q_i = 0$ is failed by everything. Neither contributes to a comparison, because every model contributes the same value. Only items in between separate them, and separation is largest near $q_i = 0.5$.

The benchmark is effectively worth:

$$N_{\text{eff}} = \sum_{i=1}^{N} 4 q_i (1 - q_i)$$

This is the Rasch item information weight, rescaled so a coin-flip item counts as one. An item at $q_i = 0.5$ contributes 1. An item at $q_i = 0.95$ contributes 0.19. An item at $q_i = 0.99$ contributes 0.04. For a 2,000 item test whose median item sits at 0.985, the sum comes to about 118. The nominal count is untouched, which is why the collapse is easy to miss.

A 0.3 point gap on an effective 118 items is about a third of an item. That does not make the gap meaningless. It makes it an unresolved difference in the classification of one item, a narrower claim than a ranking.

Benchmark Radar assembles 1,283 source records from four catalogs carrying 12,916 numeric observations across 790 of them, refreshed from 37 sources, with saturation views ([arXiv:2609.11115](https://arxiv.org/abs/2609.11115), Wu et al., 2026). A tool built to track saturation is evidence that saturation is now the normal end state.

## The Residual Two Percent Is Not a Difficulty Signal

Hard items on a saturated benchmark are not a random sample of difficulty. They are a mixture: items needing a longer chain of inference, items that are ambiguous, items whose label is contested, items with an unusual expected format, items outside the construct the benchmark was built for. One pass rate averages all of it into a column.

Kim, Gu, Tiwari and Kolter give a reason to expect concentration. On parameterized GSM8K templates they observe that failures are systematically patterned and that a small subset of inputs accounts for the majority. They exploit this with a cross-entropy search over the input space, reporting up to a 156.22x reduction in the inferences needed to estimate a failure rate, across Qwen2.5-Math-7B-Instruct, gpt-oss-20b-low and Gemini 2.5 Flash Lite ([arXiv:2605.11209](https://arxiv.org/abs/2605.11209), Kim et al., 2026).

That result is usually filed under sampling efficiency. The useful reading concerns what a saturated score is. If the failure surface is that uneven, the aggregate is a weighted average over a bumpy region, and changing the weights - a different template, a different slice of the parameter space, a different prompt - moves the number, not the model.

Their second observation matters more. Models with indistinguishable accuracy on the standard benchmark can differ substantially in estimated failure rate. The mean saturates. The tail does not.

Two failure rates, 0.026 and 0.029, both round to 0.03, and their ratio is 1.12. Near-identical accuracies can hide an eleven percent spread in how often the systems are wrong.

## The Contrarian Case: Harder Is Not Better

The reflex is to build a harder benchmark. I think that reflex is usually wrong.

Difficulty and validity are separate axes, and selecting on difficulty selects on both. Filter candidate items by low pass rate and you keep the union of every reason an item is hard: it needs a longer inference chain, it is ambiguous, the label is wrong, the expected format is unusual, it depends on trivia no deployed system should store. A pass-rate filter cannot separate those causes, because it only sees the outcome.

A supply argument makes this worse. Writing an unambiguous arithmetic item is easy. Writing one at the frontier of a technical field, with a checkable answer and a reasoning path experts would accept, is much harder and far fewer people can do it. Verification cost rises with difficulty, so label error rises with it, exactly where the new benchmark wants resolution. That is a hypothesis about construction, not something I measured.

A harder benchmark also buys headroom rather than resolution. A frontier at 40 percent gives you a spread again, but on a fresh item set whose failure causes are unknown, and you have discarded what the saturated test still had: a fixed item set you can run many times per item.

That property is worth more than it looks. Once the mean is pinned, the variance becomes measurable. Kim et al. do exactly that, spending compute on the shape of the failures instead of the size of the score. My reading is that saturation was the precondition for the work, not the obstacle to it.

Where a harder benchmark is right: when the construct is missing. If the test has no items requiring multi-step tool use, or none in the language you ship in, that is coverage, and error analysis on the existing items will not fix it. A coverage problem needs a new construct. A resolution problem needs better statistics on the old one. Harder items address neither cleanly.

## A Score Is a Claim With an Expiry Date

Gilda and Gilda argue that evaluation results are epistemic claims rather than measurements, with three properties: formality, where human judgement counts as stronger evidence than an automated metric; scope, where a result applies to the tested distribution and no further; and a validity window, where results expire as contamination accumulates and the input distribution moves ([arXiv:2607.26191](https://arxiv.org/abs/2607.26191), Gilda and Gilda, 2026). They also name what averaging does: aggregate several signals and the reported confidence can exceed the reliability of the weakest, which they call trust inflation.

The demonstration I keep returning to: on the public HELM leaderboard, across 54 frontier models and ten scenarios, the top five by mean score and the top five by weakest-link aggregation are completely disjoint. The same numbers, two defensible operators, no overlap.

That is observed, and it is not a criticism of HELM. A leaderboard order is a function of the scores and the operator, so an order that does not name its operator reports half its inputs.

Saturation is one of three ways a benchmark stops informing, and the only one visible in the aggregate.

| Mechanism | What it looks like | What separates it |
|---|---|---|
| Ceiling | scores cluster near the top and between-model variance shrinks | effective item count falls; per-item pass rates concentrate above 0.95 |
| Contamination | scores rise without a matching gain on fresh or held-out items | a gap opens between public and private splits |
| Distribution shift | scores drift or destabilise as inputs move from the original sample | slice-level variance grows before the aggregate moves |

The three get conflated. A score that stops moving is called saturated when it may be contaminated or out of scope. The ceiling is easiest to diagnose and least dangerous, because it announces itself in the spread. Contamination and shift are quiet, and both make a benchmark look healthy until the decisions resting on it are wrong.

## What to Report Instead

Error analysis is not a softer alternative to a leaderboard. It answers a better question.

A leaderboard answers how often this model is right. That answer expires and its resolution shrinks. Error analysis answers what the model is doing when it is wrong. That produces categories, and categories transfer: a model that fails on long inputs fails on long inputs in the next benchmark, while a model at 96.2 percent has 96.2 percent of nothing in particular.

Five things belong in a release note beside any accuracy figure.

**Aggregate accuracy, as a gate.** Keep the number. It catches gross regressions and it is what users experience. Stop sorting on it.

**Effective item count.** Report $N_{\text{eff}}$ next to $N$. Seeing 118 beside 2,000 conveys the resolution without the formula.

**Run-to-run disagreement.** Run the same pipeline twice with different seeds and report the spread. That spread is the floor on any difference you intend to claim, and at saturation it usually exceeds the difference.

**A failure taxonomy with counts.** Not "the model made arithmetic errors", but which arithmetic, on what kind of item, with two verbatim traces. Read the first hundred failures before counting them. That ordering is opinion, not method.

**The operator, and a second one.** Report the ranking under mean aggregation and under a weakest-link rule. If the top five do not overlap, you have learned something about your evaluation.

The smallest useful version of the first three is a paired comparison: both models see the same items, and pairing removes the item-difficulty variance that dominates an unpaired interval at saturation.

```python
import numpy as np

def compare(a, b, n_boot=10_000, seed=0):
    """a, b: per-item correctness for two models, same items, same order."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    q = (a + b) / 2                        # per-item pass rate across the pair
    n_eff = 4.0 * (q * (1.0 - q)).sum()    # items the test is effectively worth
    diff = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"nominal": int(a.size), "n_eff": round(float(n_eff), 1),
            "diff": round(float(diff), 4),
            "ci": (round(float(lo), 4), round(float(hi), 4)),
            "resolves": bool(lo > 0.0 or hi < 0.0)}
```

The `resolves` flag is the point. If the interval straddles zero, the correct headline is no measurable difference. That is a finding. It is less exciting than a bold cell, and it is what the data supports.

None of this is expensive. Any evaluation run already produces the per-item correctness vector, and usually discards it for the mean.

## What I Would Keep

I would keep accuracy, and leaderboards as an entry point: a table of 54 models across ten scenarios is useful for finding candidates worth testing. What I would drop is the second step, where the order gets treated as a fact about the models rather than one defensible summary of item-level outcomes under one operator.

My view, and it is a view: the problem is not that benchmarks are too easy. It is that a saturated benchmark keeps emitting a number in the same format it used when the number meant something. Format carries an implicit claim about resolution, and a reader cannot audit that claim from outside.

So the useful question after saturation is not what to build next. It is what is left. The failures are still there, concentrated on a small set of inputs, differing between models that look identical in the mean, and they are the only part of the result that transfers to the next distribution you meet. A benchmark that saturates has not run out of information. It has run out of information in the column you were reading.

## Sources

- [arXiv:2609.11115](https://arxiv.org/abs/2609.11115) - Wu, K. et al. (2026). Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation.
- [arXiv:2605.11209](https://arxiv.org/abs/2605.11209) - Kim, E. et al. (2026). Measuring Five-Nines Reliability: Sample-Efficient LLM Evaluation in Saturated Benchmarks.
- [arXiv:2607.26191](https://arxiv.org/abs/2607.26191) - Gilda, S. and Gilda, S. (2026). Position: Evaluation Scores Are Perishable Knowledge Claims.
