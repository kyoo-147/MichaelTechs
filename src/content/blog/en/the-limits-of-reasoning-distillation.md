---
title: "The Limits of Reasoning Distillation"
description: "Trace distillation moves a teacher's style reliably and its difficulty-aware budget unreliably, and the gap is testable before you ship."
pubDatetime: "2026-03-21T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Distillation
  - Reasoning
categories:
  - Math
---

A team fine-tunes a 4B model on eight thousand traces sampled from a reasoning teacher. Accuracy on a held-out split of the same distribution jumps. Then someone rewrites the same problems with different numerals, same structure, and re-runs. Most of the gain is gone. A second team runs the same experiment and reports the opposite. Neither team is lying, and that is the interesting part.

Distillation does something. The trouble is that the thing it does is easier to measure than the thing you wanted.

## The loss asks for token agreement

Write the teacher as $\pi_T$ and the student as $\pi_\theta$. A trace $z = (z_1, \dots, z_T)$ for prompt $x$ is one sample from $\pi_T(\cdot \mid x)$. Supervised fine-tuning on that sample minimises

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(z_t \mid x, z_{<t}).$$

In expectation over the sampling of $z$, this is $\mathrm{KL}(\pi_T \,\|\, \pi_\theta)$ up to a constant. That identity is often read as a guarantee. It is not one. The objective constrains the student's next-token distribution on contexts that appear in the training traces. It says nothing about how many tokens a step deserves, when to backtrack, or when to abandon a branch. Those are properties of the teacher's policy over lengths and branches, and a single sampled trace renders them invisible. You see one path. The distribution over paths is what the teacher actually is.

This is a hypothesis about mechanism, not a measurement. The measurement exists, and it is less kind than the hypothesis.

## The difficulty signal arrives at half strength

Hu and colleagues test what they call the superficial mimicry hypothesis across 14 models, and the result is the cleanest statement of the problem I have seen. Reasoning teachers trained with reinforcement learning spend tokens roughly in proportion to problem difficulty. Their token spend tracks human difficulty ratings with a mean correlation of $\bar{r} = 0.64$. Distilled students, trained to imitate those same traces, drop to $\bar{r} = 0.34$.

The direction matters more than the magnitude. The students did not fail to produce reasoning-shaped text. They failed to produce text whose cost responds to the problem. Several of them scored below their own pre-distillation baselines, which the authors call negative transfer.

The paper's framing is a cargo cult: the ritual form is reproduced, the causal structure behind it is not. If you have watched a distilled model write a page of "let me carefully consider" for a two-line arithmetic problem, then write the same page for a genuinely hard one, you have seen this directly. Verbosity is easy to copy because it is a marginal property of nearly every trace in the corpus. Difficulty-conditioned budget is hard to copy because it is the one property that varies between traces, and fine-tuning sums loss over tokens rather than over problems.

## The Contrarian Case: The Problem Is the Unbounded Surface

The standard conclusion is that trace distillation is a dead end and only reinforcement learning on verifiable rewards teaches reasoning. I think that conclusion overfits to one design choice, and the choice is not traces. It is free-form traces.

Balasubramanian trains a CodeT5 student for automated program repair, using a large teacher to emit structured symbolic reasoning tags alongside fix-type labels. The tags form a closed vocabulary of causal properties of bugs, not prose. On the IntroClass benchmark, reasoning supervision improves macro-averaged performance, and the improvement is largest on the less frequent bug categories, which is where a shallow-correlation model hurts most. The paper also reports the honest part: correct reasoning traces correlate strongly with correct predictions but do not determine them.

Set that next to the collapse result and a different explanation fits better than "form never transfers". A free-form trace has an enormous surface: phrasing, connectives, hedging, sentence order, all sampled, all plausible, and all of it consuming gradient because the loss weights every token equally. Most of what the student has capacity to absorb is surface. A closed tag vocabulary shrinks the surface until the only thing left to learn is which property holds. Surface entropy, not surface form, looks like the variable. If your intermediate representation has ten thousand equally valid realisations, you have designed a task whose cheapest solution is style.

I would put that as an opinion with one supporting data point, not a settled finding. It is enough to make me distrust the blanket claim that traces cannot teach reasoning, and it points somewhere useful: the fix may be constraining the intermediate rather than spending more compute on the teacher.

## Three tests that separate the skill from the format

The practical question is diagnostic. Before shipping a distilled student you want to know which of two things happened, and aggregate accuracy cannot tell you, because it averages a real effect against a real artefact.

| Test | What you change | Skill transfer predicts | Format transfer predicts |
|---|---|---|---|
| Reskin | Numerals, names, units, problem order | accuracy roughly holds | collapse toward baseline |
| Format ablation | Strip connectives, cut to bare steps | graceful loss | large loss |
| Locality | Evaluate on skills absent from training | partial carryover | no carryover |

The reskin test is the cheapest and the most often skipped. Change only the surface you believe is irrelevant. If the gain is real, it survives.

Format ablation runs the other way. Keep the problems, remove the stylistic scaffolding from both training traces and student outputs. A student that learned a procedure degrades gracefully. A student that learned a register falls apart, because you deleted the thing it learned.

The third test needs care, and there is now a method for it. Liu and colleagues build a provenance-tracing framework that classifies each sentence a distilled model emits by comparing the predictive probabilities assigned to it by the teacher, the original student, and the distilled model under identical context. Actions only the teacher would have favoured are teacher-originated. Actions the original student already favoured are that student regressing to its prior. Run the classification on test-time contexts rather than training contexts and you get a direct read on whether the new behaviour is a transfer or an impression. Their finding is that teacher-originated actions do appear at test time and do explain a meaningful share of distilled performance. The transfer is partial, not zero. That is precisely why one aggregated score is a bad instrument.

```python
def transfer_report(student, base, teacher, suites):
    out = {}
    for skill, (seen, unseen) in suites.items():
        out[skill] = {
            "seen":     acc(student, seen)   - acc(base, seen),
            "reskin":   acc(student, reskin(seen)) - acc(base, reskin(seen)),
            "unseen":   acc(student, unseen) - acc(base, unseen),
            "len_ratio": mean_len(student, seen) / mean_len(teacher, seen),
        }
    return out
```

Read the output key by key. Gain on `seen` with no gain on `unseen` is memorisation with better manners. Gain on `seen` that holds on `reskin` is a procedure. A `len_ratio` near 1.0 tells you nothing on its own, because length can be copied without the policy behind it.

## The gains land where you put them

Zhang and colleagues give the locality question its sharpest recent number. Selecting 1,000 examples from a 100K teacher-generated corpus by targeting the student's weak skills, then fine-tuning with explicit skill decomposition, beats random fine-tuning by 1.6 percentage points on Qwen3-4B and 1.4 on Qwen3-8B across five mathematical benchmarks.

The headline is that data efficiency works. The sentence that matters here is the next one: the gains concentrate on the skills emphasised during training. That is the diagonal of a transfer matrix and little else. It is good engineering, and it is also a warning. If the student's improvement is bounded by the skills you chose to show it, the student has not acquired a procedure that generalises past the curriculum. You have built an efficient lookup for a fixed skill set, which is often exactly what you want and never quite what the word "reasoning" implies.

This is where most production distillation actually lands. Not a reasoning model. A model that is very good at twelve problems you measured and unremarkable on the thirteenth, with fluent commentary over both.

## What the token budget buys instead

Boizard and colleagues make the comparison everyone skips. Reasoning traces run 5 to 20 times longer than standard instruction-tuning outputs. That length is not free: every token spent training on a trace is a token not spent training a larger model on ordinary outputs. At matched FLOPs, across students from 0.5B to 14B and 18 benchmarks, instruction fine-tuning sits on or near the Pareto frontier in the majority of configurations. Reasoning distillation reaches that frontier only on open-ended tasks at 7B and above. Even there, a curriculum mixing 25 to 50 percent reasoning data with ordinary instruction data captures most of the accuracy benefit at much lower cost.

That reframes the decision. Trace distillation is a premium option with a specific payoff profile rather than a default that cheaper methods have to beat. The profile is narrow: open-ended generation, mid-size students, and a budget that can absorb five to twenty times the tokens. If your task is closed-form and your student is small, the trace is mostly a stylistic prior bought at a markup.

## What I would actually do

Distil when you can name the skill you are buying and the test that shows it arrived. Otherwise you are paying for a voice.

Pick the intermediate representation with the smallest surface you can defend. Tags over prose when the domain allows it, prose only when the reasoning genuinely needs unconstrained articulation. Select data by teacher-student divergence rather than by heuristic length or difficulty, since divergence points at the examples the student cannot yet explain. Report reskin accuracy next to in-distribution accuracy, in the same table, every time. Report token length as a function of problem difficulty rather than as a mean, because the mean hides the collapse completely: a model that spends the same length on everything has the right average and no policy.

And treat negative transfer as a live outcome rather than an edge case. It showed up across 14 models, and the mechanism that produces it, imitating a trace whose cost structure you cannot see, is not exotic. It is the default.

## Sources

- [arXiv:2601.05019](https://arxiv.org/abs/2601.05019) - Yueqing Hu et al., 2026. *Hán Dān Xué Bù (Mimicry) or Qīng Chū Yú Lán (Mastery)? A Cognitive Perspective on Reasoning Distillation in Large Language Models.* Reports the functional alignment collapse: teacher token-spend correlation $\bar{r} = 0.64$ against human difficulty, distilled students at $\bar{r} = 0.34$, across 14 models.
- [arXiv:2512.20908](https://arxiv.org/abs/2512.20908) - Kaiyuan Liu et al., 2025. *Where Did This Sentence Come From? Tracing Provenance in LLM Reasoning Distillation.* Introduces cross-model provenance tracing and a teacher-guided data selection criterion based on teacher-student divergence.
- [arXiv:2601.10109](https://arxiv.org/abs/2601.10109) - Lechen Zhang et al., 2026. *Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation.* 1,000 selected examples from a 100K corpus, +1.6 on Qwen3-4B and +1.4 on Qwen3-8B over random fine-tuning; gains concentrate on trained skills.
- [arXiv:2601.10987](https://arxiv.org/abs/2601.10987) - Aanand Balasubramanian, 2026. *Reasoning Distillation for Lightweight Automated Program Repair.* Closed-vocabulary symbolic reasoning tags for a CodeT5 student; macro-averaged gains largest on rare bug categories, and reasoning accuracy correlates with but does not determine prediction.
- [arXiv:2509.22193](https://arxiv.org/abs/2509.22193) - Nicolas Boizard et al., 2025. *Scale or Reason? A Compute-Equivalent Analysis of Reasoning Distillation.* Traces 5 to 20 times longer than instruction outputs; at matched FLOPs instruction fine-tuning sits on or near the Pareto frontier in most of 18 benchmarks, students 0.5B to 14B.
