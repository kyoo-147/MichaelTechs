---
title: "Distillation From Reasoning Models"
description: "Trace distillation copies the shape of a teacher's solution. Logit distillation copies its uncertainty. The difference shows up in evaluation."
pubDatetime: "2025-06-14T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Distillation
  - Reasoning
categories:
  - Math
---

A team fine-tunes a small model on traces sampled from a large reasoning model. Training loss falls. The student starts opening its answers the way the teacher does: restate the question, name the quantities, then solve. On the same distribution it looks like the reasoning moved across. Re-skin the problems with different numerals and the improvement thins out or disappears.

Trace distillation and logit distillation get discussed as two price points on one curve: pay more, get more. They are different operations that transfer different things, and the difference shows in what the student does when the surface changes.

## What a trace actually carries

A trace is a sequence of tokens. Supervised fine-tuning on it minimises

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}),$$

where $y$ is one sample from the teacher. The objective has no term for what the teacher considered and rejected. A single draw is treated as ground truth with weight one. If the teacher placed sixty percent of its mass on one step and forty on another, the student is trained as though the second never existed.

The consequence is easy to state and easy to forget. Trace loss sharpens the student precisely where the teacher was uncertain, and averaged cross-entropy throws that uncertainty away, keeping only the sample.

One caveat changes the framing. Sample $k$ independent traces per prompt and average the loss, and you get an unbiased estimate of cross-entropy:

$$\mathbb{E}_{y \sim p_T}\left[-\log p_\theta(y)\right] = H(p_T) + D_{\mathrm{KL}}(p_T \,\|\, p_\theta).$$

The entropy term does not depend on $\theta$, so with enough samples per prompt, trace fine-tuning recovers the forward KL. The gap between style and distribution is partly a sample-count question. The 2023 symbolic distillation work found that sampling many chains per instance was a major factor in whether a small student benefited at all. Most pipelines sample one.

## Distilling the distribution instead

Logit distillation matches the whole next-token distribution at every position rather than one draw from it.

The direction of the divergence is a design decision, not a detail. Forward KL, $D_{\mathrm{KL}}(p_T \,\|\, p_\theta)$, punishes the student for assigning low probability where the teacher assigns mass, so it spreads probability across the teacher's tail. For a classifier that is fine. For a generator it is a bug: the student puts mass on tokens the teacher would rarely emit, and sampling finds them. Gu, Dong, Wei and Huang argued for the reverse direction, $D_{\mathrm{KL}}(p_\theta \,\|\, p_T)$, which penalises student mass where the teacher has none. It is mode-seeking, and it does not reward covering the teacher's noise floor.

Reverse KL needs samples from the student rather than from the data, which makes the estimator on-policy:

$$\nabla_\theta D_{\mathrm{KL}}(p_\theta \,\|\, p_T) = \mathbb{E}_{x \sim p_\theta}\left[\left(\log p_\theta(x) - \log p_T(x)\right)\nabla_\theta \log p_\theta(x)\right].$$

Every student token needs a teacher forward pass over the student's own prefix. That is the price, and it is not small at scale.

Deng and colleagues took the idea further and removed the words. After a teacher is trained on explicit chain of thought, the reasoning is distilled into the student's hidden states, so intermediate computation happens across layers instead of across tokens. Reasoning becomes vertical rather than horizontal. They report that this solves multi-digit multiplication and grade-school math problems the model could not solve without explicit chain of thought, at a speed comparable to no chain of thought. The visible trace was never the only carrier of the computation.

Yu, Xu, Weston and Kulikov came at it from the other side, showing that several System 2 techniques can be compiled back into single-pass generation with better results than the original single-pass model. Reasoning can be absorbed rather than verbalised.

## What each objective costs

| Objective | What it matches | Extra cost | Where it breaks |
| --- | --- | --- | --- |
| One trace per prompt | the sampled path, treated as certain | none beyond SFT | alternatives vanish, style transfers |
| Several traces per prompt | forward KL, approximately | k times the sampling | still teacher-forced prefixes |
| Off-policy logit KD | per-position distribution on fixed prompts | vocabulary-sized tensors | student never visits its own errors |
| On-policy logit KD | reverse KL on student prefixes | teacher pass per student token | wanders into the student's bad states |

Memory bites first in practice. A logits tensor is $B \times T \times V$. One sequence of 2048 positions against a 128k vocabulary in fp32 is $2048 \times 128000 \times 4$ bytes, about 1.05 GB, before activations. Chunking the time axis into 256-token blocks drops the same tensor to roughly 131 MB, which is why implementations do exactly that.

```python
def per_position_kl(student_logits, teacher_logits, chunk=256):
    total, count = 0.0, 0
    for i in range(0, student_logits.size(1), chunk):
        s = F.log_softmax(student_logits[:, i:i + chunk], dim=-1)
        t = F.log_softmax(teacher_logits[:, i:i + chunk].detach(), dim=-1)
        total = total + (t.exp() * (t - s)).sum(-1).mean()
        count += 1
    return total / count
```

That computes teacher-forced forward KL per position. The reverse-KL variant cannot reuse it, because the loop has to run over prefixes the student generated.

## The shortcuts travel with the structure

A trace is not a portable description of how to solve a problem. It records how one model, with one tokenizer, one pretraining corpus and one set of habits, got to an answer. Luo and colleagues tested this and found long chain-of-thought distillation from a strong teacher degrades significantly on nonhomologous models: traces that lift a student from the teacher's own family transfer much less to a different one. Their repair is to segment traces, drop unsolvable and redundant solutions, and fix intermediate error states, a quiet admission that raw traces contain material you should not be imitating.

That is one instance of a general hazard. Every teacher has shortcuts: pattern-matching off a restatement, an answer-shaped guess when the derivation stalls, arithmetic that happens to land, a check step that does not check. Trace loss supervises all of it. It has no notion of which token was load-bearing, so a decisive inference and a decorative preamble get the same gradient weight, scaled by their share of the sequence.

The student also inherits the teacher's compression, not just its procedure. If the shortcut is cheaper than the real derivation, fine-tuning makes the shortcut the path of least resistance, because the gradient goes to the token that was actually emitted. A student trained this way can look better while reasoning worse, and nothing in an answer-level metric will say so.

## The stronger teacher is not automatically the better teacher

Here is where the standard framing breaks. The usual description of distillation is lossy compression: a stronger teacher raises the ceiling, and you lose some fraction on the way down. Under that model, teacher quality is the knob that matters most.

Two 2025 studies found otherwise. Li and colleagues describe a Small Model Learnability Gap: small models at or below 3B parameters do not consistently benefit from long chain of thought or from distillation from larger models, and they do better when fine-tuned on shorter, simpler chains that fit their capacity. Their fix is to mix long and short traces rather than maximise reasoning depth.

Chen and colleagues ran four teacher models and seven student models across seven math and commonsense reasoning datasets and found granularity to be non-monotonic. Stronger students gain from finer-grained reasoning. Weaker students do better with simpler supervision. Format mattered a lot for large models and barely at all for small ones. And stronger teachers did not always produce better students: the diversity and complexity of the supervision could matter more than teacher accuracy.

My reading is that the ceiling is set not by the teacher's accuracy but by the intersection of what the teacher can express and what the student can represent. A trace requiring five quantities in flight across two thousand tokens is a program the student cannot execute; fine-tuning on it teaches the opening and closing lines and skips the middle. Compression is the right metaphor only when both models can represent the object being compressed.

## Evaluation that separates recall from transfer

Most distillation reports show one number: accuracy on a held-out split from the training distribution. That number cannot tell a student that learned to reason from one that learned the shape of the answers.

Four checks separate them.

**Re-skin the inputs.** Keep the structure, change the numerals, names, units and order of clauses. Report both accuracies side by side. A student that generalised holds most of its gain. A student that memorised surface patterns drops.

**Ablate the trace.** Let the student produce a trace, then corrupt middle steps and see whether the answer moves:

$$\Delta_{\mathrm{trace}} = \mathrm{Acc}(\text{faithful trace}) - \mathrm{Acc}(\text{corrupted trace}).$$

If $\Delta_{\mathrm{trace}}$ sits near zero, the trace is decoration. The model is answering from the prompt and the reasoning text is costume. A model that actually computes should get worse when its own intermediate states are broken.

**Split by template, not by row.** Random row splits leak template identity, so the student can learn template to answer-shape and score well. Split by generator or template family and the leak closes.

Compare entropy on shared prefixes. Run teacher and student on the same prefix and look at the student's per-token entropy where the teacher was uncertain. A student far more confident than its teacher, on prefixes both have seen, did not receive the distribution. You got the style with extra confidence, which is the worst combination to deploy.

One more check costs an extra run: train the same student on short answers, no traces. If long traces do not beat that baseline on re-skinned inputs, the traces added length and style, and the accuracy came from somewhere else.

## What I would ship

This section is opinion, not measurement.

Fix the evaluation before the training run, not after. Template-level splits, a re-skinned test set and a trace-ablation number belong in the first experiment, because they decide whether any later result means anything.

If teacher weights are available, per-position distillation is worth the teacher forward pass, with reverse KL over student-generated prefixes rather than a fixed prompt set. If only an API is available, sample several traces per prompt. That is the cheapest approximation to forward KL and it costs nothing but tokens.

Filter traces instead of collecting them. Prefer traces whose intermediate states check out, and drop the ones that reach a right answer through a broken derivation. Mix granularity deliberately, especially below 3B parameters, because long reasoning is not uniformly better supervision.

Watch the opening sentence. Word-for-word reproduction of the teacher's preamble is a style transfer signal and possibly the first sign that traces are being copied rather than used. Run the ablation.

Distillation moves behaviour reliably. Whether it moves the computation is an empirical question, and a single accuracy number is not an answer to it.

## Sources

Yuxian Gu, Li Dong, Furu Wei, Minlie Huang. "MiniLLM: On-Policy Distillation of Large Language Models." [arXiv:2306.08543](https://arxiv.org/abs/2306.08543), 2023. https://arxiv.org/abs/2306.08543

Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, Stuart Shieber. "Implicit Chain of Thought Reasoning via Knowledge Distillation." [arXiv:2311.01460](https://arxiv.org/abs/2311.01460), 2023. https://arxiv.org/abs/2311.01460

Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, Yejin Choi. "Symbolic Chain-of-Thought Distillation: Small Models Can Also Think Step-by-Step." [arXiv:2306.14050](https://arxiv.org/abs/2306.14050), 2023. https://arxiv.org/abs/2306.14050

Ping Yu, Jing Xu, Jason Weston, Ilia Kulikov. "Distilling System 2 into System 1." [arXiv:2407.06023](https://arxiv.org/abs/2407.06023), 2024. https://arxiv.org/abs/2407.06023

Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, Radha Poovendran. "Small Models Struggle to Learn from Strong Reasoners." [arXiv:2502.12143](https://arxiv.org/abs/2502.12143), 2025. https://arxiv.org/abs/2502.12143

Xinghao Chen, Zhijing Sun, Wenjin Guo, Miaoran Zhang, Yanjun Chen, Yirong Sun, Hui Su, Yijie Pan, Dietrich Klakow, Wenjie Li, Xiaoyu Shen. "Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning." [arXiv:2502.18001](https://arxiv.org/abs/2502.18001), 2025. https://arxiv.org/abs/2502.18001

Yijia Luo, Yulin Song, Xingyao Zhang, Jiaheng Liu, Weixun Wang, GengRu Chen, Wenbo Su, Bo Zheng. "Deconstructing Long Chain-of-Thought: A Structured Reasoning Optimization Framework for Long CoT Distillation." [arXiv:2503.16385](https://arxiv.org/abs/2503.16385), 2025. https://arxiv.org/abs/2503.16385
