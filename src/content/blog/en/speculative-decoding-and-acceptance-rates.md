---
title: "Speculative Decoding and the Arithmetic of Acceptance"
description: "Speculative decoding trades cheap draft tokens for one verification pass, and the speedup depends on acceptance rate and cost ratio in a way worth calculating."
pubDatetime: "2024-10-05T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Speculative Decoding
  - Inference
categories:
  - Math
---

A single decode step at batch size one is a memory problem dressed as a compute problem. The kernel streams every weight through the SMs to produce one token, so the ratio of arithmetic to bytes is about one FLOP per byte, and an accelerator wants two orders of magnitude more before its math pipelines are the bottleneck. The FLOPs sit idle; HBM is saturated. That gap is why speculative decoding is interesting, and it explains where the technique stops paying.

The pitch is short. A small draft model proposes several tokens, the large target model scores all of them in one forward pass, and the tokens the target agrees with are kept.

Three things are worth separating: the arithmetic of the speedup, the proof that the distribution does not change, and the conditions under which the arithmetic turns negative.

## A draft, then one verification pass

Let the target model define a next-token distribution $p$ and the draft define $q$ over the same vocabulary. In one iteration the draft runs $\gamma$ times autoregressively, appending each sample to its own prefix. The target then evaluates the full prefix plus those $\gamma$ drafted tokens in a single forward pass, producing $\gamma + 1$ conditional distributions $p_1, \dots, p_{\gamma+1}$.

The acceptance test walks left to right. At position $i$, with $x_i$ the drafted token, accept with probability $\min(1, p_i(x_i)/q_i(x_i))$. If the test fails, sample from a corrected distribution, stop, and discard the rest of the draft. If every test passes, sample one extra token from $p_{\gamma+1}$ and keep all $\gamma + 1$.

Two cost terms matter. Let $c$ be the cost of one draft step divided by the cost of one target step. Verification costs approximately one target step, because the extra positions add compute to a kernel that was already waiting on memory. That approximation does most of the work in every speedup number you will read, so it is worth naming as an assumption.

## The speedup is a ratio of two ratios

Assume each drafted token is accepted independently with probability $\alpha$. The number of tokens produced per iteration is $1$ plus the number of consecutive passes before the first failure, so

$$
\mathbb{E}[\text{tokens}] = \sum_{i=0}^{\gamma} \alpha^{i} = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha},
$$

and the iteration costs $1 + \gamma c$ target steps. The speedup is the quotient:

$$
S(\alpha, \gamma, c) = \frac{1 + \alpha + \alpha^{2} + \dots + \alpha^{\gamma}}{1 + \gamma c}.
$$

Two limits matter. At $\alpha = 0$ the numerator is one and the denominator exceeds one, so $S < 1$: you paid for drafts and threw them away. At $\alpha = 1$ the expression becomes $(\gamma + 1)/(1 + \gamma c)$, which is the ceiling no draft model can beat at that cost. Every real system lives between those two points, so every speedup figure rests on two numbers that are usually reported without their measurement conditions.

Differentiating with respect to $\gamma$ gives the stopping condition. With $L = -\ln \alpha$,

$$
\frac{\partial S}{\partial \gamma} = \frac{\alpha^{\gamma+1} L (1 + \gamma c) - c \left(1 + \alpha + \dots + \alpha^{\gamma}\right)}{(1 - \alpha)(1 + \gamma c)^{2}},
$$

so the optimum is where marginal accepted tokens stop beating marginal draft cost, which is neither $\gamma = 1$ nor the largest $\gamma$ your memory allows.

The table below evaluates the formula at $c = 0.05$, a draft that costs a twentieth of a target step per token. These are arithmetic, not measurements: they assume independence and a free verification pass.

| $\alpha$ | $\gamma = 2$ | $\gamma = 4$ | $\gamma = 6$ | $\gamma = 8$ |
| -------- | ------------ | ------------ | ------------ | ------------ |
| 0.5      | 1.59         | 1.61         | 1.53         | 1.43         |
| 0.7      | 1.99         | 2.31         | 2.35         | 2.28         |
| 0.8      | 2.22         | 2.80         | 3.04         | 3.09         |
| 0.9      | 2.46         | 3.41         | 4.01         | 4.38         |

Low acceptance makes the whole thing close to worthless: the $\alpha = 0.5$ row peaks at 1.61x and then decays as the draft gets longer. High acceptance keeps rewarding longer drafts, because each proposal is cheap relative to the target pass it rides on. Double $c$ to $0.1$ and the $\alpha = 0.8$, $\gamma = 4$ entry falls from 2.80 to 2.40, the same loss as dropping $\alpha$ to 0.72.

## Why the distribution survives rejection

The interesting property is that the procedure samples from $p$, not from something close to $p$. The argument is short.

Sample $x \sim q$. Accept it with probability $\min(1, p(x)/q(x))$, which is the standard rejection sampler for a proposal $q$ and a target $p$ when $q$ has full support. On rejection, draw from the residual

$$
r(x) = \frac{\max(0,\, p(x) - q(x))}{\sum_{y} \max(0,\, p(y) - q(y))}.
$$

The probability that the returned token is $x$ is then the accepted branch plus the rejected branch:

$$
P(x) = \min(p(x), q(x)) + Z \, r(x), \qquad Z = 1 - \sum_{y} \min(p(y), q(y)).
$$

The mass that rejection has to cover is exactly the total variation distance, $Z = \sum_{y} \max(0, p(y) - q(y))$, which is the same thing as the normaliser of $r$. Substituting gives $P(x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$. Those two cases cover every value of $x$, so the returned sample is distributed as $p$.

For $\gamma > 1$ the statement is applied position by position, with the conditionals taken on the prefix that was actually accepted. Token $i$ is tested against $p(\cdot \mid \text{prefix} + x_1 \dots x_{i-1})$ and $q(\cdot \mid \text{prefix})$. Each step is exact, so by the chain rule the joint distribution of the returned sequence is the target's. Nothing accumulates, and there is no approximation parameter.

Three conditions hold this together, and each is easy to violate. The models must share a vocabulary, or $p(x)/q(x)$ is undefined. The test must use the distribution you actually sample from, so a temperature or nucleus filter applied afterwards breaks the correspondence. And the residual sampler has to be implemented, not approximated by resampling from $q$ or by taking the target's argmax.

## The contrarian case: acceptance rate is not the objective

The prevailing framing is that a higher acceptance rate is better and that draft model selection is a search for the smallest model that agrees with the target most often. It is incomplete in a way that costs throughput, and the 2024 study by Yan, Agarwal and Venkataraman shows where.

Their reported finding is that draft latency dominates the speedup, and that language modeling quality does not correlate strongly with speculative throughput. If that holds, then ranking draft models by agreement ranks them by the wrong variable. A draft that agrees more often but runs slower can lose, because the numerator grows with $\alpha$ while the denominator grows with $c$. The right draft is the cheap one to run, not the most accurate one.

There is a second gap, and this one is my reading rather than a measured result: $\alpha$ is treated as a constant, but it is not stationary. It depends on the sampling settings and the prompt domain, and within a draft it tends to fall, because each position conditions on tokens the draft itself produced, so any drift from the target's distribution compounds. A single acceptance number without the configuration and the workload is not reproducible.

I would replace the headline number with seconds per accepted token, measured end to end, including the draft's own overhead and its memory residency. Acceptance rate is then a diagnostic, which is what it should have been.

## Where it loses

Verification stops being free as batch size grows. The argument above assumed the target pass costs about one step because the kernel was memory bound. At large batch it is compute bound, the extra $\gamma + 1$ positions cost real FLOPs, and the denominator in $S$ is no longer $1 + \gamma c$. The 2.37x figure in the multimodal paper is single-stream LLaVA; a server at batch sixteen is a different calculation.

Prefill is untouched: the technique buys decode tokens, so a workload of long prompts and short answers gains little.

Memory is the hidden cost. Two sets of weights resident on the same devices, plus a KV cache for the draft, reduces batch capacity exactly when the server is saturated, and that loss never shows up in single-stream latency numbers.

Constrained decoding is the harder case. A grammar restricts the support, so the residual has to be built over the restricted support, or the guarantee is gone.

Heterogeneous workloads break the independence assumption. Acceptance is not uniform across token types. Where the target distribution is flat, which is where numbers, identifiers and rare names live, agreement is harder to get; boilerplate is peaked and accepts readily. An average acceptance rate over a document can hide stretches well below it, and the variance is what users feel.

The multimodal result is a useful check on where the win comes from. Gagrani and coauthors show that a language-only draft model can serve as the draft for LLaVA 7B without processing image tokens at all, reaching a memory-bound speedup of up to 2.37x with a 115M parameter draft trained from scratch. Their compact draft that does include an image adapter shows only marginal gains on captioning while staying comparable elsewhere. The gain came from skipping work, not from matching the draft to the target.

Finally, the mean is not the tail. Drafting inserts a synchronization point and a second model's scheduling into every iteration. Median latency can improve while p99 gets worse, and the tail often decides whether a feature ships.

## What to measure before you ship it

Instrument three quantities: the per-position acceptance rate under your real sampling settings, the cost ratio $c$ on your hardware at your batch size, and the verification latency as a function of $\gamma$. Then compute the curve instead of guessing at it.

```python
def speedup(accept, c):
    """accept: per-position acceptance probabilities p_1..p_gamma."""
    running, expected = 1.0, 1.0
    for p in accept:
        running *= p
        expected += running
    return expected / (1.0 + c * len(accept))

def best_gamma(alpha, c, cap=16):
    return max(range(cap + 1), key=lambda g: speedup([alpha] * g, c))
```

The per-position form matters. Once acceptance decays with position, replacing the list with a single averaged $\alpha$ overstates the long drafts and picks a $\gamma$ you cannot afford. Measure the list, then let the formula choose.

The other thing to measure is the denominator at the batch size you actually serve. The $1$ in it is only accurate while verification is memory bound. Push batch until the target is compute bound and verification costs closer to $\gamma + 1$ steps, which is the regime where drafting stops paying. Same code, same models, same prompt, different answer.

None of this makes speculative decoding fragile. It makes it a calculation, and the calculation has a domain. Exactness is a theorem with three preconditions. The speedup is a quotient with two inputs. Everything else is marketing.

## Sources

- Minghao Yan, Saurabh Agarwal, Shivaram Venkataraman, "Decoding Speculative Decoding", [arXiv:2402.01528](http://arxiv.org/abs/2402.01528v4), 2024. On what drives speculative throughput: draft latency dominates, and language modeling quality does not track it.
- Mukul Gagrani, Raghavv Goel, Wonseok Jeon, Junyoung Park, Mingu Lee, Christopher Lott, "On Speculative Decoding for Multimodal Large Language Models", [arXiv:2404.08856](http://arxiv.org/abs/2404.08856v1), 2024. Language-only drafts for LLaVA 7B, the 2.37x memory-bound result with a 115M parameter draft, and the marginal gain from an image adapter.
