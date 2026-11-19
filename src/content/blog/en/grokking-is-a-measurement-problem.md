---
title: "Grokking Is a Measurement Problem"
description: "Delayed generalisation looks mysterious when the only instrument is the loss curve. On a small algorithmic task it is a measurable phase transition in a specific circuit."
pubDatetime: "2026-11-15T09:30:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Interpretability
  - Grokking
  - Generalisation
categories:
  - AI
---

A model reaches perfect training accuracy, keeps training, and then — thousands of steps later — its validation accuracy jumps from chance to near perfect. That is grokking, and for a while it was described as a puzzle about optimisation: how can a network improve on held-out data long after it has stopped improving on the data it sees?

The honest answer is that the loss curve was never the right instrument. [Nanda and coauthors](https://arxiv.org/abs/2301.05217) took a small task where the whole computation can be recovered and made the transition measurable.

## The task

Modular addition: given inputs $a, b \in \{0, \dots, p-1\}$ for a prime $p$, predict $c = (a + b) \bmod p$. A small transformer trained on a fraction of all pairs will first memorise the training set, scoring perfectly there and at chance elsewhere. Later, and abruptly, it generalises.

What changes in between is not a gradual refinement of the same solution. It is a change of solution.

## What the circuit turns out to be

The learned algorithm uses discrete Fourier structure. Write $a$ and $b$ as angles $\omega_a = 2\pi a/p$ and $\omega_b = 2\pi b/p$. The model learns to represent each number by a small set of frequencies, and addition becomes phase addition:

$$
\cos(\omega_a + \omega_b) = \cos\omega_a\cos\omega_b - \sin\omega_a\sin\omega_b .
$$

A trigonometric identity that a first-year student knows becomes a circuit: embed each input into a few frequencies, multiply the corresponding sine and cosine components, sum the products, and read the result back through the inverse transform. Nothing in the task says "use Fourier". The circuit is a consequence of what makes $a + b \bmod p$ easy for gradient descent to represent compactly.

That is the part I find genuinely interesting. The model did not learn the algorithm humans usually write. It learned the algorithm that is cheapest in the architecture at hand.

## Progress measures instead of loss curves

The paper's methodological move is more important than the circuit. Instead of watching loss, it defines quantities that track the mechanism: how much of the model's output can be explained by the trigonometric construction, whether the representation is restricted to a sparse set of frequencies, and whether the network is still using memorisation.

Measured that way, grokking is not a gradual generalisation. There is a long plateau where training accuracy is perfect and the circuit is absent, then a comparatively fast transition where the Fourier components appear and the memorising path gives way. Training loss barely moves during the plateau because it cannot: it is already at zero.

That reframing is the useful export. If the metric you are watching is saturated, it cannot tell you that the mechanism underneath is about to change.

## The framing breaks here: don't generalise the mystery

Two overreaches are common and I want to name both.

The first is treating grokking as a general phenomenon of scale. The clean demonstration is a tiny transformer on a synthetic task with a small vocabulary and an exact answer. That is a good place to find a phase transition and a bad place to extrapolate from. Large models are not trained on modular addition, and there is no evidence that the same circuit-level story explains delayed generalisation on natural data. The weights for a real language model do not sit on a $p$-element circle.

The second is treating the Fourier circuit as *the* solution. It is one solution on one task. The paper is careful about this; summaries are not. What generalises is the method — find a task small enough to reverse-engineer, define progress measures for a hypothesised mechanism, and test whether the mechanism's appearance coincides with the behavioural change. The method is transferable. The specific identity is not.

There is also a measurement caveat with teeth: progress measures are defined relative to a hypothesis. If the model used a construction nobody thought to measure, the analysis would show a mechanism appearing where the real explanation is something else. The strength of the result comes from the ground truth — you can check the recovered algorithm by running it — and most interpretability settings have no such ground truth.

## Why it matters outside interpretability

Three practical lessons survive the caveats.

**Saturated metrics hide transitions.** If a metric has hit its ceiling, it is no longer measuring the thing you care about. Watching a held-out metric and a mechanistic proxy tells you more than watching either alone.

**Memorisation and structure are different solutions, not different amounts of the same solution.** Weight decay, data order, and training duration change which one wins. That gives a knob — but only if you can tell which one you are in.

**The learned algorithm is the cheap one, not the intended one.** This generalises well beyond toy tasks. When a model solves a problem in an unexpected way, the explanation is usually that the architecture made that way cheapest, not that it discovered the mathematically elegant route.

## Sources

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt, [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (2023).
