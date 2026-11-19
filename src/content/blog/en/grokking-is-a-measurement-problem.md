---
title: "Grokking Is a Measurement Problem"
description: "Delayed generalisation looks mysterious when the only instrument is the loss curve. On a small algorithmic task it is a measurable phase transition in a specific circuit."
pubDatetime: "2023-11-11T09:00:00.000Z"
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

A model reaches perfect training accuracy, keeps training, and then - thousands of steps later - its validation accuracy jumps from chance to near perfect. That is grokking, and for a while it was described as a puzzle about optimisation: how can a network improve on held-out data long after it has stopped improving on the data it sees?

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

The second is treating the Fourier circuit as *the* solution. It is one solution on one task. The paper is careful about this; summaries are not. What generalises is the method - find a task small enough to reverse-engineer, define progress measures for a hypothesised mechanism, and test whether the mechanism's appearance coincides with the behavioural change. The method is transferable. The specific identity is not.

There is also a measurement caveat with teeth: progress measures are defined relative to a hypothesis. If the model used a construction nobody thought to measure, the analysis would show a mechanism appearing where the real explanation is something else. The strength of the result comes from the ground truth - you can check the recovered algorithm by running it - and most interpretability settings have no such ground truth.

## Why it matters outside interpretability

Three practical lessons survive the caveats.

**Saturated metrics hide transitions.** If a metric has hit its ceiling, it is no longer measuring the thing you care about. Watching a held-out metric and a mechanistic proxy tells you more than watching either alone.

**Memorisation and structure are different solutions, not different amounts of the same solution.** Weight decay, data order, and training duration change which one wins. That gives a knob - but only if you can tell which one you are in.

**The learned algorithm is the cheap one, not the intended one.** This generalises well beyond toy tasks. When a model solves a problem in an unexpected way, the explanation is usually that the architecture made that way cheapest, not that it discovered the mathematically elegant route.

## The arithmetic, worked once

Take the paper's mainline prime, $p = 113$, and one pair off the table: $a = 37$, $b = 85$. The answer is $122 \bmod 113 = 9$. The angle step is $\omega = 2\pi/113 \approx 0.055603$ radians, so the inputs sit at $37\omega \approx 2.057326$ and $85\omega \approx 4.726290$.

The four numbers the circuit needs are $\cos(2.057326) = -0.467561$, $\sin(2.057326) = 0.883961$, $\cos(4.726290) = 0.013900$, and $\sin(4.726290) = -0.999903$. Then

$$
(-0.467561)(0.013900) - (0.883961)(-0.999903) = -0.006499 + 0.883875 = 0.877376,
$$

which is $\cos(9\omega)$ to the last digit shown. The two input angles added to $6.783616$ radians, and that is $2\pi + 9\omega$ exactly: one full turn, then the answer. The extra turn is invisible to a periodic function, so the modulo never gets computed. It is absorbed by the representation. A lookup table needs a case for the wrap; a circuit built from sines and cosines has no boundary to handle.

The read-out works the same way. The model ends up using five key frequencies, $k \in \{14, 35, 41, 42, 52\}$, and the unembedding adds $\cos(\omega_k(122 - c))$ over those five for each candidate answer $c$. Give each frequency equal weight and the arithmetic is short: at $c = 9$ every term is $\cos(2\pi k) = 1$ and the score is 5, the ceiling for five frequencies. At $c = 8$ it is $-1.97$. Even the best wrong candidate, $c = 25$, reaches only 2.59. Wrong answers are not merely behind; they are pushed down, which is how five frequencies can separate 113 candidates.

Two counts show the wrap is not a corner case. Of the $113^2 = 12{,}769$ ordered pairs, exactly $6{,}328$ have $a + b \ge 113$: for each $a$ there are $a$ values of $b$ that wrap, so the total is $\sum_{a=1}^{112} a = 112 \cdot 113/2$, or 49.6% of the table. A model that mishandles the boundary loses on half its inputs. And the mainline run trains on 30% of the pairs, roughly 3,831 examples with 8,938 held out. That fraction is not a detail: the [original grokking paper](https://arxiv.org/abs/2201.02177) found the optimisation needed to generalise rises sharply as the training fraction falls, so where a run sits on that curve decides how much of it is spent getting nowhere.

None of this is a new measurement. It is the published setup evaluated by hand, rounding included, and the point of doing it is to see what "the circuit is exact" means: at the answer the five terms sum to their maximum, one step away they sum to a negative number, and nowhere in the network is there a branch that knows where the circle rejoins.

## Sources

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt, [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (2023).
- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra, [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) (2022).
