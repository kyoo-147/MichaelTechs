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

What I find interesting here is the algorithm itself. The model did not learn the algorithm humans usually write. It learned the algorithm that is cheapest in the architecture at hand.

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

The read-out works the same way. The model ends up using five key frequencies, $k \in \{14, 35, 41, 42, 52\}$, and the unembedding adds $\cos(\omega_k(122 - c))$ over those five for each candidate answer $c$. Give each frequency equal weight and the arithmetic is short: at $c = 9$ every term is $\cos(2\pi k) = 1$ and the score is 5, the ceiling for five frequencies. At $c = 8$ it is $-1.97$. Even the best wrong candidate, $c = 25$, reaches only 2.59. Wrong answers are pushed down rather than left behind, which is how five frequencies can separate 113 candidates.

Two counts show the wrap is not a corner case. Of the $113^2 = 12{,}769$ ordered pairs, exactly $6{,}328$ have $a + b \ge 113$: for each $a$ there are $a$ values of $b$ that wrap, so the total is $\sum_{a=1}^{112} a = 112 \cdot 113/2$, or 49.6% of the table. A model that mishandles the boundary loses on half its inputs. And the mainline run trains on 30% of the pairs, roughly 3,831 examples with 8,938 held out. That fraction is not a detail: the [original grokking paper](https://arxiv.org/abs/2201.02177) found the optimisation needed to generalise rises sharply as the training fraction falls, so where a run sits on that curve decides how much of it is spent getting nowhere.

None of this is a new measurement. It is the published setup evaluated by hand, rounding included, and the point of doing it is to see what "the circuit is exact" means: at the answer the five terms sum to their maximum, one step away they sum to a negative number, and nowhere in the network is there a branch that knows where the circle rejoins.

## What a frequency is worth, in bits

A score is not a loss. The five-frequency readout ranks 113 candidates, and a ranking becomes a progress measure only when it goes through a softmax and is counted in bits. Keeping the equal weighting, let a candidate sitting $\delta$ away from the answer score $s(\delta) = \sum_{k} \cos(2\pi k \delta / 113)$, so that the answer is at $\delta = 0$ and scores exactly 5.

The arithmetic is short enough to do by hand. Since $s(0) = 5$, the correct candidate contributes $e^{5} = 148.413$ to the partition sum. The 112 wrong candidates contribute 267.315, so the total is 415.728, and

$$
p(\text{correct}) = \frac{148.413}{415.728} = 0.357, \qquad -\log_2 0.357 = 1.486 \text{ bits}.
$$

The largest wrong terms are the pair at $\delta = \pm 16$, scoring 2.5898 each and contributing $e^{2.5898} = 13.33$ apiece. A wrong answer is not far behind. It is outnumbered. For scale, a uniform guess over 113 candidates costs $\log_2 113 = 6.820$ bits.

Remove frequencies, re-measure, and the price becomes visible. No frequencies costs 6.820 bits, one 5.718, two 4.616, three 3.515, four 2.483, five 1.486. The zero-frequency entry is the useful check: with no frequencies the readout is flat, so the formula has to return exactly $\log_2 113$. The first three frequencies are worth 1.10 bits each, the fourth 1.03, the fifth 1.00, and dropping any single one of the five costs between 0.96 and 1.00. Nothing here is load-bearing. The measure is the sum, and confidence is bought roughly one bit at a time.

One frequency alone shows why. Take $k = 14$ and the wrong candidate at $\delta = 8$: $14 \times 8 = 112$, so the angle between them is $2\pi \cdot 112/113 = 2\pi - 2\pi/113$, and $\cos(2\pi/113) = 0.998455$. The wrong answer scores 99.85% of the right one, and the loss is 5.718 bits against 6.820 for knowing nothing. A single frequency knows the answer is somewhere on the circle. It does not know where.

None of these numbers is a measurement. They are the idealised readout at unit weight, so the absolute bits are a convention and the differences are the content: the shape of the paper's excluded-loss measure applied to a formula instead of to a network. Two knobs move the number. Adding a frequency buys about a bit. Scaling every weight doubles every logit, leaves the ranking untouched and lowers the loss, which is why the circuit that matters is the one reaching a given margin at the lowest parameter norm - and that is the quantity [the circuit-efficiency account](https://arxiv.org/abs/2309.02390) treats as decisive. The generalising circuit is slower to learn but produces larger logits for the same parameter norm, while memorisation, on that account's hypothesis, gets less efficient as the training set grows. There is then a dataset size at which the two are equally efficient, a transition that can run the wrong way - their ungrokking result - and a plateau that becomes the expected shape rather than a mystery.

## Sources

- Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt, [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) (2023).
- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra, [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) (2022).
- Vikrant Varma, Rohin Shah, Zachary Kenton, János Kramár, Ramana Kumar, [Explaining grokking through circuit efficiency](https://arxiv.org/abs/2309.02390) (2023).
