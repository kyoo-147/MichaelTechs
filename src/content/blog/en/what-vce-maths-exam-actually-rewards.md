---
title: "What a VCE Maths Exam Actually Rewards"
description: "A worked problem in the style of Victorian Mathematical Methods, and the part of the paper that no mark scheme covers."
pubDatetime: "2023-06-03T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Exams
  - VCE
  - Probability
categories:
  - Math
---

Victoria's VCE mathematics papers are unusually explicit about what they want. Mathematical Methods and Specialist Mathematics both publish a study design that names every dot point, and the exam is built to hit those dot points in order. That makes the papers a good instrument for one thing: measuring whether you can execute a known procedure under time pressure, with a calculator, without dropping a sign.

The problem below is **written by me in that style** — it is not a past VCE question, and I am not reproducing one. I chose the shape because normal distributions and binomial counts are exactly what the probability area of study rewards.

## The problem

Hospital waiting times $X$ at a clinic are modelled as normally distributed with mean $18$ minutes and standard deviation $4$ minutes.

**(a)** Find $\Pr(X > 24)$, correct to four decimal places.

**(b)** Find the value of $k$ such that $\Pr(X < k) = 0.90$, correct to two decimal places.

**(c)** Five patients are selected at random. Find the probability that at least two of them wait more than $24$ minutes, correct to four decimal places.

## The solution, and where the marks sit

**(a)** Standardise:

$$
z = \frac{24 - 18}{4} = 1.5,
$$

so

$$
\Pr(X > 24) = \Pr(Z > 1.5) = 1 - \Phi(1.5) \approx 0.0668.
$$

Marks here are usually split between *stating the standardisation* and *the correct tail*. The common error is quoting $\Phi(1.5) = 0.9332$ and stopping, which answers the complementary question. A mark scheme will not tell you that you answered a different question; it just removes a mark.

**(b)** Invert the same relation:

$$
k = \mu + z_{0.90}\sigma = 18 + 1.2816 \times 4 \approx 23.13.
$$

The trap is using $z$ for the upper tail. $\Pr(X < k) = 0.90$ means $k$ is *above* the mean, so $z$ is positive. A negative $z$ here produces a plausible-looking number below 18 and loses the method mark.

**(c)** Let $p = 0.0668$ and $n = 5$. Then

$$
\Pr(\text{at least } 2) = 1 - \Pr(0) - \Pr(1),
$$

$$
\Pr(0) = (1-p)^5 \approx 0.7077, \qquad \Pr(1) = 5p(1-p)^4 \approx 0.2533,
$$

$$
\Pr(\text{at least } 2) \approx 1 - 0.7077 - 0.2533 = 0.0390.
$$

The complement is the whole point of part (c). Computing "at least two" directly means five separate terms and five chances to make an arithmetic slip.

## What the paper is really testing

Three things, and none of them is the modelling:

1. **Queuing the right procedure.** Recognise "normal, upper tail" and reach for the standardisation, not a remembered formula.
2. **Reading the exact clause.** "At least two" versus "more than two", "below" versus "at most". In (b), $\Pr(X < k) = 0.90$ pins down the side of the distribution before any arithmetic happens.
3. **Working with a reporting convention.** Four decimal places, two decimal places — the paper is fussy, and the fussiness is deliberate: it forces you to keep precision until the last line rather than rounding mid-calculation.

## The contrarian reading

The exam assumes the model. It hands you "waiting times are normally distributed with mean 18 and standard deviation 4" and asks nothing about whether that is true. In practice, the expensive mistakes in this kind of work happen before the arithmetic: waiting times are right-skewed (a queue cannot produce negative times but can produce very long ones), and "five patients selected at random" quietly assumes independence, which fails whenever a delay cascades through a clinic.

So the paper measures execution, and execution is genuinely worth measuring. But it never asks you to *justify the distribution*, and that is precisely the step where an engineer earns their salary. A student who can compute 0.0668 in thirty seconds and cannot say why the normal model is wrong for a queue has learned the procedure and missed the subject.

There is a second, quieter bias: every part is solvable in isolation. Real analysis is not — choosing which question to ask is most of the work, and a paper that pre-asks it cannot test the choosing.

## If you are preparing

- Write the standardisation out, every time, even when you can do it mentally. It is a mark and it is a check.
- Translate the English into symbols before touching the calculator: "at least" becomes $1 - \Pr(0) - \Pr(1)$, not "$\Pr(2)$".
- Keep four decimal places internally and round only in the final answer.
- Practise the tail convention until positive-$z$-means-above-the-mean is automatic.

## Sources

- Victorian Curriculum and Assessment Authority, VCE study designs and past examinations — [vcaa.vic.edu.au](https://www.vcaa.vic.edu.au/).
- The problem above is original, written in the style of the study design; numerical answers were checked independently ($\Pr(X>24) = 0.0668$, 90th percentile $= 23.13$, binomial tail $= 0.0390$).
