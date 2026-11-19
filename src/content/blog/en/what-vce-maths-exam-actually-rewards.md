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

The problem below is **written by me in that style** - it is not a past VCE question, and I am not reproducing one. I chose the shape because normal distributions and binomial counts are exactly what the probability area of study rewards.

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
3. **Working with a reporting convention.** Four decimal places, two decimal places - the paper is fussy, and the fussiness is deliberate: it forces you to keep precision until the last line rather than rounding mid-calculation.

## The contrarian reading

The exam assumes the model. It hands you "waiting times are normally distributed with mean 18 and standard deviation 4" and asks nothing about whether that is true. In practice, the expensive mistakes in this kind of work happen before the arithmetic: waiting times are right-skewed (a queue cannot produce negative times but can produce very long ones), and "five patients selected at random" quietly assumes independence, which fails whenever a delay cascades through a clinic.

So the paper measures execution, and execution is genuinely worth measuring. But it never asks you to *justify the distribution*, and that is precisely the step where an engineer earns their salary. A student who can compute 0.0668 in thirty seconds and cannot say why the normal model is wrong for a queue has learned the procedure and missed the subject.

There is a second, quieter bias: every part is solvable in isolation. Real analysis is not - choosing which question to ask is most of the work, and a paper that pre-asks it cannot test the choosing.

## If you are preparing

- Write the standardisation out, every time, even when you can do it mentally. It is a mark and it is a check.
- Translate the English into symbols before touching the calculator: "at least" becomes $1 - \Pr(0) - \Pr(1)$, not "$\Pr(2)$".
- Keep four decimal places internally and round only in the final answer.
- Practise the tail convention until positive-$z$-means-above-the-mean is automatic.

## What the reported precision is worth

A four-decimal answer and a two-decimal answer look like statements about a quantity. They are statements about arithmetic. The distance between the two is measurable, and it takes two derivatives.

Part (b) reached $k = 23.13$ from $k = \mu + 1.2816\sigma$. Treat $\mu$ and $\sigma$ as inputs carrying their own uncertainty, as every estimated quantity does, and differentiate:

$$
\frac{\partial k}{\partial \mu} = 1, \qquad \frac{\partial k}{\partial \sigma} = 1.2816.
$$

A one-minute error in the mean moves the 90th percentile by one minute. A one-minute error in the standard deviation moves it by 1.28 minutes, because the quantile multiplier sits in front of $\sigma$ and never divides out. No part of the paper asks for either, and both are one line of calculus.

Apply them. At $\mu = 18.5$ the answer is $23.63$, exactly half a minute higher. At $\sigma = 4.5$ it is $23.77$, up $0.64$. The answer is reported to $0.01$ minutes, so a half-minute error in the mean is fifty times the resolution being claimed.

Part (a) behaves worse because it lives on a tail. There $\partial \Pr(X > 24)/\partial \mu = \phi(1.5)/4 \approx 0.032$ per minute, so half a minute of error shifts the probability by about $0.016$, and exact recomputation gives $0.0846$ against $0.0668$, a swing of about 27 per cent in a figure quoted to four decimal places. The local slope alone predicts $0.083$; curvature supplies the remaining $0.0015$, since the probability is convex in $\mu$.

Relative sensitivity reverses the ranking. Because $k$ is linear in both inputs, the elasticities sum to one: $18/23.13 \approx 0.78$ for the mean and $5.13/23.13 \approx 0.22$ for the spread. A one per cent error in each gives shifts of $0.18$ and $0.051$ minutes, which combine in quadrature to $0.187$ minutes, about $0.8$ per cent of $k$. The spread accounts for 7.5 per cent of that variance.

Errors compound from part to part. The half minute that lifts part (a) to $0.0846$ also raises $p$, and part (c) is steep in $p$: $\mathrm{d}\Pr(\text{at least } 2)/\mathrm{d}p = 20p(1-p)^3 \approx 1.09$ at the original $p$. Recomputed at the higher value, the binomial answer moves from $0.0390$ to $0.0602$, an increase of 54 per cent. Both chains are sound arithmetic and they disagree by more than half.

This is the ordinary apparatus of an uncertainty budget: multiply each input's uncertainty by its sensitivity coefficient, square, add, take the root. The standard guides on measurement uncertainty give those derivatives a name, sensitivity coefficients, and build the combined uncertainty from them in exactly that way. A Methods paper stops one derivative short of the same machinery and never asks for it. It rewards the second decimal place because a marker can check the second decimal place. Whether the figure means anything depends on a number the paper never supplied.

## Sources

- Victorian Curriculum and Assessment Authority, VCE study designs and past examinations - [vcaa.vic.edu.au](https://www.vcaa.vic.edu.au/).
- The problem above is original, written in the style of the study design; numerical answers were checked independently ($\Pr(X>24) = 0.0668$, 90th percentile $= 23.13$, binomial tail $= 0.0390$).
- National Institute of Standards and Technology, Essentials of expressing measurement uncertainty, Physical Measurement Laboratory (1998, last updated 2017) - [physics.nist.gov](https://physics.nist.gov/cuu/Uncertainty/index.html).
