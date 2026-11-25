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

The problem below is written by me in that style - it is not a past VCE question, and I am not reproducing one. I chose the shape because normal distributions and binomial counts are exactly what the probability area of study rewards.

## The problem

Hospital waiting times $X$ at a clinic are modelled as normally distributed with mean $18$ minutes and standard deviation $4$ minutes.

(a) Find $\Pr(X > 24)$, correct to four decimal places.

(b) Find the value of $k$ such that $\Pr(X < k) = 0.90$, correct to two decimal places.

(c) Five patients are selected at random. Find the probability that at least two of them wait more than $24$ minutes, correct to four decimal places.

## The solution, and where the marks sit

(a) Standardise:

$$
z = \frac{24 - 18}{4} = 1.5,
$$

so

$$
\Pr(X > 24) = \Pr(Z > 1.5) = 1 - \Phi(1.5) \approx 0.0668.
$$

Marks here are usually split between *stating the standardisation* and *the correct tail*. The common error is quoting $\Phi(1.5) = 0.9332$ and stopping, which answers the complementary question. A mark scheme will not tell you that you answered a different question; it just removes a mark.

(b) Invert the same relation:

$$
k = \mu + z_{0.90}\sigma = 18 + 1.2816 \times 4 \approx 23.13.
$$

The trap is using $z$ for the upper tail. $\Pr(X < k) = 0.90$ means $k$ is *above* the mean, so $z$ is positive. A negative $z$ here produces a plausible-looking number below 18 and loses the method mark.

(c) Let $p = 0.0668$ and $n = 5$. Then

$$
\Pr(\text{at least } 2) = 1 - \Pr(0) - \Pr(1),
$$

$$
\Pr(0) = (1-p)^5 \approx 0.7077, \qquad \Pr(1) = 5p(1-p)^4 \approx 0.2533,
$$

$$
\Pr(\text{at least } 2) \approx 1 - 0.7077 - 0.2533 = 0.0390.
$$

Part (c) is built on the complement. Computing "at least two" directly means five separate terms and five chances to make an arithmetic slip.

## What the paper is really testing

Three things, and none of them is the modelling:

1. **Queuing the right procedure.** Recognise "normal, upper tail" and reach for the standardisation, not a remembered formula.
2. **Reading the exact clause.** "At least two" versus "more than two", "below" versus "at most". In (b), $\Pr(X < k) = 0.90$ pins down the side of the distribution before any arithmetic happens.
3. **Working with a reporting convention.** Four decimal places, two decimal places - the paper is fussy, and the fussiness is deliberate: it forces you to keep precision until the last line rather than rounding mid-calculation.

## The contrarian reading

The exam assumes the model. It hands you "waiting times are normally distributed with mean 18 and standard deviation 4" and asks nothing about whether that is true. In practice, the expensive mistakes in this kind of work happen before the arithmetic: waiting times are right-skewed (a queue cannot produce negative times but can produce very long ones), and "five patients selected at random" quietly assumes independence, which fails whenever a delay cascades through a clinic.

So the paper measures execution, and execution is worth measuring. But it never asks you to *justify the distribution*, and that is precisely the step where an engineer earns their salary. A student who can compute 0.0668 in thirty seconds and cannot say why the normal model is wrong for a queue has learned the procedure and missed the subject.

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

## What five observations would buy you

Every number in the paper arrives free. The mean is 18 minutes, the standard deviation is 4, and nobody paid for either one. Ask a clinic to produce those two figures from data and the arithmetic changes shape, because the width of an answer starts to depend on how much evidence stands behind it.

Take five observed waits, the sample size part (c) already uses. The standard error of their mean is

$$
\frac{\sigma}{\sqrt{n}} = \frac{4}{\sqrt{5}} = \frac{4}{2.2361} = 1.7889.
$$

If the standard deviation is known exactly, the 95 per cent interval is $18 \pm 1.96 \times 1.7889$, that is $18 \pm 3.5062$, or $(14.49,\ 21.51)$. This is the paper's world, and it is the one a Mathematical Methods paper examines.

If the standard deviation is estimated from the same five waits, the multiplier is not 1.96 but the $t$ quantile on $n - 1 = 4$ degrees of freedom, $2.7764$:

$$
2.7764 \times 1.7889 = 4.9666,
$$

so the interval becomes $(13.03,\ 22.97)$. That is not a rounding difference. Each wall moves $1.46$ minutes and the interval is 41.7 per cent wider, from a convention that scores nothing either way. The five-minute half-width also prices the reported decimals: $4.97$ minutes is roughly five hundred times a resolution of $0.01$ minutes. The $t$ table and this interval are standard first-course material, and OpenStax's introductory statistics text works the single-mean case at section 8.2.

Narrowing the interval is expensive, and the price is quadratic. For a half-width of one minute with $z$:

$$
n = \left(\frac{1.96 \times 4}{1}\right)^2 = 61.47,
$$

so 62 patients. With $t$ the multiplier moves with $n$, which makes the equation implicit rather than closed. Guessing 64 gives $t_{63} = 1.9983$ and $1.9983 \times 4 / 8 = 0.9992$, just inside a minute; 63 gives $1.0074$, just outside. So 64. The two conventions have met to within two patients by the time the sample is in the sixties.

Four times the data is not four times the precision. From five patients to twenty, the standard error falls to $4/\sqrt{20} = 0.8944$ and the half-width to $2.0930 \times 0.8944 = 1.8721$. That is 2.65 times narrower, not four, because the quantile fell as well. Precision is bought in square roots, and one decimal place honestly earned costs about a hundred times the data that bought the one before.

Apply that to the paper's own reporting. A $k$ of $23.13$ claims a resolution of $0.01$ minutes, and

$$
n = \left(\frac{1.96 \times 4}{0.01}\right)^2 = 784^2 = 614{,}656.
$$

Dropping from a one-minute half-width to a one-hundredth multiplies the count by $10^4$, and $62 \times 10^4 = 620{,}000$, within one per cent of the exact figure, so the two routes agree. The reported precision is not a judgement about waiting times. It is a claim about a sample of six hundred thousand patients who were never sampled.

None of this is on the paper, and none of it can be. The paper quotes $0.0668$ and $23.13$ because it was handed a model instead of a sample, and a model carries no standard error. That substitution is made silently, on your behalf, before the first question is read.

## Sources

- Victorian Curriculum and Assessment Authority, VCE study designs and past examinations - [vcaa.vic.edu.au](https://www.vcaa.vic.edu.au/).
- The problem above is original, written in the style of the study design; numerical answers were checked independently ($\Pr(X>24) = 0.0668$, 90th percentile $= 23.13$, binomial tail $= 0.0390$).
- National Institute of Standards and Technology, Essentials of expressing measurement uncertainty, Physical Measurement Laboratory (1998, last updated 2017) - [physics.nist.gov](https://physics.nist.gov/cuu/Uncertainty/index.html).
- OpenStax, Introductory Statistics 2e, section 8.2, A Single Population Mean using the Student t Distribution - [openstax.org](https://openstax.org/books/introductory-statistics-2e/pages/8-2-a-single-population-mean-using-the-student-t-distribution).
