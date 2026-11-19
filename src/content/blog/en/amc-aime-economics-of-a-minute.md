---
title: "AMC and AIME Reward the Economics of a Minute"
description: "A number-theory problem in competition style, and why answer-getting is a different skill from proving."
pubDatetime: "2023-08-05T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Exams
  - Number Theory
  - Competition
categories:
  - Math
---

American competition mathematics has two gears. The AMC is twenty-five multiple-choice questions in seventy-five minutes; the AIME is fifteen integer-answer questions in three hours. One rewards speed and elimination, the other rewards a clean exact computation. Both reward something that school papers rarely measure: deciding quickly whether you can solve a problem at all.

The problem below is **mine**, written in AMC style. It is not a past question.

## The problem

For how many positive integers $n \le 100$ is $n^2 + n + 1$ divisible by $7$?

## The solution

The key move is that the expression depends on $n$ only through its residue modulo $7$, so there are seven cases and no more. Compute them:

| $n \bmod 7$ | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $n^2+n+1 \bmod 7$ | $1$ | $3$ | $0$ | $6$ | $0$ | $3$ | $1$ |

Two residues work: $n \equiv 2$ and $n \equiv 4 \pmod 7$.

Now count how often each occurs up to $100$. The numbers congruent to $2$ are $2, 9, 16, \ldots, 100$ — that is $15$ values, since $100 = 7 \cdot 14 + 2$. The numbers congruent to $4$ are $4, 11, \ldots, 95$ — that is $14$ values. Total:

$$
15 + 14 = 29 .
$$

A useful sanity check: $7 \mid (n^2+n+1)$ is equivalent to $7 \mid (2n+1)^2 + 3$, and the answer must be roughly $2/7$ of $100$, which is about $28.6$. Getting 29 is consistent; getting 15 would tell you that you had found one residue class and stopped.

## What the contest is actually measuring

The arithmetic here is trivial. What is being tested is whether you notice, within about ninety seconds, that:

1. the question is periodic, so the search space collapses to seven cases;
2. counting residues is a separate step from finding them;
3. the answer's magnitude is predictable, so you can check yourself without re-deriving.

That third habit is the one I would keep. Competition problems are designed so that a wrong method produces a *plausible* wrong answer, and the only cheap defence is having an order-of-magnitude expectation before you commit.

## The contrarian reading

Multiple choice is an answer-getting format, and answer-getting is not proving. On the AMC you can substitute the five options into the expression and take whichever one satisfies the condition — a strategy that is entirely legitimate, requires no number theory, and would score full marks. The paper cannot distinguish a student who saw the modular structure from one who tested options fast.

The AIME fixes that to a degree: an integer answer cannot be guessed from a list. But it introduces a different distortion. Because the answer is always a non-negative integer below 1000, contestants learn to work backwards — assume the answer has a clean form, then force the computation to produce it. That is a genuinely useful heuristic in a timed contest and a genuinely bad habit in research, where the answer is usually not an integer you can anticipate.

So both papers measure something real — fluency, pattern recognition, composure under a clock — and neither measures the thing that matters most later: whether you can state precisely what you are assuming, and prove that the assumption holds.

## A note on preparation

- Convert every "for how many" question into a residue or symmetry argument before doing arithmetic.
- Count the classes you found, then count the total — the error is almost always in the counting step, not the algebra.
- For AIME, keep a running estimate of the answer's size; if your number is 3 or 900, re-read the question before continuing.

## Sources

- Mathematical Association of America, AMC and AIME competitions (the MAA site blocks automated fetches, so this is cited by name rather than linked).
- Background on the competition structure: [American Mathematics Competitions](https://en.wikipedia.org/wiki/American_Mathematics_Competitions), Wikipedia.
- [AP Calculus AB course description](https://apstudents.collegeboard.org/courses/ap-calculus-ab), College Board — useful as a contrast: the curriculum-style AP paper tests technique without the timing pressure.
- The problem and its solution are original; the count was verified by brute force over $n = 1 \ldots 100$ (29 values).
