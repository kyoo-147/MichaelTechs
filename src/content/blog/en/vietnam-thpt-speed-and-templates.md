---
title: "Vietnam's THPT Exam: Speed, Templates, and the One Clause That Costs Marks"
description: "A calculus problem in the style of the national graduation exam, and why template recognition is both the skill and the trap."
pubDatetime: "2023-07-15T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Exams
  - Calculus
  - Vietnam
categories:
  - Math
---

The Vietnamese THPT graduation exam is a speed instrument. Ninety minutes, dozens of small questions, almost all of them drawn from a fixed list of templates: survey a function, find extrema, find an asymptote, compute an integral, solve a complex equation, handle a plane in $\mathbb{R}^3$. A student who has drilled the templates does not *solve* the paper so much as recognise it.

The problem below is **mine**, written in that style. It is not a past question, and I am not reproducing one.

## The problem

Let $f(x) = x^3 - 3x + 2$ and let $d$ be the line $y = 9x$.

**(a)** Find every point on the graph of $f$ at which the tangent is parallel to $d$.

**(b)** Hence compute the distance between those points.

**(c)** Write the equation of the tangent to the graph at the point with $x = 2$.

## The solution

**(a)** Tangents are parallel when the slopes agree:

$$
f'(x) = 3x^2 - 3 = 9 \;\Longrightarrow\; x^2 = 4 \;\Longrightarrow\; x = \pm 2 .
$$

Now compute the corresponding $y$-values:

$$
f(-2) = -8 + 6 + 2 = 0, \qquad f(2) = 8 - 6 + 2 = 4 .
$$

The points are $(-2, 0)$ and $(2, 4)$.

**(b)**

$$
\text{distance} = \sqrt{(2 - (-2))^2 + (4 - 0)^2} = \sqrt{16 + 16} = 4\sqrt{2} \approx 5.657 .
$$

**(c)** At $x = 2$ the point is $(2, 4)$ and the slope is $9$, so

$$
y - 4 = 9(x - 2) \;\Longrightarrow\; y = 9x - 14 .
$$

Notice how little thinking parts (a)-(c) require once the template "tangent parallel ⇒ $f'(x) = m$" is recognised. That recognition *is* the exam.

## Where students actually lose marks

Not in the derivative. In three small places, all of them clause-reading:

1. **"Parallel", not "equal".** The tangent being parallel to $y = 9x$ means equal slopes. If the question said "perpendicular", the right-hand side becomes $-1/9$ and the whole quadratic changes. Candidates who pattern-match on "$y = mx + b$" without reading the relational word lose everything downstream.
2. **Both points, not one.** $\pm 2$ produces two points, and part (b) needs both. Answering $x = 2$ only is the single most common way to turn a two-point problem into a zero.
3. **The point, not the slope, in (c).** Writing $y = 9x + c$ and forgetting to solve for $c$ using $(2, 4)$ costs the final mark.

That third error is instructive. The formula $y = f'(x_0)(x - x_0) + f(x_0)$ is memorised by nearly everyone; the substitution is what gets skipped under time pressure. Speed is precisely what makes the omission likely, which is why the exam is a test of discipline as much as knowledge.

## The contrarian reading

This format trains a real skill and hides a real gap. The skill is fast retrieval: seeing "tangent" and producing the derivative condition in under five seconds is genuinely useful, and the same reflex serves you well when debugging a numerical loss surface.

The gap is that every question has a known template, and the paper never asks a student to *choose* a representation. In practice the hard part of applied mathematics is exactly that choice: whether to model a system as a recurrence or a differential equation, whether to linearise or solve exactly, whether the quantity you want is a value or a distribution. A template exam cannot test that, because a template exam has a syllabus and the syllabus pre-commits the choice for you.

There is a second cost: the exam's standard forms are all clean. $x^2 = 4$ gives nice integers. Real problems give $x^2 = 4.13$ and a domain restriction, and the candidate trained only on the clean case has never had to decide whether an approximate root is acceptable.

## If you are preparing

- Read the relational word first (parallel, perpendicular, "cut at two distinct points", "has a maximum at"), then choose the equation. Most errors are lexical, not algebraic.
- Write the general tangent form and substitute the point explicitly; do not shortcut.
- When a part says "hence", expect the previous result to be the input, and expect no new computation.
- Time yourself on templates until recognition is immediate, then spend the saved time on the clause.

## The external point: counting is the answer

The tangent template has a second form that drills usually skip: tangents drawn through a point that is *not* on the graph. Keep $f(x) = x^3 - 3x + 2$ and take the external point $P(2, 0)$. The tangent at $x_0$ is

$$
y = (3x_0^2 - 3)(x - x_0) + (x_0^3 - 3x_0 + 2),
$$

so requiring it to pass through $P$ means substituting $(2, 0)$:

$$
0 = (3x_0^2 - 3)(2 - x_0) + (x_0^3 - 3x_0 + 2).
$$

Expand term by term: $3x_0^2 \cdot 2 = 6x_0^2$, $3x_0^2 \cdot (-x_0) = -3x_0^3$, $(-3) \cdot 2 = -6$, $(-3) \cdot (-x_0) = 3x_0$. Adding the remaining $x_0^3 - 3x_0 + 2$ cancels the linear pair:

$$
-3x_0^3 + 6x_0^2 + 3x_0 - 6 + x_0^3 - 3x_0 + 2 = -2x_0^3 + 6x_0^2 - 4,
$$

and dividing by $-2$ leaves the cubic $x_0^3 - 3x_0^2 + 2 = 0$.

Now the count, which is the whole difficulty. $x_0 = 1$ is a root, since $1 - 3 + 2 = 0$. Synthetic division by $(x_0 - 1)$ on the coefficients $1, -3, 0, 2$ gives $1, -2, -2, 0$, so the quotient is $x_0^2 - 2x_0 - 2$ and

$$
x_0 = \frac{2 \pm \sqrt{4 + 8}}{2} = 1 \pm \sqrt{3}.
$$

A second route checks it without reusing the first. Put $u = x_0 - 1$; then $x_0^3 - 3x_0^2 + 2 = u^3 - 3u$, which factors as $u(u^2 - 3) = 0$, so $u = 0, \pm\sqrt{3}$ and $x_0 = 1, 1 \pm \sqrt{3}$. Both routes give $1$, $2.7321$, $-0.7321$.

With $m = 3x_0^2 - 3$ the slopes are $0$ at $x_0 = 1$, then $9 + 6\sqrt{3} \approx 19.3923$ at $x_0 = 1 + \sqrt{3}$, then $9 - 6\sqrt{3} \approx -1.3923$ at $x_0 = 1 - \sqrt{3}$. All three tangents pass through $(2,0)$: $y = 0$, $y = (9 + 6\sqrt{3})(x - 2)$, and $y = (9 - 6\sqrt{3})(x - 2)$.

Then audit the count, which takes seconds. For $x_0^3 - 3x_0^2 + 0x_0 + 2$ the roots must satisfy sum $= 3$, pairwise sum $= 0$, product $= -2$. Writing $t = \sqrt{3}$: sum $= 1 + (1 + t) + (1 - t) = 3$; pairwise $= [(1 + t) + (1 - t)] + (1 - t^2) = 2 + (-2) = 0$; product $= 1 \cdot (1 + t)(1 - t) = 1 - 3 = -2$. All three hold, so every root is accounted for.

Two things follow that the earlier parts cannot show. First, the sign flip that worked in part (a) is now illegal: those roots were $\pm 2$, symmetric about $0$, while these are symmetric about $1$, and negating $1 + \sqrt{3}$ gives $-1 - \sqrt{3}$, which satisfies nothing. Second, integer coefficients do not imply rational answers. The curve is as clean as the paper's own examples, and two of its three tangent slopes are surds. A candidate who reads $9 + 6\sqrt{3}$ as evidence of a mistake deletes two correct answers, and the template gave no warning, because the template supplies the condition and never the count. The count is a separate obligation.

## Sources

- Vietnam Ministry of Education and Training, examination regulations and subject structure - [moet.gov.vn](https://moet.gov.vn/).
- The problem is original and its numbers were checked independently: the tangent points are $(-2,0)$ and $(2,4)$, the distance is $4\sqrt{2}$, and the tangent at $x = 2$ is $y = 9x - 14$.
- OpenStax, Calculus Volume 1, section 3.1 "Defining the Derivative", for the slope of a tangent line and the derivative - [openstax.org](https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative).
