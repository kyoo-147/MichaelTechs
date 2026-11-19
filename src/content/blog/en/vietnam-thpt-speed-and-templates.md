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

## Sources

- Vietnam Ministry of Education and Training, examination regulations and subject structure - [moet.gov.vn](https://moet.gov.vn/).
- The problem is original and its numbers were checked independently: the tangent points are $(-2,0)$ and $(2,4)$, the distance is $4\sqrt{2}$, and the tangent at $x = 2$ is $y = 9x - 14$.
