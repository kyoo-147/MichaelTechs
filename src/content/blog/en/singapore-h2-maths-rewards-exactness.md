---
title: "Singapore H2 Maths Rewards Exactness. Real Work Rarely Does."
description: "An integration-by-parts problem in the style of the H2 paper, and why 'hence, or otherwise' is the most transferable phrase on the page."
pubDatetime: "2026-11-18T09:30:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Exams
  - Integration
  - Singapore
categories:
  - Math
---

Singapore's H2 Mathematics paper has a distinctive temperament. It is short, precise, and almost allergic to numerical approximation. Answers are expected in exact form — $\pi$, $e$, $\ln 2$ — and a decimal is usually only acceptable when the question explicitly says so. Where a Victorian paper trains calculator fluency, an H2 paper trains algebraic control.

The problem below is **my own**, written in that style; it is not a past question.

## The problem

**(a)** Show that

$$
\int_{1}^{e} \left(\ln x\right)^2 \, dx = e - 2 .
$$

**(b)** Hence find the exact volume of the solid generated when the region bounded by $y = \ln x$, the $x$-axis, $x = 1$ and $x = e$ is rotated completely about the $x$-axis.

## The solution

**(a)** Integrate by parts with $u = (\ln x)^2$ and $dv = dx$. Then

$$
du = \frac{2\ln x}{x}\,dx, \qquad v = x,
$$

so

$$
\int (\ln x)^2\,dx = x(\ln x)^2 - 2\int \ln x \, dx .
$$

The remaining integral is standard, and worth doing the same way rather than quoting: with $u = \ln x$, $dv = dx$,

$$
\int \ln x\,dx = x\ln x - x + C .
$$

Substituting back,

$$
\int (\ln x)^2\,dx = x(\ln x)^2 - 2x\ln x + 2x + C .
$$

Evaluate at the limits. At $x = e$: $e \cdot 1 - 2e \cdot 1 + 2e = e$. At $x = 1$: $0 - 0 + 2 = 2$. Therefore

$$
\int_1^e (\ln x)^2\,dx = e - 2 \approx 0.7183 .
$$

**(b)** The volume of revolution about the $x$-axis is

$$
V = \pi\int_1^e \left(\ln x\right)^2 dx = \pi(e - 2).
$$

Part (b) contains no new integration. That is the whole design of the question: part (a) exists so that part (b) is one line.

## What this paper is testing

The phrase to notice is "**hence, or otherwise**". It is an instruction about method, not decoration. "Hence" means *use the result you just proved*; "otherwise" gives you permission to find another route (numerical integration, a different substitution) but never relieves you of justifying it.

Three skills are being measured:

- **Integration by parts, twice, without losing a factor.** The classic error is dropping the $2$ from $du = 2\ln x / x$ on the second pass.
- **Evaluating at $1$ without assuming it vanishes.** $\ln 1 = 0$ kills the first two terms but not the $2x$. Students who write "at $x=1$ everything is zero" lose the whole difference.
- **Carrying an exact value forward.** Part (b) must be $\pi(e-2)$, not $\pi \times 0.7183$. An exactness requirement is a quiet way of testing whether you understood that $0.7183$ is a decimal approximation of a specific number.

## The contrarian reading

Exactness is a beautiful discipline and a poor description of practice. Almost every integral that matters in engineering or machine learning is evaluated numerically, and the interesting questions are about *error*: how many digits survive, how the quadrature rule behaves on a singularity, whether the answer changes in float32.

An H2 script will never ask any of that. It asks for a closed form, and closed forms exist for a tiny, carefully chosen family of functions. The result is a graduate who is very good at the twenty integrals that have closed forms and has never been asked what happens to the other ten thousand.

The "hence" structure has the same double edge. On one hand, it teaches a genuinely useful habit: notice that a hard problem contains an easier problem you have already solved. On the other hand, it *tells* you that the connection exists. In real work nobody labels the subproblem for you, and finding that $(\ln x)^3$ can be reduced to $(\ln x)^2$ is the actual skill — the paper gives it away and then marks the execution.

## If you are preparing

- Do parts twice; quoting $\int \ln x\,dx$ from memory is fine, but only after you have derived it enough times to trust the sign.
- Never substitute a decimal into a later part unless told to.
- When a question says "hence", look back at the previous result before touching the page again; that is the intended route and usually the fastest.
- Check the boundary substitution explicitly, term by term.

## Sources

- Singapore Examinations and Assessment Board, GCE A-Level syllabuses and examinations — [seab.gov.sg](https://www.seab.gov.sg/).
- The problem is original; the integral was verified numerically as well as symbolically ($\int_1^e (\ln x)^2 dx = 0.718282\ldots = e-2$).
