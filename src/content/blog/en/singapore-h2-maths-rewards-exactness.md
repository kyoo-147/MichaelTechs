---
title: "Singapore H2 Maths Rewards Exactness. Real Work Rarely Does."
description: "An integration-by-parts problem in the style of the H2 paper, and why 'hence, or otherwise' is the most transferable phrase on the page."
pubDatetime: "2023-06-24T09:00:00.000Z"
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

Singapore's H2 Mathematics paper has a distinctive temperament. It is short, precise, and almost allergic to numerical approximation. Answers are expected in exact form - $\pi$, $e$, $\ln 2$ - and a decimal is usually only acceptable when the question explicitly says so. Where a Victorian paper trains calculator fluency, an H2 paper trains algebraic control.

The problem below is my own, written in that style; it is not a past question.

## The problem

(a) Show that

$$
\int_{1}^{e} \left(\ln x\right)^2 \, dx = e - 2 .
$$

(b) Hence find the exact volume of the solid generated when the region bounded by $y = \ln x$, the $x$-axis, $x = 1$ and $x = e$ is rotated completely about the $x$-axis.

## The solution

(a) Integrate by parts with $u = (\ln x)^2$ and $dv = dx$. Then

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

(b) The volume of revolution about the $x$-axis is

$$
V = \pi\int_1^e \left(\ln x\right)^2 dx = \pi(e - 2).
$$

Part (b) contains no new integration. That is the whole design of the question: part (a) exists so that part (b) is one line.

## A second integral, one power higher

The paper rarely stops at $(\ln x)^2$. The natural follow-up raises the power, and the follow-up is where the reduction structure becomes visible instead of memorised.

Find $\displaystyle\int_1^e x\ln x\,dx$ in exact form.

Integrate by parts with $u = \ln x$ and $dv = x\,dx$, so $du = dx/x$ and $v = x^2/2$:

$$
\int x\ln x\,dx = \frac{x^2}{2}\ln x - \int \frac{x}{2}\,dx = \frac{x^2}{2}\ln x - \frac{x^2}{4} + C .
$$

Evaluate. At $x = e$: $\frac{e^2}{2} - \frac{e^2}{4} = \frac{e^2}{4}$. At $x = 1$: $0 - \frac{1}{4} = -\frac{1}{4}$. The difference is

$$
\int_1^e x\ln x\,dx = \frac{e^2}{4} + \frac{1}{4} = \frac{e^2 + 1}{4} \approx 2.097 .
$$

The interesting part is not the answer but the pattern. Every power of $\ln x$ reduces to the one below it by the same move: integrate by parts with $u = (\ln x)^n$ and $dv$ equal to whatever polynomial sits next to it. That is why $(\ln x)^2$ appears in part (a) of a longer question and $x\ln x$ appears in a later part. The paper is not testing two unrelated integrals; it is testing whether you noticed they are the same recursion.

Two places to lose marks here. First, the sign on the second term: $-\int x/2\,dx$ becomes $-x^2/4$, and a dropped minus turns $e^2/4 + 1/4$ into $e^2/4 - 1/4$. Second, evaluating at $1$: the polynomial terms do not vanish at the lower limit, only the logarithm does. Both errors are arithmetic, and both are invisible unless the antiderivative is written out before the limits are substituted.

## What this paper is testing

The phrase to notice is "hence, or otherwise". It is an instruction about method, not decoration. "Hence" means *use the result you just proved*; "otherwise" gives you permission to find another route (numerical integration, a different substitution) but never relieves you of justifying it.

Three skills are being measured:

- **Integration by parts, twice, without losing a factor.** The classic error is dropping the $2$ from $du = 2\ln x / x$ on the second pass.
- **Evaluating at $1$ without assuming it vanishes.** $\ln 1 = 0$ kills the first two terms but not the $2x$. Students who write "at $x=1$ everything is zero" lose the whole difference.
- **Carrying an exact value forward.** Part (b) must be $\pi(e-2)$, not $\pi \times 0.7183$. An exactness requirement is a quiet way of testing whether you understood that $0.7183$ is a decimal approximation of a specific number.

## The contrarian reading

Exactness is a beautiful discipline and a poor description of practice. Almost every integral that matters in engineering or machine learning is evaluated numerically, and the interesting questions are about *error*: how many digits survive, how the quadrature rule behaves on a singularity, whether the answer changes in float32.

An H2 script will never ask any of that. It asks for a closed form, and closed forms exist for a tiny, carefully chosen family of functions. The result is a graduate who is very good at the twenty integrals that have closed forms and has never been asked what happens to the other ten thousand.

The "hence" structure has the same double edge. On one hand, it teaches a useful habit: notice that a hard problem contains an easier problem you have already solved. On the other hand, it *tells* you that the connection exists. In real work nobody labels the subproblem for you, and finding that $(\ln x)^3$ can be reduced to $(\ln x)^2$ is the actual skill - the paper gives it away and then marks the execution.

## If you are preparing

- Do parts twice; quoting $\int \ln x\,dx$ from memory is fine, but only after you have derived it enough times to trust the sign.
- Never substitute a decimal into a later part unless told to.
- When a question says "hence", look back at the previous result before touching the page again; that is the intended route and usually the fastest.
- Check the boundary substitution explicitly, term by term.

## One power higher, and where the recurrence gives out

The same trick generates a whole family. Put $I_n = \int_1^e (\ln x)^n\,dx$ and integrate by parts once, with $u = (\ln x)^n$ and $dv = dx$:

$$
I_n = \Big[x(\ln x)^n\Big]_1^e - n\int_1^e (\ln x)^{n-1}dx = e - nI_{n-1},
$$

because $\ln e = 1$ and $\ln 1 = 0$ kills the boundary term at the lower limit. With $I_0 = e-1$ the family follows:

| $n$ | exact | decimal |
| --- | --- | --- |
| 0 | $e-1$ | 1.7182818 |
| 1 | $1$ | 1.0000000 |
| 2 | $e-2$ | 0.7182818 |
| 3 | $6-2e$ | 0.5634363 |
| 4 | $9e-24$ | 0.4645365 |
| 5 | $120-44e$ | 0.3955995 |
| 6 | $265e-720$ | 0.3446845 |
| 7 | $5040-1854e$ | 0.3054900 |

Check two of them against the antiderivative, since a recurrence can be wrong and still look plausible. $\int(\ln x)^3dx = x[(\ln x)^3 - 3(\ln x)^2 + 6\ln x - 6]$, which is $-2e$ at $e$ and $-6$ at $1$, giving $I_3 = 6-2e$. And $\int(\ln x)^4dx = x[(\ln x)^4 - 4(\ln x)^3 + 12(\ln x)^2 - 24\ln x + 24]$, which is $9e$ at $e$ and $24$ at $1$, giving $I_4 = 9e-24$. Both agree.

Look again at $n = 7$. The exact answer is $5040 - 1854e$, and both terms are close to five thousand while the value is about $0.305$. The cancellation is already in the algebra, long before any machine sees it.

And it gets worse predictably. $I_n$ is positive and decreasing in $n$, and $nI_{n-1}$ approaches $e$ from below, so each step is a difference of two quantities sliding toward each other. Every step also multiplies an error in $I_{n-1}$ by $n$, so one rounding unit in $I_0$ becomes $n!$ rounding units in $I_n$. With $18! \approx 6.4 \times 10^{15}$ and a double carrying about sixteen digits, the budget is gone around $n = 18$. Carried forward in double precision the recurrence returns $0.15035$ at $n=16$ against a true $0.15146$, $0.16236$ at $n=17$ against $0.14345$, then a negative number at $n=18$. The $n=17$ value is larger than the $n=16$ value, which a decreasing sequence cannot do, so the failure is visible without knowing any answer.

Run it backward instead, $I_{n-1} = (e - I_n)/n$, and each step divides the inherited error by $n$. Seeded with $I_{30} = 0$, which is simply wrong, the backward run still returns $I_2 = e-2$ to fifteen digits. Gautschi's 1961 paper is the standard account of why one direction survives and the other does not.

The closed form was never the problem. What the recurrence shows is narrower: the same exact identity is a good method or a useless one depending on which way you read it.

## The price of not having the closed form

The complaint above has an arithmetic answer on this very integral. Simpson's rule fits a parabola through three points and integrates that instead. Put it on this integral and the arithmetic is short.

With two subintervals, $h = (e-1)/2 = 0.859141$ and the midpoint is 1.859141, where $\ln x = 0.620115$ and $(\ln x)^2 = 0.384542$. Then

$$
S_2 = \frac{0.859141}{3}\left[0 + 4(0.384542) + 1\right] = 0.286380 \times 2.538168 = 0.726881,
$$

which is 0.008599 above $e - 2 = 0.718282$, wrong in the second decimal. Four subintervals give $h = 0.429570$ and weights 1, 4, 2, 4, 1, over the values 0, 0.127716, 0.384542, 0.685566 and 1:

$$
S_4 = 0.143190\left[4(0.127716) + 2(0.384542) + 4(0.685566) + 1\right] = 0.143190 \times 5.022212 = 0.719131 .
$$

The error is now 0.000849, and two decimals are secure. Halving the step divided the error by 10.1; the next halving divides it by 13.2, not by the 16 that a fourth-order rule suggests, because the error is only the leading term of a series:

$$
S_n - I = \frac{(e-1)^5}{180\,n^4}f^{(4)}(\xi), \qquad f^{(4)}(x) = \frac{22 - 12\ln x}{x^4} .
$$

The error term is standard, and Talvila and Wiersma derive it with nothing beyond integration by parts. The ratio is climbing toward 16; the asymptotics are asymptotic, and two halvings are not enough to reach them.

Here is the price. On $[1,e]$ that fourth derivative falls from 22 at $x=1$ to 0.183 at $x=e$, so the bound at $n=2$ is $\frac{14.979}{2880} \times 22 = 0.1144$, thirteen times the error that actually occurred. The bound is pessimistic by construction: $\xi$ is a point, not the worst point. Solving $(22 - 12\ln x)/x^4 = 1.65$ puts $\xi$ at $x = 1.74$, where the fourth derivative has already fallen a long way from 22. Without the closed form you cannot tell 0.008599 from 0.1144; the bound is safe and uninformative.

To reach six decimals the law says the error must fall by 849, so $n$ rises by $849^{1/4} = 5.4$, to about 22 subintervals and 23 evaluations. That is nothing for $\ln x$. When the integrand is a pricing model, it is the whole budget. Each refinement buys $\log_{10} 16 = 1.2$ decimal digits: double the work, one digit and a fifth.

The two methods are not rivals, and the same arithmetic shows why. Because the error carries a factor of 16, the error still in $S_4$ is one fifteenth of the gap between $S_4$ and $S_2$, so subtracting that fifteenth of the gap removes it:

$$
S_4 + \frac{S_4 - S_2}{15} = 0.719131 - 0.000517 = 0.718614,
$$

and the same step on $S_8 = 0.718346$ gives 0.718294. Nine evaluations, four correct decimals, and the grids are nested, so those nine serve all three estimates. The gain over $S_4$ is 2.5 times and not 64, because the observed ratio is 13.2 and not 16: cancelling an error term you have not yet reached leaves the remainder behind.

So the paper's insistence on exact form and the engineer's quadrature are one computation read from opposite ends, with the factor of 16 as the rate of exchange between them.

## Sources

- Singapore Examinations and Assessment Board, GCE A-Level syllabuses and examinations - [seab.gov.sg](https://www.seab.gov.sg/).
- The problem is original; the integral was verified numerically as well as symbolically ($\int_1^e (\ln x)^2 dx = 0.718282\ldots = e-2$).
- Walter Gautschi, Recursive computation of certain integrals, Journal of the ACM 8(1), January 1961, pp. 21-40 - [doi:10.1145/321052.321054](https://doi.org/10.1145/321052.321054).
- Erik Talvila and Matthew Wiersma, Simple derivation of basic quadrature formulas, arXiv:1202.0249, February 2012 - [arxiv.org/abs/1202.0249](https://arxiv.org/abs/1202.0249).
