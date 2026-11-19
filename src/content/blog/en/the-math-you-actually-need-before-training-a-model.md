---
title: "The Math You Actually Need Before Training a Model"
description: "A study order chosen by engineering leverage rather than mathematical dependency, what each subject is used for, and what can honestly be deferred."
pubDatetime: "2026-11-17T09:30:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Learning Path
  - Foundations
  - Curriculum
categories:
  - Math
  - AI
---

Most "mathematics for machine learning" curricula are ordered by mathematical dependency: linear algebra, then calculus, then probability, then optimisation. That order is correct if the goal is to be able to read a textbook in sequence. It is a poor order if the goal is to stop making the mistakes you are currently making.

This is the map I would hand someone who already ships models and wants the mathematics that pays for itself first.

## Order by failure, not by dependency

I would start from the failures that are expensive and common:

| failure | subject that explains it | what it changes |
| --- | --- | --- |
| Training diverges late | Conditioning, precision | Which layer, not which LR |
| Loss looks fine, probabilities do not | Probability, calibration | Whether the number can be trusted |
| Gradients vanish or explode | Jacobians, chain rule | Where depth breaks |
| Model compresses badly | Singular values, effective rank | Whether low-rank is viable |
| Optimiser changes everything | Curvature, conditioning | Whether the problem is scale or shape |

Every row points at a subject, and no row starts with "prove that a vector space has a basis". That is the difference between a syllabus and a diagnostic table.

## The five subjects, in the order I would take them

**1. Linear algebra, as geometry plus spectrum.** Vectors as directions, matrices as linear maps, and then — immediately — singular values. Not because the SVD is elegant, but because it is the decomposition that answers "how much does this layer amplify, and along which directions". Strang's 18.06 is the standard course and it is good; the only thing I would add is the singular-value habit early rather than late.

**2. Calculus with shapes: Jacobians and the chain rule.** The derivative as the best linear approximation, the Jacobian as its matrix, and the chain rule as Jacobian composition. Parr and Howard's [matrix calculus paper](https://explained.ai/matrix-calculus/) is the most efficient route I know, because it derives the rules instead of listing them. Olah's [calculus on computational graphs](https://colah.github.io/posts/2015-08-Backprop/) supplies the algorithmic picture that makes the rules computable.

**3. Probability, as a language for uncertainty.** Distributions, expectation, conditional probability, and maximum likelihood. The specific payoff is realising that squared error is a Gaussian assumption and cross-entropy is a Bernoulli one, and that the loss is therefore a statement about your noise model. Goodfellow's chapter 3 covers the ground; Murphy's book is there when you need the Bayesian version.

**4. Optimisation, as curvature plus a certificate.** Gradient descent, step size versus curvature, convexity as a certificate rather than a vibe. Boyd's book is the reference for the convex case; the honest framing is that deep learning is non-convex and what transfers is the local intuition, not the guarantee.

**5. Numerical linear algebra, as the difference between the formula and the result.** Floating point, accumulation order, conditioning, and why a mathematically correct computation can produce garbage. Trefethen and Bau is the book that made this concrete for me.

Notice what is not in the top five: measure theory, real analysis, topology, information geometry. They are not useless. They are simply not where the first year of mistakes is.

## What each subject actually buys you at work

- **Linear algebra**: you stop being surprised by shape errors, you can reason about what a layer can and cannot represent, and you know why a low-rank assumption is or is not justified.
- **Jacobians and chain rule**: you can debug a gradient by hand, write a custom backward pass, and understand why reverse mode costs what it costs.
- **Probability**: you can say what your model's number means, and you can tell the difference between a well-ranked model and a calibrated one.
- **Optimisation**: you can diagnose a training run — curvature, step size, conditioning — instead of trying hyperparameters at random.
- **Numerical linear algebra**: you can explain why the same code gives different losses on two machines, and why fp16 broke one layer and not the others.

## What can be deferred without much cost

Real analysis in full rigour, measure-theoretic probability, and abstract algebra are worth having eventually, and they rarely change a decision in the first two years of shipping models. The people who insist otherwise are usually describing the path that worked for them as researchers, not the path that would work for an engineer.

The exception is if you intend to do research on optimisation or generalisation theory. Then the analysis is not optional, it is the job.

## The contrarian reading

The standard curriculum is ordered by mathematical dependency because that is how mathematics is *organised*, not because that is how it is *used*. A dependency order guarantees that every theorem you meet has its prerequisites; it guarantees nothing about whether you will ever encounter the situation the theorem describes.

Engineering leverage runs the other way. The subjects that explain your failures are worth taking first, even when they arrive early with prerequisites you have not finished. You can learn the spectrum of a matrix before you can prove that every vector space has a basis, and you will get more out of the spectrum.

There is a real risk in that advice: skipping foundations produces people who can run a diagnostic and cannot extend it. So the rule I would state is directional rather than absolute — take the applied subject first, then go back for the foundation once you know why you need it. The motivation that comes from having been burned is worth more than the tidiness of a syllabus.

## How to test yourself

Three tests, in increasing order of honesty:

1. **Re-derive.** Close the book and reconstruct the Jacobian of a two-layer network. If you cannot, the calculus is not there yet.
2. **Implement.** Write backpropagation in NumPy and check it against finite differences. If the check disagrees, you have found the exact place your understanding is thin.
3. **Explain.** Explain to someone why `nn.Linear` is not linear, or why a good likelihood does not imply good calibration. If the explanation needs the book open, it is not yours yet.

The third test is the one I trust. Everything I think I know about this subject is something I have had to explain badly first.

## Sources

- Gilbert Strang, [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
- Terence Parr and Jeremy Howard, [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/).
- Christopher Olah, [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/).
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, [Deep Learning, ch. 3](https://www.deeplearningbook.org/contents/prob.html).
- Kevin P. Murphy, [Probabilistic Machine Learning](https://probml.github.io/pml-book/).
- Stephen Boyd and Lieven Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).
- Lloyd N. Trefethen and David Bau III, [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html).
