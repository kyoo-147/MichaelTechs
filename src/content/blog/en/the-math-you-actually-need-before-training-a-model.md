---
title: "The Math You Actually Need Before Training a Model"
description: "A study order chosen by engineering leverage rather than mathematical dependency, what each subject is used for, and what can honestly be deferred."
pubDatetime: "2023-12-16T09:00:00.000Z"
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

**1. Linear algebra, as geometry plus spectrum.** Vectors as directions, matrices as linear maps, and then - immediately - singular values. Not because the SVD is elegant, but because it is the decomposition that answers "how much does this layer amplify, and along which directions". Strang's 18.06 is the standard course and it is good; the only thing I would add is the singular-value habit early rather than late.

**2. Calculus with shapes: Jacobians and the chain rule.** The derivative as the best linear approximation, the Jacobian as its matrix, and the chain rule as Jacobian composition. Parr and Howard's [matrix calculus paper](https://explained.ai/matrix-calculus/) is the most efficient route I know, because it derives the rules instead of listing them. Olah's [calculus on computational graphs](https://colah.github.io/posts/2015-08-Backprop/) supplies the algorithmic picture that makes the rules computable.

**3. Probability, as a language for uncertainty.** Distributions, expectation, conditional probability, and maximum likelihood. The specific payoff is realising that squared error is a Gaussian assumption and cross-entropy is a Bernoulli one, and that the loss is therefore a statement about your noise model. Goodfellow's chapter 3 covers the ground; Murphy's book is there when you need the Bayesian version.

**4. Optimisation, as curvature plus a certificate.** Gradient descent, step size versus curvature, convexity as a certificate rather than a vibe. Boyd's book is the reference for the convex case; the honest framing is that deep learning is non-convex and what transfers is the local intuition, not the guarantee.

**5. Numerical linear algebra, as the difference between the formula and the result.** Floating point, accumulation order, conditioning, and why a mathematically correct computation can produce garbage. Trefethen and Bau is the book that made this concrete for me.

Notice what is not in the top five: measure theory, real analysis, topology, information geometry. They are not useless. They are simply not where the first year of mistakes is.

## What each subject actually buys you at work

- **Linear algebra**: you stop being surprised by shape errors, you can reason about what a layer can and cannot represent, and you know why a low-rank assumption is or is not justified.
- **Jacobians and chain rule**: you can debug a gradient by hand, write a custom backward pass, and understand why reverse mode costs what it costs.
- **Probability**: you can say what your model's number means, and you can tell the difference between a well-ranked model and a calibrated one.
- **Optimisation**: you can diagnose a training run - curvature, step size, conditioning - instead of trying hyperparameters at random.
- **Numerical linear algebra**: you can explain why the same code gives different losses on two machines, and why fp16 broke one layer and not the others.

## What can be deferred without much cost

Real analysis in full rigour, measure-theoretic probability, and abstract algebra are worth having eventually, and they rarely change a decision in the first two years of shipping models. The people who insist otherwise are usually describing the path that worked for them as researchers, not the path that would work for an engineer.

The exception is if you intend to do research on optimisation or generalisation theory. Then the analysis is not optional, it is the job.

## The contrarian reading

The standard curriculum is ordered by mathematical dependency because that is how mathematics is *organised*, not because that is how it is *used*. A dependency order guarantees that every theorem you meet has its prerequisites; it guarantees nothing about whether you will ever encounter the situation the theorem describes.

Engineering leverage runs the other way. The subjects that explain your failures are worth taking first, even when they arrive early with prerequisites you have not finished. You can learn the spectrum of a matrix before you can prove that every vector space has a basis, and you will get more out of the spectrum.

There is a real risk in that advice: skipping foundations produces people who can run a diagnostic and cannot extend it. So the rule I would state is directional rather than absolute - take the applied subject first, then go back for the foundation once you know why you need it. The motivation that comes from having been burned is worth more than the tidiness of a syllabus.

## How to test yourself

Three tests, in increasing order of honesty:

1. **Re-derive.** Close the book and reconstruct the Jacobian of a two-layer network. If you cannot, the calculus is not there yet.
2. **Implement.** Write backpropagation in NumPy and check it against finite differences. If the check disagrees, you have found the exact place your understanding is thin.
3. **Explain.** Explain to someone why `nn.Linear` is not linear, or why a good likelihood does not imply good calibration. If the explanation needs the book open, it is not yours yet.

The third test is the one I trust. Everything I think I know about this subject is something I have had to explain badly first.

## The one sum worth doing before you rent a GPU

Most pre-training surprises are not mathematical. They are accounting errors, and the bill arrives days into a run, at step 3000, as an out-of-memory kill. The arithmetic is short enough to do on paper, and it should happen before you reserve anything.

Take a 3 billion parameter model trained in bf16 with AdamW. The relevant quantity is bytes held per parameter during training, and it is not two:

- bf16 weights: 2 bytes
- bf16 gradients: 2 bytes
- fp32 master weights: 4 bytes
- fp32 first moment of the gradient: 4 bytes
- fp32 second moment of the gradient: 4 bytes

That is 16 bytes per parameter, so 3,000,000,000 × 16 = 48,000,000,000 bytes, which is 48 GB in decimal or 44.7 GiB in the units the allocator reports. The figure people usually carry around for the same model is 6 GB, because 3,000,000,000 parameters at 2 bytes each is the weight tensor on its own. The ratio is 16/2 = 8, and it does not move when you change batch size or sequence length. It is the cost of the optimiser, not the data; the levers are fewer parameters, a cheaper optimiser state, or sharding.

The same arithmetic is why mixed precision is not a memory fix. An all-fp32 run costs 4 bytes each for weights and gradients, then 4 and 4 for the two Adam moments, which is also 16. bf16 buys throughput and bandwidth and a smaller activation footprint; it saves nothing at all on model states.

Put the 48 GB on hardware. An 80 GB card holds the states and leaves 32 GB for activations, temporary buffers and allocator slack. Rajbhandari et al. report out-of-memory failures in extreme cases with over 30 percent of memory still nominally free, because the allocator cannot find one contiguous block, so 32 GB nominal is not 32 GB usable. Sharding across four 40 GB cards gives 12 GB of state per card and 28 GB free on each, which is the shape of a configuration that actually trains.

Then the arithmetic stops being obvious. Sharding halves the state term each time you double the device count and leaves the activation term where it was, unless you shard activations as well. If activations cost 20 GB per device, four cards carry 12 + 20 = 32 GB and eight cards carry 6 + 20 = 26 GB. Doubling the hardware bought 6 GB per device rather than 12, and it cost a full all-gather per step.

Ten minutes of paperwork produces a budget. It says nothing about whether the loss will descend at that batch size, or whether the optimiser will hold past the first few thousand steps. Budget arithmetic tells you what you can afford to attempt. The rest of this piece is about knowing whether the attempt was sound.

## The other sum: what the run costs in days

Memory decides whether the run fits. It says nothing about whether you can stand to wait for it, and the two gates fail independently: a configuration can sit inside 80 GB and still take a quarter.

The quantity to estimate is floating-point operations. For a dense transformer, Kaplan et al. put training compute at roughly 6ND, with N the non-embedding parameter count and D the number of tokens seen. The 6 is not a mystery. The forward pass costs about 2ND, one multiply and one add per parameter per token, and the backward pass costs twice the forward, so 2 + 4 = 6. That is the whole derivation, and worth being able to redo, because every estimate you meet at work is a variation on it.

Take 1.5 billion parameters and 300 billion tokens.

Step 1, the FLOPs. Multiply the three numbers:

6 × 1,500,000,000 × 300,000,000,000

Group the powers of ten: 1.5 × 10^9 times 3 × 10^11 is 4.5 × 10^20, and 6 × 4.5 = 27, so the total is

2.7 × 10^21 FLOPs

Step 2, effective throughput. An A100 is rated 312 TFLOP/s in bf16 at peak, and peak is a fiction. What matters is model FLOPs utilisation, the fraction of peak the kernels sustain. Pick 40 percent and write it down, because the answer is linear in it:

312 × 10^12 × 0.4 = 1.248 × 10^14 FLOP/s

Step 3, divide one by the other:

(2.7 / 1.248) × 10^7 = 2.16 × 10^7 seconds

One device would need 2.16 × 10^7 / 86,400 = 250 GPU-days. That unit is the one to carry around, because it does not change when the cluster does.

Step 4, wall clock. Eight cards: 250 / 8 = 31 days. Sixty-four cards: 250 / 64 = 3.9 days.

Now check it twice, because a number this easy to compute is also easy to compute wrongly.

The first check is linearity. Compute is linear in tokens, so a tenth of the data is a tenth of the days: 30 billion tokens gives 25 GPU-days, not 125. When an estimate does not move that way, a per-step cost has been mistaken for a total.

The second is the step count. 300 billion tokens at 2 million tokens per step is 150,000 steps, and 31 days is 2.7 × 10^6 seconds, so a step takes about 18 seconds. If you land on 144 seconds, you divided single-device seconds by the step count and forgot the other seven cards.

The result is a floor, not a forecast. It counts model FLOPs only: the attention term grows past what 6ND captures once sequences get long, data loading and evaluation sit outside it, and a restart costs every step since the last checkpoint. Sharding divides compute cleanly in the arithmetic, less cleanly in practice once the all-gathers stop hiding behind the matmuls. But the same 1.5 billion parameter model at 300 billion tokens and at 30 billion differs tenfold in days and not at all in the memory budget. One of those is a decision about the model. The other is a decision about your calendar, and it is the one that usually gets made by accident.

## Sources

- Gilbert Strang, [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
- Terence Parr and Jeremy Howard, [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/).
- Christopher Olah, [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/).
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, [Deep Learning, ch. 3](https://www.deeplearningbook.org/contents/prob.html).
- Kevin P. Murphy, [Probabilistic Machine Learning](https://probml.github.io/pml-book/).
- Stephen Boyd and Lieven Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/).
- Lloyd N. Trefethen and David Bau III, [Numerical Linear Algebra](https://people.maths.ox.ac.uk/trefethen/text.html).
- Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He, [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
- Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei, [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
