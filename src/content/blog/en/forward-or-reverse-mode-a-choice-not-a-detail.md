---
title: "Forward or Reverse Mode: A Choice, Not a Detail"
description: "Automatic differentiation is usually taught as the chain rule with a framework attached. The direction you evaluate it in decides the cost, and reverse mode comes with a memory bill."
pubDatetime: "2023-09-10T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Automatic Differentiation
  - Backpropagation
  - Systems
categories:
  - Math
---

I have watched people learn automatic differentiation in two stages. First they learn that it is "the chain rule, applied by a computer", which is true and almost useless. Then they learn that a framework calls `.backward()` and gradients appear. Somewhere in between, the actual decision gets skipped: which direction the chain rule is evaluated in, and what that direction costs.

That decision is the whole subject.

## Two ways to walk the same graph

A function $f:\mathbb{R}^n \to \mathbb{R}^m$ computed by a program is a composition of primitives, and its derivative is a Jacobian $J \in \mathbb{R}^{m\times n}$. Nobody wants $J$. What the program can produce cheaply is a product of $J$ with a vector, and there are two ways to arrange that product.

Forward mode pushes a tangent vector forward alongside the computation. At every node you carry the derivative of that node with respect to one chosen input direction. A single sweep gives $J\mathbf{v}$ for one $\mathbf{v}\in\mathbb{R}^n$, so getting all of $J$ takes $n$ sweeps.

Reverse mode runs the program forward, keeps what it will need, then walks backward applying transposed local derivatives. A single sweep gives $\mathbf{u}^\top J$ for one $\mathbf{u}\in\mathbb{R}^m$, so getting all of $J$ takes $m$ sweeps.

Baydin, Pearlmutter, Radul and Siskind put the trade in exactly these terms: the cost of a gradient is proportional to the number of inputs in forward mode and to the number of outputs in reverse mode. That is the whole rule of thumb, and it is a statement about shapes, not about deep learning.

| mode | one sweep produces | sweeps for the full Jacobian | what must be kept |
| --- | --- | --- | --- |
| forward | $J\mathbf{v}$ | $n$ (inputs) | one tangent per node |
| reverse | $\mathbf{u}^\top J$ | $m$ (outputs) | activations from the forward pass |

## Why training lands on the reverse side

A classifier maps an image to a scalar loss. Here $n$ is in the millions and $m$ is one. Forward mode would need one sweep per parameter; reverse mode needs one sweep, full stop. This is not because neural networks are special - it is the shape of the problem. Whenever you optimise a scalar objective with respect to many parameters, you are standing in the one corner of the cost table where reverse mode wins by the largest factor.

The same table explains why inference-time Jacobians are awkward. If you want the sensitivity of a $d$-dimensional output to a $d$-dimensional input - the Jacobian of a normalising flow, say - reverse mode gives one row per sweep and forward mode one column, and neither is cheap. You pick whichever dimension is smaller, and sometimes you use both in different parts of one computation.

## The memory bill arrives later

Reverse mode is cheap in arithmetic and expensive in storage. To apply transposed local derivatives on the way back, you need the intermediate values from the way forward. For a deep network that means holding activations for every layer until their gradient has been computed. The requirement grows with depth and batch size, and at some point it becomes the binding constraint rather than compute.

Gradient checkpointing is the standard trade. You store only a subset of activations - often just block boundaries - and recompute the rest during the backward pass. Extra forward compute buys back memory. What interests me is that this is a knob on the same axis as the forward/reverse choice: you are deciding how much of the graph to materialise. Olah's post on computational graphs is the clearest place to see why that matters, because it makes the graph itself the object of study rather than the formula, and fan-out, accumulation and traversal order become visible.

CS231n's notes are where I point people who want the mechanics: local gradients at each node, multiplied along paths, summed at branches. It is a good way to see why reverse mode accumulates instead of duplicating.

## "Backprop is just the chain rule" explains nothing about the bill

The chain rule is an identity. It says the derivative of a composition is the product of derivatives. It does not say in which order to multiply, whether to keep the intermediates, or what shape the answer has. "Backprop is just the chain rule" is true the way "sorting is just comparisons" is true: it names the ingredient and skips the engineering.

Two things get lost.

First, the memory. Reverse mode is not free differentiation. It is differentiation plus a cache, and every framework that implements it is betting that you would rather spend RAM than FLOPs. That bet is usually right and occasionally wrong - long sequences, large batches, deep stacks - and then you go looking for checkpointing, offloading, or a different formulation of the same derivative.

Second, the non-scalar case. Most tutorials define backpropagation on a scalar loss and then talk about "the gradient" as if it were the general object. It is not. For a vector output, reverse mode computes vector-Jacobian products: you supply a cotangent $\mathbf{u}$ and get $\mathbf{u}^\top J$ back. The full Jacobian is $m$ separate VJPs, and autograd systems do not materialise it unless asked. This is why a custom autograd function is defined by its backward rule rather than by a derivative: the primitive is the VJP, not the Jacobian. Once that clicks, the API stops looking arbitrary. A `grad_outputs` argument is not a quirk, it is the actual interface, and the scalar case is the special one where the cotangent happens to be 1.

## What I check in practice

When I want to reason about gradient cost, I ask three questions in order. What shape is the output I need derivatives for? That picks the mode. How much graph sits between the parameters and that output? That sets the activation memory. Is there a cheap recomputation path? That decides whether checkpointing is a knob or a rewrite.

The answers are usually obvious once the questions are stated. The failure mode is not getting them wrong. It is never asking, because the framework hid the choice behind one method call and a sentence about the chain rule.

## The bill, itemised

When the argument stays abstract, I find it worth picking a shape and counting. Take 100 dense blocks of width 1024, batch 4096, fp32 throughout. The weights are $100 \times 1024^2 \approx 1.05 \times 10^8$ parameters, 400 MiB. The gradients are another 400 MiB and Adam's two moments another 800 MiB, so 1,600 MiB of state exists before anything is differentiated.

Now the part reverse mode adds. One activation vector of width 1024 for one example in one block is $1024 \times 4$ bytes, so 4 KiB; 4096 examples make 16 MiB per block, and 100 blocks make 1,600 MiB. The same figure as the weights, gradients and optimizer combined. That match is a property of the batch, not the model: halve the batch and the cache halves while the model does not move. This counts one stored vector per block, where real blocks keep several, so read it as a floor.

Store every tenth block and the checkpoints are $10 \times 16 = 160$ MiB, plus 160 MiB for the segment being replayed, so the activation peak lands near 320 MiB and the step near 1,920 MiB instead of 3,200 MiB. Ten is not a guess. For a uniform chain the checkpoints go every $\sqrt{100} = 10$ blocks, and Chen, Xu, Zhang and Guestrin state the bound as $O(\sqrt{n})$ memory for an $n$ layer network at "the computational cost of an extra forward pass per mini-batch".

The compute side counts cleanly. A dense block spends one block's worth of multiply-adds going forward and about two going back, one for the input gradient and one for the weight gradient. Without checkpointing a step is near 3 forward-equivalents, with it near 4, since each segment is replayed once. A third more arithmetic for 40 percent less memory. The two and the three assume dense matmuls, and a block that is mostly elementwise work shifts them. Chen and coauthors report 30 percent extra running time on a 1,000 layer residual net that drops from 48 GB to 7 GB.

If the step already fits, this is pure loss and checkpointing buys nothing. The case for it is narrower. The activation peak, not the parameter count, is what stops the batch from growing, so a shape that fits at 1024 can stop fitting at 4096 without one parameter changing.

One correction to the rule of thumb. A forward sweep carries about two forward passes of arithmetic and a reverse sweep about three, so forward mode wins while $2n < 3m$. The crossover sits at $n \approx 1.5m$, not at $n = m$. Which makes reverse mode the right default for scalar objectives, a narrower claim than the one it usually arrives with.

## Sources

- Atilim Gunes Baydin, Barak Pearlmutter, Alexey Radul, Jeffrey Siskind, [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) (2015).
- Christopher Olah, [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) (2015).
- Stanford CS231n, [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/).
- Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin, [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (2016).
