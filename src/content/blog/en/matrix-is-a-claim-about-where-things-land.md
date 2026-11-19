---
title: "A Matrix Is a Claim About Where Things Land"
description: "Reading a weight matrix as a set of destinations instead of a grid of numbers, and where that picture quietly stops being true."
pubDatetime: "2026-11-12T08:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Linear Algebra
  - Matrix Calculus
  - Foundations
categories:
  - Math
---

Most people meet matrices as a grid of numbers and matrix multiplication as a rule you follow with your finger: row times column, add, move on. That order is backwards. The grid is a notation. The operation is a statement about motion.

In the third chapter of *Essence of Linear Algebra*, Grant Sanderson puts the useful reading in one line: "you can interpret the columns as the two special vectors where î and ĵ land" ([3Blue1Brown](https://www.3blue1brown.com/lessons/linear-transformations), 2016). Once you accept that, $A\mathbf{x}$ stops being an arithmetic ritual and becomes a sentence: *take $x_1$ of the first column, $x_2$ of the second, and add them up.*

![The same square before and after a matrix is applied; the arrows are the columns of the matrix.](/assets/blog/math-la-columns/figure.svg)

## The definition most courses skip

A map $T:\mathbb{R}^n \to \mathbb{R}^m$ is linear if it preserves addition and scaling:

$$
T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}), \qquad T(c\mathbf{u}) = c\,T(\mathbf{u}).
$$

Those two axioms force $T(\mathbf{0}) = \mathbf{0}$. They also force something stronger and much more useful: because every vector is a weighted sum of basis vectors, $\mathbf{x} = \sum_i x_i \mathbf{e}_i$, linearity means

$$
T(\mathbf{x}) = \sum_i x_i\, T(\mathbf{e}_i).
$$

The images of the basis vectors are the only data you need. Stack them as columns and you have a matrix. Margalit and Rabinoff make this the punchline of their chapter: "every linear transformation is a matrix transformation; we just haven't computed its matrix yet" ([Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/linear-transformations.html), Georgia Tech, 2017).

So a matrix is not an array that happens to multiply vectors. It is the compressed record of a linear map, and the compression is lossless.

## Reading a weight matrix

For a layer computing $\mathbf{y} = W\mathbf{x}$ with $W \in \mathbb{R}^{m \times n}$, the columns are the *directions the layer can produce*. Column $j$ is what happens to the $j$-th input coordinate when everything else is zero. If two columns are nearly parallel, the layer cannot distinguish those two inputs — it maps them into almost the same direction, and no downstream layer can undo that, because the information is already gone.

That gives three questions worth asking about any weight matrix, none of which involve its individual entries:

1. **Rank.** How many independent directions survive? A rank-$k$ matrix in $\mathbb{R}^{m \times n}$ crushes the input onto a $k$-dimensional subspace, no matter what $m$ and $n$ are.
2. **Singular values.** How much does the layer stretch or shrink each direction? A large ratio between the largest and smallest is where gradients and float16 both start to hurt.
3. **What the bias is doing.** See the next section, because this is where the textbook picture and the framework disagree.

## Where the picture breaks: `nn.Linear` is not linear

Here is a detail that took me an embarrassing amount of time to say out loud. Every source above defines linearity the same way, and that definition forces $T(\mathbf{0}) = \mathbf{0}$. A PyTorch `nn.Linear(in_features, out_features)` computes

$$
\mathbf{y} = W\mathbf{x} + \mathbf{b}.
$$

With $\mathbf{b} \neq \mathbf{0}$ this is **affine**, not linear. It does not fix the origin. Stacking affine maps still gives an affine map, which is exactly why a "deep linear network" collapses to a single matrix times input plus a constant — adding depth without a nonlinearity buys you nothing, and the usual explanation ("linear functions compose to linear functions") is slightly wrong in a way that hides the real reason.

The bias is a translation of the image. Geometrically it shifts the whole subspace away from the origin; practically it lets the layer produce a nonzero output for a zero input, which matters for things like logits of a never-seen class. Keeping the distinction costs one word and prevents a whole family of confused conversations.

## Where the picture breaks again: geometry is basis-free, diagnostics are not

The grid-warping picture is beautiful and it is invariant under change of basis. That is the point of it — and also its limitation. Two matrices with the same determinant can behave completely differently, because the determinant measures volume scaling and says nothing about error amplification.

Concretely, the quantity that predicts whether your forward pass survives is the ratio of largest to smallest singular value, $\sigma_{\max}/\sigma_{\min}$. A rotation has determinant 1 and ratio 1: perfectly stable. A near-singular stretch can also have determinant 1 and a ratio of $10^{4}$: it will destroy a float16 activation and blow up the backward pass. Textbooks usually introduce the determinant early and the singular values late, which trains the eye to look at the wrong scalar first.

Olah's essay on manifolds makes the same caveat in topological language: "Each layer stretches and squishes space, but it never cuts, breaks, or folds it" ([colah, 2014](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)). That statement is true for an invertible weight matrix composed with an invertible activation such as $\tanh$ or sigmoid. It is false for ReLU, false for max-pooling, false for dropout, and false for quantisation. Most modern networks live entirely inside the exceptions. The intuition is still worth having, as long as you remember which assumption paid for it.

## What I actually do with this

Nothing exotic. Before debugging a model, I read the matrices:

- check the shapes first, and derive rank from the shapes rather than from the entries;
- look at the singular-value spectrum of the first and last linear layer, not the whole network, since those are where information enters and leaves;
- remember that `Linear` adds a shift, and that "linear layer" in framework vocabulary means "affine layer" in mathematics;
- keep the determinant out of arguments about numerical stability.

The grid picture is a good instrument. It is not a proof, and it is not the whole story — which is the reason to hold both the geometric reading and the algebraic one at the same time instead of picking a favourite.

## Sources

- Grant Sanderson, [Linear transformations and matrices](https://www.3blue1brown.com/lessons/linear-transformations), *Essence of Linear Algebra*, 2016.
- Dan Margalit and Joseph Rabinoff, [Interactive Linear Algebra, §3.3](https://textbooks.math.gatech.edu/ila/linear-transformations.html), Georgia Tech, 2017 (GFDL).
- Gilbert Strang, [18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/), MIT OpenCourseWare, 2010 (CC BY-NC-SA).
- Chris Olah, [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/), 2014.
