---
title: "Graph Laplacians and Why GNNs Smooth"
description: "The graph Laplacian explains why graph neural networks smooth node features, why deep propagation collapses them, and why useful depth stays small."
pubDatetime: "2023-07-23T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Graph Theory
  - Laplacian
  - Graph Neural Networks
categories:
  - Math
---

Most introductions to graph neural networks start with message passing and never mention the Laplacian. I think that is backwards. A graph convolution layer, stripped of its learned weights and its nonlinearity, is a fixed linear operator that repeatedly averages a signal over its neighbours. That operator is a polynomial in the graph Laplacian. Once you say that out loud, a lot of GNN behaviour stops looking like an empirical mystery and starts looking like a theorem you could have written down first.

## A graph is a sparse symmetric structure

Take an undirected graph $G = (V, E)$ with $n$ nodes. The adjacency matrix $A$ has $A_{ij} = 1$ when nodes $i$ and $j$ are joined, and $0$ otherwise. For a simple graph $A$ is symmetric with a zero diagonal, and real graphs leave it overwhelmingly sparse: the number of nonzeros is $2|E|$, a tiny fraction of $n^2$ for anything that looks like a social or citation network.

Sparsity is not incidental. It makes multiplication by $A$ cost $O(|E|)$ instead of $O(n^2)$, and it makes the Laplacian a local operator: the matrix only couples nodes one hop apart, so any statement about it has a clean spatial reading.

The degree matrix $D$ is diagonal with $D_{ii} = \sum_j A_{ij}$. For weighted graphs $A_{ij} = w_{ij} \geq 0$ and the degree is the weighted sum; nothing below changes. The combinatorial Laplacian is

$$
L = D - A.
$$

Symmetric, sparse, and the discrete analogue of the negative Laplacian on a manifold. That analogy is not decorative — it is why the same algebra reappears in spectral clustering, graph partitioning, and the analysis of GNNs.

## The quadratic form is a sum of squared differences

The most useful fact about $L$ is what happens when a vector is sandwiched between two copies of it. Let $x \in \mathbb{R}^n$ be a signal on the nodes. Then

$$
x^\top L x = x^\top D x - x^\top A x
= \sum_i D_{ii} x_i^2 - \sum_{i,j} A_{ij} x_i x_j.
$$

Writing the first term as $\sum_{i,j} A_{ij} x_i^2$ and using symmetry to average the two orderings of the second gives

$$
x^\top L x = \frac{1}{2} \sum_{i,j} A_{ij} (x_i - x_j)^2
= \sum_{\{i,j\} \in E} w_{ij} (x_i - x_j)^2.
$$

Every term is non-negative, so $x^\top L x \geq 0$ for all $x$: the matrix is positive semi-definite. That is the whole proof, and it never touches an eigenvalue.

The reading is immediate. $x^\top L x$ measures how much $x$ varies across edges — a Dirichlet energy. A signal nearly constant on every edge has small energy; one that swings between adjacent nodes has large energy. Minimising this quantity under a constraint is the discrete version of solving Laplace's equation, and it is why the Laplacian appears wherever something is being smoothed on a graph.

## Kernel, spectrum, and normalisation

Positive semi-definiteness puts the eigenvalues $\lambda_1 \leq \dots \leq \lambda_n$ at or above zero. The quadratic form says exactly when the smallest one is zero: when every term in the sum vanishes, meaning $x_i = x_j$ on every edge. So $\ker L$ consists of vectors constant on each connected component. A graph with $c$ components has $\lambda_1 = \dots = \lambda_c = 0$; a connected graph has a one-dimensional kernel spanned by $\mathbf{1}$, since the rows of $L$ sum to zero. The eigenvectors form an orthonormal basis, ordered from slowly varying to rapidly oscillating, and in that basis $L$ is diagonal. That is the precise sense in which $L$ is a frequency decomposition of a graph.

Because $L$ has a null space it is not invertible. Solving anything with it needs a constraint, a pseudoinverse, or a regulariser — the same fact that forces the limit in the last section.

The combinatorial Laplacian is also degree-sensitive: its scale grows with the degrees, so two graphs with the same structure and different degree profiles are not comparable. The normalised Laplacian fixes this:

$$
L_{\text{norm}} = I - D^{-1/2} A D^{-1/2}.
$$

It is symmetric, positive semi-definite by the same averaging argument, and its eigenvalues lie in $[0,2]$ regardless of size or degree scale. Each edge is rescaled by the geometric mean of the endpoint degrees, so a node attached to a hub is not dominated by that hub. In spectral clustering this is the difference between the ratio cut and the normalised cut, and the normalised version avoids the habit of isolating one low-degree node as a cluster. In a GNN the same effect appears on the forward pass: the symmetrically normalised operator keeps representation magnitudes from growing with degree.

The spectrum carries structure. The second eigenvalue of a connected graph, $\lambda_2 > 0$, is the algebraic connectivity — small for graphs close to disconnected, large for expanders — and it bounds how fast diffusion mixes. That bound is the quantitative form of "information travels well here", and it is the number to look at when someone claims a graph is deep in some meaningful sense.

## One propagation step is a smoothing operator

The layer of Kipf and Welling (2017) is written

$$
H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right),
$$

with $\tilde{A} = A + I$ adding self-loops. The linear core, $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, is a weighted averaging operator: each node's new feature is a normalised mix of its own and its neighbours' features. Setting $W = I$ leaves pure averaging.

Use the identity $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} = I - L_{\text{norm}}$ on the self-looped graph. A step with scalar coefficient $\alpha$ is then

$$
(I - \alpha L_{\text{norm}})\, x.
$$

Diagonalised, this multiplies the component along eigenvector $u_k$ by $(1 - \alpha \lambda_k)$. For $\alpha \in [0,1]$ every factor has magnitude at most one, and the factors shrink as $\lambda_k$ grows. That is a low-pass filter: edge-to-edge oscillations are attenuated more than smooth variation. The smoothing is structural, not a side effect of training.

So a graph convolution is not a general-purpose feature transform that happens to work on graphs. At its linear core it is an anisotropic diffusion step, with the diffusion coefficient set by $\alpha$ and the graph's spectrum.

```python
# the operator is local, so it fits in a few lines
L = D - A                      # combinatorial Laplacian
L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt
step = lambda x: x - alpha * (L_norm @ x)   # one propagation, no weights
```

## Stacking steps is repeated smoothing

Apply the operator $k$ times and the factor along $u_k$ becomes $(1 - \alpha\lambda_k)^k$. For any $\lambda_k > 0$ with $|1 - \alpha\lambda_k| < 1$ this decays geometrically to zero. Only the $\lambda_k = 0$ directions survive, and those are the component-constant vectors. Hence

$$
(I - \alpha L)^k x \;\longrightarrow\; P x,
$$

where $P$ projects onto $\ker L$. With enough propagation steps every node's representation becomes identical within its connected component; if the graph is connected, all nodes converge to a single vector, and no downstream classifier can separate them.

Nonlinearity and learned weights do not rescue this. They sit inside each step, so the composition is not literally one matrix power, but each step still contracts toward the low-frequency subspace, and a composition of contractions is a contraction. The nonlinearity bends the trajectory without moving the fixed point. That is oversmoothing, and it follows from the spectral decomposition of an averaging operator before any experiment is run.

## The standard framing is wrong: depth does not buy range

Now the contrarian part. The usual story says a $k$-layer GNN aggregates over a $k$-hop neighbourhood, so more layers means a larger receptive field, so deeper models capture longer-range dependencies. The claim about the receptive field is true. The conclusion is misleading.

A $k$-hop field is not a set of independent signals. Distant influence arrives multiplied by a product of normalised edge weights along each path, and meanwhile every node is being averaged with everything reachable. The number of paths of length $k$ between two nodes grows exponentially for most graphs, and the propagation sums over all of them, weighted. The "longer range" information therefore arrives diluted by exactly the averaging that produced it. You are not gathering more structure; you are gathering the same structure more times.

The spectral picture makes the failure explicit. Useful content sits in eigenvectors with moderate eigenvalues: the low ones are component-constant directions carrying no discriminating information, the high ones are oscillations the step suppresses. A deep stack pushes mass toward the bottom of the spectrum, which is the part you cannot use. Depth is a bandwidth control, and it reduces the bandwidth you need.

This predicts several things before a model is fitted. Accuracy should peak at a small number of propagation steps and then decline, not plateau. The decline should come with representations becoming more similar across nodes — shrinking effective rank, rising pairwise cosine similarity — rather than with an optimisation failure. The effect should be stronger where the spectral gap is large relative to graph size, because non-constant modes decay faster there. And a wide shallow model should beat a narrow deep stack of equal parameter count, since width does not consume the spectrum. All four follow from $(1-\alpha\lambda)^k$, and none needs a new dataset to check.

The right response is not to add depth and hope residuals, jumping-knowledge connections, or per-layer readouts repair it. Those are sound engineering, and they concede the point: if depth bought range, you would not need to tap intermediate layers to recover accuracy. The honest response is to decide what range you need, then either choose a propagation operator whose passband covers it, or encode the long-range dependency in the input features through something other than propagation. Laplacian eigenvectors, personalised PageRank vectors, or domain features that already aggregate the relevant neighbourhood are all legitimate — and each is an admission that the propagation step is a filter whose band you are choosing on purpose.

That is the argument for doing the linear algebra first. Once the layer is written as $I - \alpha L$ and diagonalised, the depth question has an answer independent of dataset, optimiser, and seed. You know the operator's response before you train it, and the experiments confirm what you already knew instead of sending you hunting for a bug in the code.

## Sources

- Thomas Kipf and Max Welling, 2017. Semi-Supervised Classification with Graph Convolutional Networks. https://arxiv.org/abs/1609.02907
- Lloyd N. Trefethen and David Bau III, 1997. Numerical Linear Algebra. https://people.maths.ox.ac.uk/trefethen/text.html
