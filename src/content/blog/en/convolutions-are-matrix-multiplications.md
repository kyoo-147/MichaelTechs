---
title: "Convolutions Are Matrix Multiplications"
description: "A convolution is a linear operator with a Toeplitz, banded matrix; weight sharing, stride, padding and dilation are all properties of that matrix."
pubDatetime: "2023-04-02T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Convolution
  - Linear Algebra
  - Computer Vision
categories:
  - Math
---

Most explanations of convolution begin with a sliding window. A small grid of weights travels across the image, and at each position you multiply the patch by the weights and add. The picture is correct, and it is close to how the operation is coded. What it hides is that a convolution is one of the most familiar objects in linear algebra: a linear map, and therefore a matrix.

I care about the matrix view for a practical reason: the sliding-window picture makes parameter counts sound like a consequence of an implementation choice, while the matrix makes them a consequence of structure. Structure is what you reason about when a layer is too big or too slow.

## A convolution is a linear map

Work in one dimension first. Let $x \in \mathbb{R}^n$ be a signal and $w \in \mathbb{R}^k$ a filter. Valid convolution produces $m = n - k + 1$ outputs:

$$ y_i = \sum_{j=0}^{k-1} w_j\, x_{i+j}. $$

Every $y_i$ is an inner product between $w$ and a window of $x$ starting at index $i$. Nothing in that expression is nonlinear, and the map $x \mapsto y$ respects addition and scaling:

$$ A(\alpha x + \beta z) = \alpha A x + \beta A z. $$

So a convolution is a linear operator, and every linear operator between finite-dimensional spaces is a matrix once a basis is fixed. For signals the basis is the coordinate basis, so the matrix has a simple shape.

Two consequences matter later. Convolutions compose: the convolution of two filters is a filter, so consecutive convolution layers with no nonlinearity between them collapse into one wider kernel. And, away from the boundaries, they commute with shifts, which is what separates them from a general linear map.

## The matrix behind the sliding window

Write the matrix for $n = 6$ and $k = 3$:

$$ A = \begin{pmatrix} w_0 & w_1 & w_2 & 0 & 0 & 0 \\ 0 & w_0 & w_1 & w_2 & 0 & 0 \\ 0 & 0 & w_0 & w_1 & w_2 & 0 \\ 0 & 0 & 0 & w_0 & w_1 & w_2 \end{pmatrix}. $$

The entries are constant along each diagonal, which makes it Toeplitz. The nonzeros live inside a band of width $k$, which makes it banded. The matrix has $mk$ nonzero entries, and every one is one of the $k$ numbers in $w$.

That is where most of the confusion starts. The matrix is large but carries almost no information: $k$ numbers determine all $mk$ nonzeros. Someone counting entries concludes the layer has $mk$ parameters; someone counting the filter concludes the matrix view is an empty formalism. Both look at the same object and pick a different noun.

In two dimensions the matrix becomes block Toeplitz with Toeplitz blocks, one block per input-output channel pair. The structure repeats at a coarser level and the conclusions do not change.

## Weight sharing is a constraint on the matrix

A general linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$ has $mn$ free parameters; a convolution of width $k$ has $k$. Between those counts sits a set of equality constraints, and weight sharing is the name we give them.

Row $i+1$ of $A$ is row $i$ shifted one column right, with the vacated position filled by the next tap. If $S$ denotes the shift, then $AS = SA$ away from the boundary. Translation equivariance is not a separate design goal layered on top of convolution. It is the same statement as "the rows are shifts of one another."

For a layer with $C_{in}$ input channels, $C_{out}$ output channels, and a $k \times k$ kernel acting on an $H \times W$ map:

$$ \#W_{\text{conv}} = C_{out} C_{in} k^2, \qquad \#W_{\text{dense}} = C_{out} C_{in} H^2 W^2. $$

The ratio is $H^2 W^2 / k^2$. That ratio, not any claim about locality, is why a convolutional layer fits in memory at a width where the equivalent dense layer does not. Both compute maps of the same shape; one is restricted to a banded Toeplitz structure and the other is not.

| Quantity | Dense layer on the same map | $k \times k$ convolution |
| --- | --- | --- |
| Free parameters | $C_{out}C_{in}H^2W^2$ | $C_{out}C_{in}k^2$ |
| Weight matrix | dense | banded Toeplitz |
| Multiply-adds | $H^2W^2 C_{in}C_{out}$ | $H'W' C_{in}C_{out}k^2$ |
| What you store | the matrix | the filter |

## Stride, padding and dilation are edits to the matrix

None of these change what kind of object the layer is. They change which rows exist and where the nonzeros sit inside them.

Padding $p$ adds $p$ zeros to each end of the signal. In the matrix that is $p$ zero columns on each side and correspondingly more rows; the filter is untouched. Padding exists because otherwise every layer shrinks the map and the boundary positions get fewer terms.

Stride $s$ keeps every $s$-th row of the unstrided matrix and discards the rest: a downsampling operator composed with a convolution, written as one object.

Dilation $d$ moves the taps apart, so $w_j$ sits at column $i + jd$ rather than $i + j$. The band widens to $d(k-1)+1$ while each row keeps exactly $k$ nonzeros. Banded with holes, still sparse.

The arithmetic consequence is worth stating plainly. A dilated convolution with kernel $k$ performs $k$ multiply-adds per input channel per output position, exactly as an undilated one does. What changes is reach: dilation spreads the same taps over a wider interval rather than adding taps.

Receptive fields fall out of the same picture. The receptive field of an output unit is the set of input coordinates where its row has nonzeros: the support of that row. Stacked layers multiply matrices, so supports compose. For $L$ layers with kernels $k_\ell$ at stride one, the final support spans $\sum_\ell (k_\ell - 1) + 1$ inputs, because supports of contiguous intervals add. With stride greater than one, the accumulated stride multiplies each layer's contribution. That is the cheapest way I know to compute a receptive field by hand.

## im2col: the matrix gets built after all

There is an apparent contradiction here. The matrix is never formed, yet libraries turn convolutions into matrix multiplications. Both are true: the matrix being formed is not $A$.

im2col rearranges the input so that each receptive field becomes a column. The filter bank becomes a matrix with $C_{in}k^2$ columns, the rearranged input has $C_{in}k^2$ rows and one column per output position, and the product is a single GEMM:

$$ \underbrace{W}_{C_{out} \times C_{in}k^2}\; \underbrace{X_{col}}_{C_{in}k^2 \times H'W'}\; =\; \underbrace{Y}_{C_{out} \times H'W'}. $$

The cost is memory: each input element appears in up to $k^2$ columns, so the buffer is roughly $k^2$ times the size of the input. The compute is unchanged, because the GEMM's inner dimension is the patch and duplicates are never multiplied against anything extra. The duplication is paid in bytes, not FLOPs.

```python
import numpy as np

def im2col(x, k, stride=1):
 # x: (C, H, W) -> (C*k*k, n_positions)
 patches = []
 for i in range(0, x.shape[1] - k + 1, stride):
 for j in range(0, x.shape[2] - k + 1, stride):
 patches.append(x[:, i:i + k, j:j + k].reshape(-1))
 return np.stack(patches, axis=1)
```

Hardware likes regular inner products. Once the convolution is a GEMM, the same kernels that serve dense layers serve convolution, and convolution throughput tracks GEMM throughput on the same chip. That is a fact about libraries, and a quiet admission that the operator was a matrix all along.

## The contrarian case: local does not mean non-matrix

The standard framing goes like this. A dense layer connects every input to every output, so it is a matrix multiplication. A convolution connects each output to a handful of neighbours, so it is something else: a local operation, a filter, a pattern matcher. Locality is offered as the reason the two are different kinds of thing.

The first half is right and the second half does not follow. Locality is a statement about where the nonzeros sit. A matrix with few nonzeros is a sparse matrix, and a sparse matrix is still a matrix. What the argument actually demonstrates is that a convolution is not a dense matmul, which nobody claimed. The error is treating "matrix multiplication" as a synonym for "dense matrix multiplication."

Four things follow once you take the identification seriously.

Counting becomes mechanical. Parameters are the number of distinct entries: $C_{out}C_{in}k^2$ for a $k \times k$ filter over $C_{in}$ channels feeding $C_{out}$ outputs. Cost is rows times columns times inner dimension: $H'W' \cdot C_{out} \cdot C_{in}k^2$ multiply-adds. The sliding-window picture gives the same answers if you are careful about which positions share weights, and that bookkeeping is exactly what the matrix does for you.

Composition becomes legible. Two banded matrices multiply into a banded matrix whose bandwidth is the sum of the bandwidths minus one, so stacking convolutions widens support predictably. The sliding window describes one layer; the matrix describes a stack.

Grouped and depthwise convolutions become block structure: block diagonal with one block repeated per channel, or with $g$ distinct blocks. Seen that way, their parameter counts write themselves.

And im2col explains itself. It exists because the GEMM is what hardware is good at, so the operands get reshaped until the structured product can be handed to a dense kernel. The objection that the matrix is never materialized is true and beside the point. Nobody builds the DFT matrix either; the FFT is how you apply it. Forming $A$ would cost quadratic memory and destroy the only reason the layer is cheap. Exploiting structure instead of storing it is what every structured-operator library does.

Where the matrix view needs care is composition across nonlinearities. A ReLU between two convolutions breaks the linearity, so the network as a whole is not a linear map. It is piecewise linear, and each piece, on a fixed activation pattern, is a matrix. That is a real limit, but it is a limit on "the whole network is linear," not on "a convolution is a matrix." The two get conflated constantly.

The correction I would make is small. A convolution is not an operation that escapes linear algebra. It is a linear map with equality constraints on its entries, and nearly everything people value about it follows from that constrained matrix: parameter sharing, translation equivariance, receptive field growth, arithmetic that stays linear in the size of the filter. The sliding window is a good way to picture one layer. The matrix is a better way to reason about a network.

## Sources

- Li, Johnson & Yeung (2023). *Stanford CS231n, Convolutional Networks* - https://cs231n.github.io/convolutional-networks/
- Trefethen & Bau (1997). *Numerical Linear Algebra* - https://people.maths.ox.ac.uk/trefethen/text.html
