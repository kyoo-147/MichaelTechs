---
title: "SVD Is the Useful Decomposition"
description: "Eigenvalues are taught first because the algebra is tidy, not because they predict anything. For matrices that come out of data, the singular-value spectrum answers the questions worth asking."
pubDatetime: "2023-06-18T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Linear Algebra
  - SVD
  - Numerical Methods
categories:
  - Math
---

Linear algebra courses spend weeks on eigenvalues before the singular value decomposition appears, if it appears at all. For matrices that come out of data, that order is backwards. Eigenvalues answer a narrow question about a narrow class of matrices. The SVD answers the questions I actually have.

## Eigenvalues ask a lot of the matrix

For $A\mathbf{x} = \lambda \mathbf{x}$ to mean "directions the matrix preserves", $A$ must be square. A data matrix with fifty thousand rows and three hundred columns is not square and has no eigenvectors at all.

Even for square matrices the story fails quietly. Eigenvectors can be complex, so a rotation has none you can see. They can be incomplete: $A = \begin{bmatrix}1 & 1\\ 0 & 1\end{bmatrix}$ has eigenvalue $1$ twice but only one eigenvector, so there is no eigenbasis. And they are generally not orthogonal — only normal matrices, those with $A^\top A = AA^\top$, get an orthogonal eigenbasis, and almost nothing that comes out of a trained model is normal.

## The SVD does not

$$
A = U\Sigma V^\top
$$

$U$ and $V$ are orthogonal, $\Sigma$ is diagonal with non-negative entries in decreasing order, and there are no conditions on $A$ beyond being a matrix. Read it as three operations: rotate by $V^\top$, stretch each axis by the matching $\sigma_i$, rotate by $U$. The input is taken to a basis where the map is diagonal, and the output is rotated into whatever basis you want.

The singular values are gains, not eigenvalues. Each $\sigma_i$ is exactly the factor by which $A$ stretches the direction $v_i$, so $\sigma_1 = \|A\|_2$. And the same factorisation sorts the four subspaces: the first $r$ columns of $U$ span the column space, the rest span the left nullspace, and $r$ is the rank. One object, both domains, orthogonal bases throughout.

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(s[0] / s[-1])        # spectral condition number
print((s > 1e-10).sum())   # numerical rank at that threshold
```

## Rank is a number, not a property

Over the reals, rank counts non-zero singular values. In floating point it counts the ones above a threshold, and the threshold is a decision. A matrix whose exact rank is 300 and whose effective rank is 12 is the same object on paper and a different object in practice, and the spectrum is what tells them apart. A geometric tail means a few terms carry almost everything; a flat spectrum means nothing compresses and a low-rank method is the wrong tool however it is tuned.

Truncating at $k$ terms is the best rank-$k$ approximation in Frobenius norm, so discarding the tail is optimal for that objective rather than a heuristic.

## Compression, and its low-rank cousin

Keeping $k$ terms stores $U_k \Sigma_k V_k^\top$ at a cost proportional to $k(m+n)$ instead of $mn$. That is the compression argument, and it is only as good as the spectrum allows.

Low-rank adaptation is the cousin, and the two get conflated. A LoRA-style update writes $W' = W + BA$ with $B \in \mathbb{R}^{m\times r}$, $A \in \mathbb{R}^{r\times n}$ and $r \ll \min(m,n)$. That is not the truncated SVD of anything. The SVD approximates an existing matrix after the fact; an adapter constrains the update during training, so the optimiser never writes outside a rank-$r$ subspace. The trained factors are not the leading singular triplets of the final delta, because gradient descent is not performing an SVD. What they share is an assumption — that the useful part of a big matrix lives in a low-dimensional subspace — not a mechanism.

## The error bound comes from singular values, not eigenvalues

Here is where the textbook ordering costs real time. Someone looks at a weight matrix, computes its spectral radius, sees a value slightly below one, and concludes the layer is stable. For a normal matrix that inference is fine. For everything else it is not.

$$
A = \begin{bmatrix} 1 & 100 \\ 0 & 1 \end{bmatrix}
$$

Both eigenvalues are $1$. Powers do not grow exponentially, so the spectral-radius check passes. But

$$
A^k = \begin{bmatrix} 1 & 100k \\ 0 & 1 \end{bmatrix},
$$

so the norm grows without bound, linearly in depth, and a single application already takes $(0,1)$ to $(100,1)$ — an amplification of about a hundred. The gain is $\sigma_1 \approx 100$, the smallest singular value is about $0.01$, and their ratio is about $10^4$. None of that was visible in the eigenvalues. The matrix is non-normal, its eigenvectors are nearly parallel, and the eigenbasis is a terrible coordinate system in which to reason about growth.

This is the general shape of the mistake. Error amplification, effective rank, compressibility, sensitivity to quantisation are all singular-value facts. The eigenvalue spectrum describes the matrix in a basis the matrix itself chose; for non-normal matrices that basis can be skewed enough to mislead. When I check a weight matrix now I look at the decay of $s$ and at $\sigma_{\max}/\sigma_{\min}$ first, and often I do not look at the eigenvalues at all.

## What I take from this

The SVD is not a harder version of the eigenvalue decomposition. It always exists, it always gives an orthogonal basis, it always orders components by importance, and it always answers "how much does this operator amplify, and along which directions". Eigenvalues answer a narrower question about a smaller class. Courses teach them first because the algebra is tidier, not because they are more useful.

## Sources

- Gilbert Strang, [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (2010).
- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, SIAM — [people.maths.ox.ac.uk/trefethen](https://people.maths.ox.ac.uk/trefethen/text.html).
