---
title: "Superposition: More Features Than Dimensions"
description: "Why a layer with fewer dimensions than features can still represent most of them, what sparsity has to do with it, and where the toy model stops mapping onto a real network."
pubDatetime: "2026-11-14T09:30:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Interpretability
  - Superposition
  - Features
categories:
  - AI
---

A linear layer with $m$ dimensions can hold at most $m$ mutually orthogonal directions. If I want it to represent $n > m$ independent features, the linear algebra says no. Yet trained networks behave as if they hold far more concepts than they have neurons, and the [Anthropic toy model](https://transformer-circuits.pub/2022/toy_model/index.html) is the cleanest demonstration I know of why that is possible.

## The setup

There are $n$ input features and a hidden layer with $m < n$ units. The model reconstructs its input. Each feature is active only sometimes, and feature $i$ carries an importance weight $I_i$. The loss is importance-weighted reconstruction:

$$
\mathcal{L} = \sum_i I_i \left(x_i - \hat{x}_i\right)^2 .
$$

With dense features the model cannot reconstruct everything, so it represents at most $m$ of them and gives up on the rest. That is the textbook outcome, and it is not superposition. Something changes when the features become sparse.

## Sparsity changes the optimal packing

Suppose the model gives feature $i$ a direction $\mathbf{w}_i$ and reads it back along the same direction. The linear reconstruction of feature $i$ is

$$
\hat{x}_i = \sum_j x_j \langle \mathbf{w}_i, \mathbf{w}_j \rangle .
$$

The $j=i$ term is the signal, $x_i\|\mathbf{w}_i\|^2$. Every other term is interference from a feature that happens to be active at the same time. Orthogonal directions eliminate interference, but only $m$ of them fit. Allow near-orthogonal directions and more than $m$ fit, at the price of crosstalk.

Sparsity makes that price acceptable. If feature $j$ is usually off when feature $i$ is on, the crosstalk is usually zero, even though it would be large in the rare case where both fire. The model optimises expected loss, so it takes a bad worst case in exchange for a good average. That is the mechanism: represent more features than the layer has dimensions, accept a little interference, and rely on features being off most of the time.

## The geometry is a packing problem

The interesting result is what the directions look like. The paper maps a phase diagram over sparsity and the ratio of features to dimensions, and the regimes have recognisable geometries: antipodal pairs, triangles, tetrahedra, pentagons, and a regular simplex when there is exactly one more feature than dimensions. As the feature count grows at high sparsity, the directions spread into a roughly uniform arrangement and interference is smeared across many small overlaps rather than concentrated in a few large ones.

I find the small cases most useful. Three features in two dimensions cannot be mutually orthogonal; the best the model can do is spread them evenly, and a triangle is what that looks like. Four in three gives a tetrahedron. The model is not choosing a favourite subset and discarding the rest — it is equalising the damage.

Importance shifts the picture. When some features matter more, the model protects them and lets the unimportant ones take more interference. That is a sensible trade, and it means the geometry depends on an importance distribution a practitioner rarely knows in advance.

## Why neurons look polysemantic

A neuron is a coordinate axis. If several feature directions have non-zero components along that axis, the neuron responds to several features. In the toy model this falls straight out of the packing: the feature directions are not axis-aligned, so the axes are shared.

That is the part I would hand to anyone puzzled by a neuron that fires for both French text and Python code. The neuron is one coordinate in a basis that several concepts happen to use, not a strange composite concept.

## The framing breaks here: a toy model is a map, not the territory

This is where citing the paper goes wrong. Superposition is often invoked as if it explained polysemanticity in general, and the paper itself is more careful than its readers.

The toy model assumes a linear readout with a specific sparsity distribution and a single hidden layer. Real networks have nonlinear readouts, multiple layers of composition, attention mixing across positions, and sparsity that is neither independent nor identically distributed across features. The clean phase diagram is a property of the toy problem. What transfers is the argument — sparsity plus a capacity limit makes interference worthwhile — not the specific geometry or the phase boundaries.

There is a second, subtler limit. The toy model measures interference against a reconstruction objective. In a real network there is no ground-truth feature list to reconstruct, so "how much interference is acceptable" is not a quantity the training objective states explicitly. Interpretability work infers the features after the fact, which means the interference the model actually tolerates is defined relative to whatever feature basis the analysis assumes.

## How I use it

Superposition is a good prior when reading activation patterns: expect sharing, expect axes to mix concepts, and do not assume one neuron equals one idea. It is a bad prior for predicting how many features a specific layer holds, because the number depends on sparsity and importance in that layer, and neither is measured.

When I want to use the idea seriously, the check is empirical: find directions that are nearly orthogonal, test whether they activate independently across inputs, and see whether ablating one leaves the others intact. That is slow. It is also the difference between using a model of superposition and assuming one.

## Sources

- Anthropic, [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (2022).
