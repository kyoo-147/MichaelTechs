---
title: "PCA Is Linear. t-SNE Is Not. That Changes What You Can Read Off a Plot."
description: "PCA preserves variance through a linear projection; t-SNE preserves neighbourhoods. Only one of them supports the reading you are doing."
pubDatetime: "2023-12-09T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Dimensionality Reduction
  - PCA
  - t-SNE
categories:
  - Math
---

I have seen the same mistake in review decks over and over. A model's penultimate layer gets projected into two dimensions, a colourful scatter plot goes into a slide, and someone points at two blobs and says "those are far apart, so the representations are different." The plot was probably made with t-SNE. The sentence is not supported by it.

The confusion is not carelessness. It is that two algorithms get called "dimensionality reduction" and treated as interchangeable, when their objectives are not even the same kind of object. PCA answers a question about variance and linear directions. t-SNE answers a question about neighbourhood probabilities. You can read different things off each, and one of them will let you believe things the other will not.

## Two completely different contracts

Suppose your data lives in $\mathbb{R}^D$ and you want a map to $\mathbb{R}^2$. Both methods produce such a map. That is where the similarity ends.

PCA commits to a linear map. It looks for orthonormal directions $u_1, u_2, \dots$ that maximise retained variance, which for the first direction means

$$
u_1 = \arg\max_{\lVert u \rVert = 1} \operatorname{Var}(X u).
$$

The solution is the leading eigenvector of the covariance matrix, and the projected coordinate of a point $x$ is just $u_1^\top x$. Because the map is linear, the coordinate has a direct additive interpretation: it is a weighted sum of the original features, with the weights being the loadings. This is why PCA plots come with loading arrows and why those arrows are readable. Nothing in the plot requires a stochastic optimiser, and running PCA twice gives the same answer up to sign.

t-SNE commits to nothing linear. It defines a probability distribution over pairs in the high-dimensional space, using a Gaussian kernel whose bandwidth is set per point so that the conditional distribution has a specified entropy; that entropy target is what perplexity parameterises. Then it defines a second distribution over pairs in the low-dimensional space using a heavy-tailed Student-$t$ kernel, and it moves the low-dimensional points to minimise the mismatch between the two distributions, measured as a Kullback–Leibler divergence.

That is a different contract. PCA promises: among all linear maps, this one loses the least squared reconstruction error. t-SNE promises: the arrangement of neighbours in two dimensions looks like the arrangement of neighbours in $D$ dimensions, as measured by one specific divergence. Neither promise implies the other, and neither is a promise about the thing people most want to read, which is global geometry.

## What perplexity actually controls

Perplexity is usually described as "roughly the number of effective neighbours." That is accurate enough to be useful and loose enough to mislead. Concretely, for each point the algorithm performs a binary search for a Gaussian bandwidth $\sigma_i$ such that

$$
\operatorname{Perp}(P_i) = 2^{H(P_i)}
$$

where $H$ is the Shannon entropy of the conditional distribution over neighbours. So perplexity is an exponential of entropy, not a count. If you set perplexity to 30, you are not asking for exactly 30 neighbours; you are asking for a distribution whose entropy matches a distribution that spreads mass over about 30 points.

The practical consequence is that small perplexities suppress mid-range structure and emphasise very local cliques, while large perplexities blur the local structure and start pulling in a coarser organisation. Neither setting is the "true" one; the standard advice to try several acknowledges that the parameter is a scale knob and the plot scale-dependent.

Thinking of the two pair distributions as objects being aligned by a divergence explains the failure modes for free. If the high-dimensional neighbourhoods are not well described by a Gaussian with a single bandwidth, the fit is forced and the picture is a compromise. If the low-dimensional kernel cannot represent a certain kind of structure, the optimiser will find whatever low-dimensional arrangement reduces the divergence, whether or not that arrangement means anything to you.

## The asymmetry that ruins distance readings

Here is the part I care most about. The cost function is a KL divergence of the form

$$
C = \sum_{i} \mathrm{KL}(P_i \,\|\, Q_i) = \sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}.
$$

Look at which distribution appears as the weight. The $p_{j|i}$ terms multiply the log ratio. Neighbours that are close in the high-dimensional space — those with substantial $p_{j|i}$ — dominate the cost. Pairs that are far apart in the original space have $p_{j|i}$ near zero, so their contribution to the gradient is negligible no matter where they end up in the embedding.

That asymmetry is deliberate and it is the source of the method's strength. It lets t-SNE ignore the huge number of uninformative far-apart pairs and spend its effort getting local neighbourhoods right. It is also the reason the plot cannot tell you about global arrangement. Two clusters sitting on opposite sides of a t-SNE figure cost almost nothing in the objective. The optimiser had no incentive to place them there and no incentive to move them; the placement is a byproduct of initialisation and the path the optimiser took.

Attraction and repulsion balance locally. The heavy-tailed low-dimensional kernel creates a repulsive force even between moderately distant points, which is what prevents collapse, and the attractive forces only act where $p_{j|i}$ is large. The balance is local. There is no term in the objective that says "cluster A should be twice as far from cluster B as cluster C is."

## The contrarian point: cluster separation is not in the data

The standard framing treats the t-SNE plot as a faithful but fuzzy picture of the data manifold, with the fuzziness being a nuisance you learn to see through. I think that framing is wrong in a way that matters, because two of the quantities people read off the plot are not properties of the data at all.

**Cluster separation.** The empty space between two blobs in a t-SNE figure is not a measured distance. It is the residue of repulsion that exists by construction: the Student-$t$ kernel is heavy-tailed, so all pairs experience some repulsion, and the optimiser's job is only to balance that against local attraction. When two groups of points have small $p_{j|i}$ between them, the objective is nearly indifferent to whether the groups end up adjacent or at opposite edges. Different runs place them differently. Reading significance into the gap is reading an artefact.

**Cluster size.** The area a cluster occupies in the embedding is likewise not a data property. It reflects how many points the optimiser assigned to that region and how much local repulsion they exert on each other, which depends on density in the original space and on perplexity. A tight cluster in the plot may be a tight cluster in the data, or it may be a dense region whose points could not be pushed apart without violating the local neighbourhood constraints. A sprawling cluster may be genuinely diffuse or may be a moderately tight group that the optimiser spread to make room.

Both of these are consequences of the objective, and both are stable enough across reasonable hyperparameters to look like findings. That is the trap. The plot is not lying — it is answering a different question than the one being asked of it.

The honest use of a t-SNE plot is narrower and, I would argue, more useful: check local neighbourhood consistency. Pick points, find their nearest neighbours in the original space, and verify that the same points are near them in the plot. Look for structure that is stable across seeds and across perplexities. Look for points that never co-locate regardless of the run, because those are the interesting failures. Use it as a diagnostic on a representation, not as a map of a landscape.

There is a related and often overlooked point: because the method is stochastic and the objective has many near-equivalent minima, a single seed is a sample, not a measurement. If you want to show a t-SNE figure and claim a structure exists, generate it several times with different initialisations and show what survives. The structures that survive tend to be local; the ones that do not are exactly the global readings people want to make.

## What to use instead, depending on the question

If you want a plot whose axes mean something, PCA is still the tool, with the caveat that the variance it preserves is variance along linear directions, and a nonlinear manifold can have small linear variance and large intrinsic structure. A curve embedded in a plane can be invisible to the first two principal components if it is oriented unfavourably, because the leading directions capture spread, not shape.

For a somewhat better global picture, the usual advice is to initialise the embedding from PCA and check whether the global arrangement is stable, or to compare against a method that explicitly tries to preserve distances. The most reliable practice is not a better algorithm but a better protocol: decide which question you are answering, pick the method whose objective matches that question, and state the parameter settings alongside the figure so the reader knows what scale they are looking at.

None of this makes t-SNE a bad method. It is exceptionally good at what it was designed for, which is revealing local structure in high-dimensional data that linear methods smear out. The problem is downstream, in the reading, and it is a reading problem that better defaults will not fix.

## The practical protocol I use

When a t-SNE figure appears in a paper draft or a review deck, I ask four questions. What was the perplexity, and was more than one tried? How many seeds, and is the claimed structure visible in all of them? Are the claims local — neighbourhoods, grouping — or global — separation, relative size? And is there an independent check, such as a nearest-neighbour overlap score computed directly in the original space, that corroborates the visual impression?

If the claim is global and the answers are thin, the figure is decoration. If the claim is local and survives seeds and perplexities, the figure is evidence. That distinction is the whole discipline, and it costs nothing but a rerun.

The deeper lesson generalises past these two methods. A visualisation is a function from data to pixels with an objective, and every objective privileges some structure and discards the rest. PCA privileges linear variance and discards nonlinearity. t-SNE privileges local neighbourhood probabilities and discards global geometry. Neither discards those things by accident. Knowing which one you are looking at is not pedantry; it is the difference between a plot that supports a claim and a plot that only appears to.

## Sources

- Laurens van der Maaten and Geoffrey Hinton, *Visualizing Data using t-SNE*, 2008. https://jmlr.org/papers/v9/vandermaaten08a.html
- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, 1997. https://people.maths.ox.ac.uk/trefethen/text.html
