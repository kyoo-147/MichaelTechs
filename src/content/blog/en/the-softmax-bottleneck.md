---
title: "The Softmax Bottleneck"
description: "The output layer factorizes logits as a hidden state times an embedding matrix, which caps the rank of the log-probability matrix at the hidden width."
pubDatetime: "2023-11-05T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Softmax
  - Low Rank
  - Language Models
categories:
  - AI
---

Every language model ends the same way: a vector goes in, a matrix of word embeddings is multiplied against it, and a softmax turns the resulting scores into a distribution over the vocabulary. This last step is where a specific, checkable claim lives. The parametrization of the output layer imposes a rank ceiling, and natural language asks for more rank than that ceiling permits at typical widths.

## The factorization at the end of the network

Let a stack of layers produce a context vector $h(x) \in \mathbb{R}^{d}$ for an input context $x$, and let $W \in \mathbb{R}^{V \times d}$ hold one row $w_y$ per vocabulary item. The logits are $h(x)^\top w_y$. Writing $P_{ij} = \log p(y_j \mid x_i)$ for a batch of $N$ contexts and $H \in \mathbb{R}^{N \times d}$ for the stacked hidden states, the whole output layer collapses to

$$ P = H W^\top - \mathbf{1} z^\top, \qquad z_i = \log \sum_{j=1}^{V} \exp\left(h_i^\top w_j\right). $$

The first term is a product of an $N \times d$ matrix and a $d \times V$ matrix, so its rank is at most $d$. The second term is an outer product of a column of ones with the vector of log-partition values, which has rank one. Rank is subadditive (Trefethen and Bau, 1997), so

$$ \operatorname{rank}(P) \le d + 1. $$

That is the entire observation, and it assumes almost nothing. Nothing about the data, nothing about how $H$ was produced, nothing about training. Any model that computes logits as a linear map of a width-$d$ vector and normalizes with a softmax has a log-probability matrix bounded this way. The "+1" is just the normalizer.

## What the rank bound constrains

A matrix of rank $r$ spans at most $r$ independent directions across its rows. If the conditional distributions $p(\cdot \mid x)$ that a task demands require substantially more than $d+1$ independent directions, then no choice of $H$ and $W$ reproduces them exactly, and the gap does not close with more training.

The argument in Yang et al. (2017) is that natural language next-token distributions across contexts form exactly such a matrix: it is far from low rank, while the softmax family is pinned below a ceiling set by $d$. Two things are worth separating in that statement. First, the claim concerns exact representation of a whole matrix, not just the identity of the argmax. A model can pick the right next token in the great majority of positions while still assigning badly calibrated mass elsewhere. Second, the ceiling bites unevenly. A context whose next token is nearly determined costs almost nothing to represent; one low-rank row does the job. A context that must hedge across many plausible continuations with different relative weights is where a single direction across the vocabulary becomes a poor fit, and it is in those contexts that the loss is concentrated.

The bound is a ceiling on rank, not on expressiveness in general. A matrix of rank $d+1$ can still hold arbitrarily many distinct rows as long as they lie in that span; what it cannot do is produce rows that are mutually independent beyond $d+1$ directions. Whether language needs more is empirical, and the honest reading of the paper is that it does at realistic widths, not that it must.

## Mixtures as a way to raise the rank

The natural repair is to stop letting one linear map own the distribution. A mixture of softmaxes puts a latent variable in front of the output layer:

$$ p(y \mid x) = \sum_{k=1}^{K} \pi_k(x)\, \frac{\exp\left(h_k(x)^\top w_y\right)}{\sum_{j} \exp\left(h_k(x)^\top w_j\right)}. $$

The log of that expression is no longer a low-rank object. Each component is constrained in probability space, but the sum is a convex combination, and mixing before taking the log is something no single rank-$d$ factorization reproduces. The ceiling on $\operatorname{rank}(P)$ is gone; what replaces it is the cost of $K$ distributions per token.

A sketch of the mechanics, close enough to the standard implementation:

```python
class MixtureOfSoftmaxes(nn.Module):
 def __init__(self, d_model, d_vocab, n_components):
 super().__init__()
 self.prior = nn.Linear(d_model, n_components)
 self.mixture = nn.Linear(d_model, n_components * d_model)
 self.output = nn.Linear(d_model, d_vocab, bias=False)
 self.k = n_components

 def forward(self, h):
 pi = torch.softmax(self.prior(h), dim=-1) # (B, K)
 z = torch.tanh(self.mixture(h))
 z = z.view(*h.shape[:-1], self.k, -1) # (B, K, D)
 logits = self.output(z) # (B, K, V)
 return torch.logsumexp(
 pi.unsqueeze(-1).log() + torch.log_softmax(logits, dim=-1), dim=-2
 )
```

The table below makes the trade explicit.

| Parametrization | Ceiling on $\operatorname{rank}(P)$ | Cost per token |
| --- | --- | --- |
| Single softmax at width $d$ | $d + 1$ | $O(Vd)$ |
| Mixture of $K$ softmaxes | not bounded by $d$ alone | $O(KVd)$ |
| Single softmax widened to $d'$ | $d' + 1$ | $O(Vd')$ |

The last row matters. There is more than one way to buy rank, and widening the residual stream buys it too. That is the first hint that the bottleneck is a property of a parametrization rather than an iron law about the model.

## This is a statement about the head, not the body

It is easy to read the result as a verdict on the architecture, and that reading is wrong. The bound applies to the map from the final hidden state to the vocabulary. The transformer body that computes $h(x)$ is untouched by it. Attention blocks, feed-forward layers, normalization, and the residual stream are not rank-limited in the same way, because their outputs are not passed through a single linear map onto a fixed vocabulary followed by a normalizer.

This distinction has consequences. When people say a model is "bottlenecked," they normally mean the body cannot represent the input well enough. Here the body may represent the input perfectly and the head still cannot turn that representation into the required distribution. The fix therefore lives at the boundary: add components, widen, share the embedding matrix or not, factorize the vocabulary. Changing the body does not remove the ceiling; it changes $d$, which changes where the ceiling sits.

The two quantities are also measured differently. The capacity of the body is about the geometry of $h(x)$ across contexts - how distinguishable the representations are, how much of the input survives to the last layer. The capacity of the head is the rank of a matrix over contexts times vocabulary. Confusing the two produces a whole genre of confused commentary.

## The contrarian point: width is not a wall

The standard framing of this result is that language models are fundamentally limited by the softmax output layer, and that the limitation is structural rather than practical. I think that framing is wrong in the way that matters, and the reasons are concrete.

The first is that the bound is on the rank of the logit matrix, and models buy rank aggressively. Widths that would have seemed absurd a few years ago are ordinary, and increasing $d$ raises the ceiling linearly. Depth compounds the effect indirectly, because a deeper network can place more of the discriminative work in the representation itself, leaving the head with an easier matrix to approximate. The claim "rank is capped at $d+1$" is true and, at contemporary widths, unremarkable. Saying it louder does not make it a wall.

The second reason is statistical. Exact representation of the full matrix is the wrong target. The objective is the log-likelihood of one observed token per position, and most of the vocabulary carries negligible mass. A rank-deficient $P$ can be excellent where it counts - on the head of the distribution, which is what perplexity and downstream accuracy care about - while being visibly wrong in the tail, which little of the loss touches. The reported gains from higher-rank parametrizations are real, but the question worth asking is whether they look like a wall coming down or like a nuisance being relieved.

There is a third, subtler problem with the strong reading. The argument needs the true conditional distributions of natural language to be high rank. Measuring that is hard, and any measurement depends on the corpus, the tokenizer, and the context window. A claim about the rank of a matrix you cannot observe is a claim about a model of the data, not about the data. It is entirely possible that the effective rank any given benchmark requires is well below the ceiling, in which case the bottleneck is not binding at all on that benchmark. That raises the practical question.

## Measure the rank you need, then compare

If the interesting question is empirical, the useful move is to stop arguing about ceilings and start measuring both sides of the comparison. Spectral entropy summarizes how much rank a matrix is actually using:

$$ \operatorname{erank}(P) = \exp\left(-\sum_{i} \bar\sigma_i \log \bar\sigma_i\right), \qquad \bar\sigma_i = \frac{\sigma_i}{\sum_j \sigma_j}, $$

where $\sigma_i$ are singular values. Apply it to the log-probability matrix your model produces, and apply it, as far as the data permits, to the empirical distribution you are trying to match. The comparison tells you whether the ceiling is close to binding or far away, which is the only version of the question that changes any decision.

I would do it in three steps. Compute $\operatorname{erank}$ of the model's log-probability matrix over a held-out batch of contexts, and compare it against the best estimate of the target's rank you can build on the same contexts. Look at where the rank is consumed: if a small set of contexts accounts for most of the spectral energy, the average is hiding the interesting cases. Then check whether widening, adding mixture components, or improving the body moves the number - a ceiling that does not respond to any of these is telling you something different from one that moves monotonically with width.

None of this settles whether the bottleneck matters for a particular training run. It does replace a claim repeated as though it were proven with a measurement that can be wrong in public, which is the more useful kind of claim.

The softmax output layer is a linear map with a normalizer, and its rank is capped by the width feeding into it. That is a real and checkable fact about the parametrization, and it is worth knowing. It is not a proof that language models cannot represent language, and the difference between those two statements is where most of the engineering lives.

## Sources

- Zhilin Yang, Zichao Yang, Yiming Cui, Zihang Dai, Ruobing Xie, William Cohen, Ruslan Salakhutdinov, Graham Neubig (2017). *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model*. https://arxiv.org/abs/1711.03953
- Lloyd N. Trefethen and David Bau III (1997). *Numerical Linear Algebra*. https://people.maths.ox.ac.uk/trefethen/text.html
