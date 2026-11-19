---
title: "Embedding Anisotropy and Retrieval"
description: "Embeddings collapse into a narrow cone, which shrinks score margins; how normalisation and centring reorder results, and how to measure it."
pubDatetime: "2025-12-06T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Embeddings
  - Retrieval
categories:
  - Math
---

Retrieval systems rarely fail by returning nothing. They fail by returning the right document at rank four, behind three that scored a few thousandths higher, and by returning a different order every time the index is rebuilt. The embedding model has not changed. The margin has.

## The cone shows up as lost margin

Most teams meet anisotropy as a symptom. A relevant passage and an irrelevant one both come back around 0.8. The order is correct, but every threshold or relevance cutoff calibrated on that score now behaves badly, and nothing flagged it.

The geometry is simple. Write an embedding as a shared component plus a residual, $x_i = \mu + r_i$, with $\|\mu\|^2 = c^2$ and residual variance $v = \mathbb{E}\|r_i\|^2$. For uncorrelated residuals the cosine collapses to a ratio of the shared part to the whole:

$$
\cos(x_i, x_j) = \frac{c^2 + \langle r_i, r_j \rangle}{c^2+v} \approx \frac{c^2}{c^2+v}.
$$

If the shared component carries four times the variance of the residual, unrelated documents sit at 0.8 by construction. The cone is not an arithmetic defect. It is what happens when most of every vector is identical across inputs. Stollenwerk and Stollenwerk trace that shared component to the second moment in Adam and modify the optimiser to reduce it.

Cosine stays monotone in the residual angle, so the cone does not scramble the order. It compresses the range. If unrelated pairs start at 0.8, the useful ranking lives between 0.8 and 1.0, and the gap between rank one and rank fifty can be smaller than the error from product quantisation, float16, or a traversal that misses the right neighbourhood.

That decides what to fix. A cone that compresses scores is a precision and calibration problem. A cone that also flattens the residual structure is a representation problem. They look identical in a histogram.

## Normalisation and centring are not the same operation

Both are sold as fixes. They act on different things.

For exact cosine, L2 normalisation is a no-op. Cosine already divides by the norms, so scaling every vector onto the unit sphere cannot change an ordering. It matters once you stop computing exact cosine. Inner-product indexes rank by $q^\top x_i$, where the norm is part of the score, and normalising changes the order unless every norm was equal. It matters at build time too: centroids and codebooks are fitted to whatever distribution you hand them.

Centring translates the origin, and cosine is the angle at the origin. The resulting change is not a monotone function of the old score. Take $q=(1,0)$, mean $\mu=(1,-0.5)$, and documents $d_1=(2,-0.5)$, $d_2=(0.5,2)$. Before centring the cosines are $0.970$ and $0.243$. After centring the query is $q'=(0,0.5)$, the documents are $d_1'=(1,0)$ and $d_2'=(-0.5,2.5)$, and the cosines are $0.000$ and $0.981$. The document that was nearly parallel to the query is now orthogonal, because the query moved onto the axis that document does not occupy. All vectors were unit-normalised, so the flip is not a norm effect.

Any corpus whose mean points away from the query distribution will reorder. Whether that helps is an empirical question.

Two variants deserve separate names. Subtracting the mean is R1; projecting each vector off the mean direction is R2. Ren, Sun and Liang show that R2 cancels the part of the mean-estimation error lying along the mean, which R1 leaves in. In their MMTEB sweep, mild single-direction removal helped and full PCA whitening hurt every model they tested.

## A good representation and a good index fail differently

Representation quality asks whether the geometry separates what the task needs. Index quality asks whether the search structure returns the true nearest neighbours inside a latency and memory budget. A system can be good at one and broken at the other.

An exhaustive scan over a strongly anisotropic representation returns the right neighbours. It is slow and correct. A well-shaped representation inside a badly tuned graph returns plausible neighbours that are not the nearest ones. Most production incidents are the second kind; most anisotropy discussions are about the first.

Inside the index the cone stops being neutral. Product quantisation fits codebooks to the data, then encodes each vector as a residual against its centroid. When most of every vector is the shared component, the quantiser spends its dynamic range on the part that is identical across documents, and the distinguishing part gets whatever bits remain. The same argument covers float16: an absolute error budget applied to a shrunk signal.

Cosine is also a proxy. Wang, Ngo and Lim show that for image-to-recipe retrieval a photograph cannot express every detail of a recipe, so similarity judgments inherit a bias a causal adjustment can remove. Geometry measures something, and it is not always relevance.

Nor does a better encoder repair the ordering, because the ordering is produced by an objective. Mekonnen, Tang and de Rijke skip reward modelling and optimise document relevance directly with a pairwise ranking loss, reporting MRR gains on MS MARCO and Natural Questions. Zhuang and coauthors train an instruction-aware embedding model on triplets where instruction and query are the variables. Both representations are shaped by what the objective asked them to order.

## Measuring before you blame the model

Order matters, because the cheapest explanations come first.

Start with a floor. Sample a few thousand documents, compute mean pairwise cosine, write it down. That is what unrelated looks like here. If it sits at 0.75, scores are compressed and every absolute threshold downstream is suspect.

Then test whether the anisotropy is an artifact of a few dimensions. Timkey and van Schijndel found that one to three rogue dimensions dominate cosine and Euclidean similarity in contextual models, and that the dimensions dominating similarity are not the ones that matter to behaviour. Their correction is unglamorous: standardise the dimensions. If z-scoring changes your ranking substantially, you were measuring those dimensions.

Then read the spectrum instead of the average. The effective rank,

$$
\mathrm{erank} = \exp\left(-\sum_k p_k \log p_k\right), \qquad p_k = \frac{\lambda_k}{\sum_j \lambda_j},
$$

and the participation ratio,

$$
\mathrm{PR} = \frac{\left(\sum_k \lambda_k\right)^2}{\sum_k \lambda_k^2},
$$

both come from the eigenvalues of the document covariance. A space with a thousand dimensions and an effective rank of thirty is a thirty dimensional space wearing a costume, and no index tuning changes that.

Yamagiwa and Shimodaira turn the description into a diagnosis. Total variance splits into within-cluster and between-cluster parts, and they observe that as layers deepen the embeddings move away from the origin and between-cluster variance falls. If that share collapses, different-topic documents are no easier to separate than same-topic ones.

Finally, evaluate with ranks. Recall@k, MRR, nDCG on held-out queries, before and after each transform. A score histogram is not an evaluation.

```python
import numpy as np

X = embeddings                          # (n_docs, d), float32
X = X / np.linalg.norm(X, axis=1, keepdims=True)

idx = np.random.default_rng(0).choice(len(X), 4096, replace=False)
S = X[idx] @ X[idx].T
n = len(idx)
print("mean pairwise cosine:", (S.sum() - n) / (n * (n - 1)))

lam = np.linalg.eigvalsh(np.cov(X.T))[::-1]
p = lam / lam.sum()
print("effective rank:", np.exp(-(p * np.log(p + 1e-12)).sum()))
print("participation ratio:", lam.sum() ** 2 / (lam ** 2).sum())

Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
print("mean cosine after per-dim standardisation:",
      ((Z @ Z.T).sum() - len(Z)) / (len(Z) * (len(Z) - 1)))
```

## The case against fixing anisotropy

The standard framing: embeddings are anisotropic, high average cosine means poor separation, so centre or whiten, and a lower average cosine means better retrieval. Each link can fail.

Anisotropy does not have to break the task. Wisniewski, Guillaume and Fernández measured strong anisotropy in wav2vec2 and HuBERT, with high similarity between random embeddings, then ran keyword spotting with dynamic time warping on them. The measure still identified words without transcription, across speakers. Whatever the cone was doing, it had not destroyed the phonetic structure the task needed.

Removing more anisotropy is not monotonically better. Ren, Sun and Liang report that mild single-direction removal helped while full PCA whitening hurt every model in their ablation. Forooghi, Sadeghi and Lu find whitening degrades classification and its efficacy depends on both model and task. A transform that lowers average cosine while lowering recall is a regression with tidier geometry.

The scalar also hides the shape. If one to three dimensions dominate similarity, the average mostly reports those dimensions. A model at 0.78 and one at 0.55 may rank equally well. A model at 0.60 may be broken. The number does not constrain the property you care about.

The corrected framing is narrower. Anisotropy describes where the data sits. It does not diagnose the system. The diagnosis is whether your metric moves on held-out queries when the geometry changes. If recall@10 does not move under centring, anisotropy was not your problem.

My own view is that centring should be a hyperparameter with a cost, not a default. It rewrites a stored index, invalidates thresholds, and the evidence that it helps is task dependent. That is a lot of blast radius.

## What survives contact with production

Freeze the transform and version it alongside the index. A query encoded before a centring change and compared against documents indexed after it produces silent nonsense, and it is the most common bug here.

Decide whether the norm carries signal before you normalise. If the encoder puts information there, normalising discards it. Prefer removing one direction over a full whitening unless evaluation says otherwise.

Recompute every absolute threshold afterwards, because a duplicate cutoff of 0.8 means little once scores move. Then measure the index cost too. A transform that lifts recall a point and doubles the memory bill is a decision, not a free win.

| Operation | What it changes | Where it can hurt |
| --- | --- | --- |
| L2 normalisation | Removes magnitude, keeps direction | Inner-product indexes and PQ codebooks see a different distribution |
| Mean subtraction (R1) | Translates the origin | Ranking stops being a monotone function of the old score |
| Mean-direction removal (R2) | Deletes one direction, keeps the rest | The mean estimate must be frozen and versioned with the index |
| Full whitening | Rescales every axis to unit variance | Reported to degrade classification and ranking; amplifies noise directions |
| Per-dimension standardisation | Rescales axes independently | Changes the implicit weighting of the metric |

The cone is real and it costs resolution. It is not the same as a bad encoder, and lowering the average cosine is not the same as improving retrieval. Measure the floor, test whether a few dimensions own the similarity, read the spectrum, and decide with a rank metric.

## Sources

- Felix Stollenwerk and Tobias Stollenwerk, "Better Embeddings with Coupled Adam", [arXiv:2502.08441](https://arxiv.org/abs/2502.08441), 2025.
- Xingyu Ren, Youran Sun and Haoyu Liang, "Correcting Mean Bias in Text Embeddings: A Refined Renormalization with Training-Free Improvements on MMTEB", [arXiv:2511.11041](https://arxiv.org/abs/2511.11041), 2025.
- William Timkey and Marten van Schijndel, "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality", [arXiv:2109.04404](https://arxiv.org/abs/2109.04404), 2021.
- Hiroaki Yamagiwa and Hidetoshi Shimodaira, "Norm of Mean Contextualized Embeddings Determines their Variance", [arXiv:2409.11253](https://arxiv.org/abs/2409.11253), 2024.
- Guillaume Wisniewski, Séverine Guillaume and Clara Rosina Fernández, "Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting", [arXiv:2506.11096](https://arxiv.org/abs/2506.11096), 2025.
- Ali Forooghi, Shaghayegh Sadeghi and Jianguo Lu, "Whitening Not Recommended for Classification Tasks in LLMs", [arXiv:2407.12886](https://arxiv.org/abs/2407.12886), 2024.
- Kidist Amde Mekonnen, Yubao Tang and Maarten de Rijke, "Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval", [arXiv:2504.05181](https://arxiv.org/abs/2504.05181), 2025.
- Yuchen Zhuang et al., "Towards Better Instruction Following Retrieval Models", [arXiv:2505.21439](https://arxiv.org/abs/2505.21439), 2025.
- Kawin Ethayarajh, "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings", [arXiv:1909.00512](https://arxiv.org/abs/1909.00512), 2019.
