---
title: "Sparse Autoencoders at Scale"
description: "An SAE is a dictionary assumption about activations, so a feature that reconstructs well may not be the direction the model actually reads."
pubDatetime: "2026-09-26T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Interpretability
  - Sparse Autoencoders
categories:
  - Math
---

Suppose a feature fires on citations. You clamp it to zero at layer twelve, run the prompt again, and the output is the same token sequence. Reconstruction loss at that layer is small, the feature log is clean, and the intervention does nothing.

That is not a training bug. The SAE was trained to find a sparse dictionary that spans the activations, and it did. Whether those directions are the ones the model computes with is a separate question, and the loss never asks it.

## The assumption is a dictionary

Take one token's activation at one layer, $x \in \mathbb{R}^d$. An SAE asserts

$$
x \approx W_d f(x) + b, \qquad f(x) = \sigma(W_e x + b_e),
$$

with $f$ sparse. The dictionary has $m$ columns and $m > d$, so they are dependent and the decomposition is not unique. Uniqueness comes from the penalty, the $\ell_1$ norm on $f$ or a top-$k$ constraint. Sparsity is not a property of the model. It is the tie-breaker that picks one representation from the many that fit.

Three claims are bundled here. Features are directions, so a concept is a vector. A token activates few of them, so the code is sparse per position. Contributions add, so a composed activation is a sum. Each is a claim about the model. The loss checks none of them.

Identifiability follows. When two concepts co-occur, one direction explaining both costs less $\ell_1$ than two explaining one each, so the penalty merges them. When a narrow concept always appears inside a broad one, the broad feature absorbs it and the narrow goes dark. Merging and absorption are consequences of the objective, not optimisation failures.

So a trained SAE is a sparse overcomplete basis for a layer's activations, selected by a penalty, at a width you chose. That is a coordinate system. Calling its coordinates features adds the claim that the model uses that coordinate system too.

## Reconstruction is a spanning test

The loss is mean squared error. Read as variance, it is the part of the activation the dictionary failed to span. Minimising it selects the dictionary whose span covers the data, weighted by how much variance each direction carries. That is a real criterion. A dictionary missing a large subspace is useless downstream, and the loss says so.

Spanning is weaker than it sounds. Two dictionaries can reach the same error and mean different things. Say two concepts co-occur nine times in ten. The dictionary can hold one direction for the pair, or two at a small angle. Both span the same subspace and the error agrees to first order. One says the model has a combined feature. The other says it has two that fire together. Reconstruction cannot choose.

The weighting is the second problem. Reconstruction is dominated by high-variance directions, and variance is not importance. A formatting artifact that varies with every token can carry more variance than a switch that changes a decision, so capacity goes to the artifact. A refusal flag can be small in norm and decisive in effect.

That is why a good reconstruction number is compatible with a feature set that explains nothing about behaviour. The number answers a question about coverage. You are asking one about mechanism.

| Level | Measured | A positive result establishes | It does not establish |
| --- | --- | --- | --- |
| Reconstruction | $\lVert x - \hat x \rVert^2$ | the dictionary spans the activations | that its directions match the computation |
| Decoding | probe or linear map from $f$ | the feature carries the property | that the model reads the property there |
| Intervention | ablation, clamping, steering | the direction changes behaviour | that it is the direction used in the forward pass, or that it is monosemantic |

Most reports give the first row plus one example from the third. The middle row carries a great many interpretability claims, and it is the weakest, because a probe finds information whether or not the model routes it anywhere.

## The two directions are not the same direction

Every SAE has an encoder and a decoder, and they are not tied. The encoder rows decide when a feature is on, $f_i = \sigma(w_i^\top x + b_i)$. The decoder columns decide what it writes back, $\hat x = \sum_i f_i d_i$. Nothing in the loss forces $w_i$ and $d_i$ to be parallel. The objective only needs the product to land near $x$.

The consequence gets missed. Probing uses the encoder activation $f_i$. Steering usually uses the decoder direction $d_i$. Those are different experiments on different objects, and a feature can be selective on the encoder side while writing something tilted on the decoder side.

My view is that this is the most under-reported diagnostic in the area. Cosine similarity between $w_i$ and $d_i$, reported across a layer, would say whether the coordinate system is self-consistent. Where it is not, the encoder activation is not a description of what the direction does.

## A feature that decodes is not a feature the model uses

Decodability is a statement about the linear span of activations. Use is a statement about the computation that reads them. A probe can recover a property the model never routes to any head, because the residual stream is a wide bus and everything upstream writes to it. Presence is cheap. Use needs a path.

Paek and coauthors ([arXiv:2510.23802](https://arxiv.org/abs/2510.23802), 2025) make the gap visible in audio. They train SAEs on audio autoencoder latents, then learn linear maps from SAE features to discretised acoustic properties: pitch, amplitude, timbre. The mapping is the useful part, because it ties a feature to a property measurable without a human label. They validate on continuous and discrete latents, then use the features to manipulate generation and trace how those properties emerge during synthesis.

Read the mapping carefully. A linear map from a feature to a discretised pitch value shows the feature carries pitch. It does not show the generator reads pitch from it. The manipulation experiments move the claim from carrying to using, and they are separate work. The authors name the obstacle directly: audio is dense, compression obscures semantic meaning, and automatic characterisation is limited. Grounding in measurable acoustics answers the last one.

He and coauthors ([arXiv:2606.26620](https://arxiv.org/abs/2606.26620), 2026) keep two axes apart for the same reason. Their Qwen3-Instruct SAE suite covers Qwen3-1.7B, 4B and 8B, with layer-wise SAEs at residual streams, MLP outputs and attention outputs for the smaller two, and residual-stream layers for the 8B. They evaluate with activation-level reconstruction metrics and model-level recovery metrics, and report distinct sparsity-fidelity trade-offs across layers and components. If reconstruction implied use, the second axis would be redundant.

Their refusal-steering case study then supplies causal evidence, with a qualifier worth keeping: selected features steer refusal. Selected, not all of them.

## Against the reconstruction-interpretability trade-off

The standard framing is a frontier. Raise sparsity and reconstruction degrades. Lower it and features blur into polysemantic directions. The frontier is presented as a property of the method.

I think it is a property of the architecture, and the architecture was chosen.

Muchane, Richardson, Park and Veitch ([arXiv:2506.01197](https://arxiv.org/abs/2506.01197), 2025) make that concrete. Sparse dictionary learning, they note, neither exploits nor represents the semantic relationships between the concepts it finds. A flat list treats "dog" and "mammal" as orthogonal, which is a strange prior for language. They modify the architecture so the SAE models a semantic hierarchy explicitly, and report that the hierarchy can be learned from LLM internals, that it improves both reconstruction and interpretability, and that it improves computational efficiency.

If the trade-off were fundamental, that combination would be unavailable. You would buy one with the other and never both. The frontier moved, so it was not a law.

The reading that follows is mine. Most SAE ablations sweep sparsity at fixed width and a fixed flat dictionary. That traces a slice through one design, and the trade-off inside the slice is real. It is not the whole space.

I would not overclaim from one paper. One result on one architecture does not show hierarchy is the right prior, or that it survives at scale. It does show the frontier depends on assumptions you built in.

## What scale changes

At millions of features the training run stops being the hard part.

He and coauthors say directly that training SAEs is expensive and that open suites are limited, which is why a release covering three model sizes at several activation sites is worth having.

Evaluation changes shape. You cannot read a million features. Dashboards and automated interpretation are sampling devices, and the sample is biased toward features that fire often, look clean, and map to a concept English already has a word for. Features that fire rarely, or that encode something unnamed, are under-represented in exactly the reports claiming to characterise the set.

Then there is selection. Scan a million features for the one that fires on your phenomenon and you will find one. The interesting number is not how well it fires. It is how many you looked at before you chose it. A single feature reported without its search is a maximum, not a measurement.

Paek and coauthors offer one route out of the adjudication problem: attach the feature to a property measurable without a human in the loop. Pitch can be measured from audio. "Legal disclaimer" cannot be measured from anything. Where a ground truth exists, grounding turns an interpretation into a test.

## What I would measure

Four numbers per layer and activation site, in the order I would defend them.

Reconstruction, with width and sparsity attached. This is the coverage number and the least interesting of the four.

A behavioural gap. Patch the true activation back into the model, patch the reconstruction, compare next-token distributions. KL divergence and top-1 agreement are what I would publish. If the reconstruction changes the model's answer, the SAE is not describing the computation at that site, whatever the loss says.

A decoder-side intervention. Ablate the write-back direction, not the encoder activation. That matches the causal claim people want to make, and it costs one forward pass per feature.

A random sample of features, tested causally, with the sample size fixed in advance. Not the ones that looked good.

The behavioural gap is short enough to state in a few lines.

```python
def behavioral_gap(model, sae, site, batch):
    x = capture(model, site, batch)         # true activation
    x_hat = sae.decode(sae.encode(x))       # what the SAE keeps
    p = next_token_probs(model, batch, patch=(site, x))
    q = next_token_probs(model, batch, patch=(site, x_hat))
    return {
        "kl": kl_div(p.log(), q, reduction="batchmean"),
        "top1_agreement": (p.argmax(-1) == q.argmax(-1)).float().mean(),
    }
```

The gap between $p$ and $q$ is the honest measure of whether the dictionary describes the model or merely spans it. On a well-behaved layer it is small. On a layer where the SAE is fitting an artifact it is not, and the reconstruction number will not warn you.

The SAE is a lens with a known distortion. Reconstruction tells you the lens covers the field. It does not tell you the lens is in focus on the thing you are looking at, because focus is a property of the object and the lens together. The results I trust most here moved something and watched the output change: refusal steering in the Qwen3 suite, acoustic manipulation in the audio work. Those claims are earned at the intervention, not at the loss curve. Everything upstream is a candidate.

## Sources

He, XinYang, Wei Wang, Bing Zhao, Xuan Ren, WenBo Li, WeiXu Qiao, Hu Wei, Lin Qu. "Discovering Millions of Interpretable Features with Sparse Autoencoders." [arXiv:2606.26620](https://arxiv.org/abs/2606.26620). 2026.

Muchane, Mark, Sean Richardson, Kiho Park, Victor Veitch. "Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures." [arXiv:2506.01197](https://arxiv.org/abs/2506.01197). 2025.

Paek, Nathan, Yongyi Zang, Qihui Yang, Randal Leistikow. "Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders." [arXiv:2510.23802](https://arxiv.org/abs/2510.23802). 2025.
