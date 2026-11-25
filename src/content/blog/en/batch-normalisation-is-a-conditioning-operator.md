---
title: "Batch Normalisation Is a Conditioning Operator"
description: "The layer standardises each feature across a batch, which changes the conditioning of the next linear map, not the capacity of the network."
pubDatetime: "2023-06-10T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Batch Normalisation
  - Conditioning
  - Training
categories:
  - AI
---

## The layer is smaller than its reputation

Batch normalisation is introduced in most courses as the trick that makes deep networks train, and the explanation usually arrives in the same breath: it reduces internal covariate shift. I have repeated that sentence many times, in papers and in conversation, without stopping to ask what it commits me to. The layer has earned its place. The explanation has not earned the confidence it receives.

This post is not an argument about whether to use the layer. On most convolutional and fully connected networks I work with, it helps, and when it hurts the reason is usually diagnosable. The argument is narrower. The layer performs a specific, checkable operation on the data; that operation is best understood as a change to the conditioning of the optimisation problem, not to the expressive power of the model; and the story that is normally told instead has survived on repetition rather than on evidence.

## What the layer computes, exactly

For a mini-batch of $m$ activations $x_1,\dots,x_m$ at one position in the network, batch normalisation first computes the batch mean and variance per feature,

$$\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i,\qquad \sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2,$$

then standardises and applies a learned affine map:

$$\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y_i=\gamma\odot\hat{x}_i+\beta.$$

The $\epsilon$ keeps the division well behaved when a feature is nearly constant, and $\gamma$ and $\beta$ are learned, one pair per feature. Everything else in the layer follows from those two equations.

The important structural fact is that $\gamma$ and $\beta$ make the layer invertible as a map on a fixed batch. Choose $\gamma=\sqrt{\sigma_B^2+\epsilon}$ and $\beta=\mu_B$ and the identity comes back. A layer that contains the identity in its parameter space cannot be adding representational power in any meaningful sense, because undoing it would undo the benefit it is credited with. It does not.

## Conditioning, not capacity

Take a linear map $z=Wy$ following the layer. If the entries of $y$ have very different scales, the columns of $W$ receive inputs of very different magnitude, and the curvature of the loss with respect to $W$ inherits that spread. The classical measure of how badly a problem behaves under small perturbations is the condition number $\kappa(W)=\lVert W\rVert\,\lVert W^{-1}\rVert$. A large condition number means the loss surface has directions of very different stiffness, so a single global learning rate is a compromise that fits no direction well: small enough for the stiff directions, needlessly slow along the flat ones. Trefethen and Bau put the general principle plainly - the condition number measures how much a problem amplifies error, and it belongs to the problem as posed, not to the function being approximated.

Batch normalisation re-poses the problem. It hands the next linear map an input whose first two moments are fixed, computed from data rather than chosen by the architecture. The next map therefore starts from a standardised coordinate system instead of whatever scale happened to fall out of the previous layer's initialisation. That is preconditioning in the ordinary numerical sense: a change of coordinates that makes the same problem easier to solve iteratively.

There is a second, sharper piece of evidence for this reading. At inference time the statistics are constants, and the layer folds away:

$$y_i=\gamma\frac{Wx_i+b-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta=W'x_i+b',\qquad W'=\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W.$$

A deployed network with batch normalisation is a network without it, with rescaled weights. Nothing about the function class changed between training and deployment; what changed is the scale of the parameters the optimiser had to find. Whatever the layer contributes is contributed during optimisation and then absorbed into the weights.

The gradient story points the same way. Because $\mu_B$ and $\sigma_B^2$ depend on every example in the batch, the Jacobian of the layer is not diagonal: differentiating through the statistics produces terms that subtract the component of the gradient along the batch mean and along the current normalised activation. The layer projects part of the gradient away. That is a statement about the geometry of the update, not about the richness of the model.

## Where the covariate shift story comes from

The original paper's motivation is that the input distribution to each layer shifts during training as the layers below it change, so each layer is repeatedly adapting to a moving target. Normalising the inputs is supposed to hold the target still.

The premise is true and nearly vacuous. In any network trained by gradient descent, the distribution of inputs to every layer above the first changes constantly, because the parameters below it change. That is what training is. To turn the observation into a testable hypothesis you need a measure that separates drift that hurts from drift that is ordinary, plus evidence that suppressing the first removes the difficulty. The paper argued the case; it did not isolate the cause.

That gap was taken seriously later. In 2018, Santurkar, Tsipras, Ilyas and Madry tested the causal claim directly: they compared networks with and without the layer, and they deliberately injected covariate shift into batch-normalised networks to see whether the benefit disappeared. It did not. What they found instead was that the layer improves the Lipschitzness of the loss and of its gradients, which is to say it makes the optimisation space smoother. Their conclusion, stated plainly, is that the mechanism is better explained by smoothing of that space than by the suppression of covariate shift. I do not read that as a refutation of the original paper so much as a correction of what it is allowed to claim. The layer was introduced with a mechanism attached, the mechanism was not demonstrated, and the field has spent years teaching the mechanism as though it had been.

## Training and inference are different functions

Because the statistics come from the batch during training and from stored estimates at inference, the layer is literally a different function in the two modes:

| | Training | Inference |
| --- | --- | --- |
| Mean and variance | Computed from the current mini-batch | Read from running estimates |
| Depends on other examples in the batch | Yes, through $\mu_B$ and $\sigma_B^2$ | No |
| Effect of batch size | Changes the function being optimised | None |

Two consequences follow, and both are easy to forget in a training loop.

First, the running estimates are a separate object to maintain. They are updated with an exponential moving average whose momentum is another hyperparameter, and they are only as good as the batches they were accumulated from. If the training distribution is not stationary, or if the last batches seen are unrepresentative, the inference-time statistics are wrong in a way that no training loss will reveal. A model can look excellent on its training objective and behave differently the moment the mode switch flips.

Second, batch size stops being only a compute decision. The statistics are estimated from $m$ examples, so their noise falls as $m$ grows; small batches give a noisy normalisation, which is part of why very small batch training with this layer is delicate. The gradient of any single example's loss depends on the other examples in the batch, because the mean and variance are shared. Batch size therefore changes the objective, not merely the estimate of its gradient. That is why it has to be treated as a hyperparameter of the layer itself, and why changing it late in a project can invalidate tuning done earlier.

## The contrarian case: the explanation has never caught up with the layer

Here is the uncomfortable position, stated plainly. The covariate shift justification for batch normalisation has never been convincingly demonstrated, while the empirical effect of the layer is solid and easy to reproduce. We have a layer that works and a story that does not explain it. The honest response is to keep the layer and retire the story, not to keep repeating the story because the layer works.

Three things make me confident about this. First, the identity argument: a layer whose parameter space contains the identity cannot be adding capacity, so explanations that lean on what the network can represent are dead on arrival. Second, the inference argument: the layer vanishes into the weights at deployment, so whatever it does, it does to the trajectory of optimisation rather than to the final function class. Third, the direct test: when the proposed mechanism was manipulated experimentally, the benefit survived the manipulation. A hypothesis that survives the removal of its own proposed cause is not the explanation.

What replaces it is less satisfying and more accurate. The layer re-scales and re-centres its input using data-dependent statistics, which changes the conditioning of the following map and smooths the loss surface the optimiser sees. That is a real mechanism, it is measurable in principle, and it accounts for the facts I know of. It is also incomplete: smoothing is a description of an effect, not a derivation of why this particular smoothing helps this particular architecture. Nobody has that derivation, and saying so is more useful than borrowing confidence from a story.

The cost of repeating the old story is not rhetorical. It teaches people to reason about the layer in the wrong coordinate system: to expect it to help wherever distributions shift, to be surprised when it fails on small batches or recurrent stacks, and to reach for it when the actual problem is initialisation or learning rate. Folklore repeated in enough papers becomes an assumption, and assumptions stop being tested.

## What this changes in practice

Practically, the conditioning reading gives better questions. Does the next layer's input need a common scale? Is the batch large enough for the statistics to be stable? Are the running estimates representative of deployment? Does the batch size interact with the learning rate in a way that makes a tuning run non-transferable?

None of those questions requires believing the covariate shift story, and all of them are answerable with experiments on the model in front of you. That seems like the right trade: keep the operator, drop the folklore, and let the mechanism be as uncertain as the evidence actually is.

## Sources

- Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", 2015. https://arxiv.org/abs/1502.03167
- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, 1997. https://people.maths.ox.ac.uk/trefethen/text.html
