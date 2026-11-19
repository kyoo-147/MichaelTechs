---
title: "Decoding Is a Distribution Edit"
description: "Greedy, beam, temperature, top-k and nucleus sampling are all edits to the next-token distribution, and choosing that edit is the real design decision."
pubDatetime: "2023-12-02T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Decoding
  - Sampling
  - Language Models
categories:
  - AI
---

A language model does not produce text. It produces a probability distribution over the next token, and then some piece of code — well below the level of anything you would call the model — decides what to do with it. Almost every argument I have had about model "creativity" is really an argument about that second step. People attribute to the weights a property that belongs to the sampler.

The framing is worth laying out plainly. Once decoding is a sequence of edits to a distribution, the field's vocabulary stops being a bag of tricks and becomes a small design space.

## The object the model actually returns

Start with a single step. Attention builds a context-dependent representation of everything so far; the final layer projects it onto a vector of size equal to the vocabulary, and softmax maps that onto the simplex. If $z \in \mathbb{R}^{V}$ are the raw logits and $V$ is the vocabulary size, then

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}
$$

and $p$ is a point on the probability simplex: $V$ non-negative entries summing to one. That is the entire output; everything downstream is arithmetic on $p$.

Two details matter later. First, $p$ is not a suggestion or a ranking with noise attached — it is a fully specified categorical distribution, and a draw from it is well defined. Second, $p$ is what it is: fixed by the weights, the prompt, and the preceding context. Nothing at decode time can add probability to a token the model considered unlikely, except by renormalising over a smaller set, which removes mass rather than creating it. That asymmetry is the whole story.

The transformer that produces this distribution was described by Vaswani et al. (2017); the autoregressive loop that consumes it — feed the chosen token back in, repeat — is older than any of this. The loop is where a *distribution* becomes a *string*, and the string is what everyone judges.

## Every decoder is an edit plus a draw

A decoder takes $p$, returns a modified distribution $q$, and samples from $q$ — or, in the degenerate case, takes its mode. Different decoders are different edits.

Greedy decoding replaces $q$ with a point mass on $\arg\max_i p_i$. Beam search keeps the top $k$ partial sequences and re-ranks completed ones by summed log-probability, which edits across time rather than within a step. Temperature divides the logits by $T$ before the softmax. Top-$k$ zeroes all but the $k$ largest entries and renormalises. Nucleus sampling keeps the smallest set of tokens whose cumulative probability exceeds $p$ and renormalises.

That is the practical list. Five edits, one draw. Laid out this way, three things become visible that are otherwise buried in folklore: maximisation is not the safe default, the difference between top-$k$ and nucleus is structural rather than cosmetic, and temperature does something narrower than most people think.

## Temperature is a power, not a mood

Temperature acts on the logits, which means it acts multiplicatively on the probabilities. With $q_i \propto \exp(z_i/T)$, for any two tokens,

$$
\frac{q_i}{q_j} = \left(\frac{p_i}{p_j}\right)^{1/T}
$$

So temperature is a power applied to the probability ratio. As $T \to 0$ the ratios diverge and the distribution collapses onto the mode — greedy decoding is the $T \to 0$ limit, not a separate algorithm. As $T \to \infty$ the ratios converge to one and $q$ approaches uniform. At $T = 1$ nothing happens at all.

The quantity that captures "how spread out is this" is the entropy,

$$
H(p) = -\sum_{i=1}^{V} p_i \log p_i
$$

maximised at $\log V$ by the uniform distribution and equal to $0$ at a point mass. Temperature moves $H$ monotonically: cooling lowers it, heating raises it. That is what the knob does. It changes the concentration of a distribution whose shape — which tokens are high, which are low, and their ordering — was already fixed by the model.

That last clause is the part people skip. The power transform is monotone on the positive reals, so it cannot reorder tokens. You are not changing *what the model prefers*. You are changing *how decisively it prefers it*.

## Truncation: fixed size versus fixed mass

Both top-$k$ and nucleus sampling throw away the tail. They differ in what they hold fixed.

Top-$k$ holds the count. With $k = 50$ you always keep 50 tokens, whether the model is confident or lost. On a step where one token carries $p = 0.99$, keeping 50 means keeping 49 tokens with total mass near $0.01$, renormalised up roughly a hundredfold. Every so often you draw one. On a step where 500 continuations each sit near $0.002$, keeping 50 truncates most of the real mass and forces a choice among a small arbitrary subset. The same $k$ is too permissive in one regime and too restrictive in the other.

Nucleus sampling, from Holtzman et al. (2019), holds the mass instead. Sort $p$ descending, take the smallest prefix whose cumulative sum reaches $p$, renormalise. The kept set now adapts: a confident step yields a nucleus of one or two tokens, an uncertain one yields hundreds. The boundary follows the shape of the distribution instead of imposing a shape on it — the difference between a fixed-width window and one sized to the thing you are looking at.

The parameter has a clean reading from here: $p$ is the fraction of the model's mass you have committed to exposing. At $p = 0.5$ you discard half of what it believes, which is far more aggressive than the number suggests.

## The degeneracy of pure maximisation

The obvious objection to sampling is that it introduces error. If the model's best guess is the highest-probability token, why randomise?

Because the argmax of a conditional is not the argmax of the sequence. Greedy decoding maximises $p(x_t \mid x_{<t})$ at each step; what you care about is $p(x_{1:T})$, and these are different objectives. A locally optimal choice can walk you into a region of context space where every continuation is low-probability, and once there you cannot leave — each subsequent argmax is conditioned on the bad prefix.

The symptom is repetition. Text that loops, that echoes a phrase until the context runs out, that produces a sentence and then its own paraphrase. This is not a bug in any meaningful sense. High-probability continuations of a repetitive prefix are more repetition, because the context has become self-similar. The distribution does exactly what it should and the decoder traps you in it.

Beam search does not rescue you: it optimises sequence log-probability, a better proxy, but the highest-probability string under a language model tends to be bland, because high probability accrues to continuations seen often. Maximising likelihood selects for typicality, and typicality is boring.

So sampling is not a cosmetic choice or a creativity slider bolted on for demos. It is a correction to an objective mismatch. You are not adding randomness for fun; you are declining to follow a locally greedy path that provably leads somewhere worse.

## The contrarian read: temperature does not control creativity

The standard framing, repeated in every tutorial I have read, is that temperature controls creativity. Low is safe and factual, high is creative and surprising. I think that framing is a category error, and it makes people reach for the wrong knob.

Temperature is a monotone reparameterisation of a fixed distribution. It cannot introduce a zero-probability token, and it cannot reorder the non-zero ones. If the weights do not represent a connection between two ideas, no temperature surfaces it. If they do represent it but assign it low probability, heat makes that token likelier to be drawn — and also makes every other low-probability token likelier, because it flattens the whole distribution. You get surprise. You also get the specific kind of surprise that comes from drawing nearly uniformly from a vocabulary, which is noise, not creativity.

The information-theoretic statement is blunt. The mutual information between the sampled token and the prompt is bounded by what the conditional contains. Decoding is a channel from $p$ to a token, and no channel widens its own input. Heating raises the entropy of the output, but entropy is not information — it is the absence of it. Raising $T$ trades fidelity for diversity along a fixed curve, and that curve is not a creativity curve.

What the knobs control is risk exposure. Low temperature bets on the mode; high temperature bets on the tail. The tail holds both the apt-but-unlikely continuation and the incoherent one, and the sampler cannot tell them apart, because that distinction lives in the weights rather than the edit. When you raise temperature hoping for insight, you are deciding to expose more tail and accepting whatever is in it.

If you want a model to produce something it currently cannot, the fix is upstream: prompting, retrieval, fine-tuning, or a different model. If you want the same model to explore a region it already assigns non-trivial probability to, sampling helps — and nucleus helps more than temperature, because it truncates the genuinely hopeless tail while keeping diversity among plausible options. Temperature is the crudest version of that, the only knob that raises entropy with no notion of which tokens deserve the mass.

Push temperature past 1 on a task where the model is unsure of a fact and you get a wrong answer stated with more confidence, which then gets blamed on the model running hot. The model was doing what it was told: the distribution held a wrong token as plausible, and the edit made it likelier to be drawn. The bug was in the belief, not the draw.

The parameter you are choosing is how much of the tail to expose, and that is a question about your tolerance for being wrong, not about how imaginative you want the output to be.

## What this means in practice

Treat the sampler as part of the specification. Decoding parameters are not a config detail to leave at defaults; two deployments with identical weights and different $p$ are different systems, and evaluating one tells you about that one.

Prefer nucleus over top-$k$ unless you have a reason. The fixed-count failure modes are predictable: too permissive on confident steps, too restrictive on uncertain ones.

Keep temperature at 1 when measuring what the model knows, because temperature distorts the model's stated beliefs.

Use low temperature for extraction and high for exploration, but remember that exploration means sampling the tail, not discovering knowledge. The tail is where uncertainty lives. Drawing from it tells you what the model is unsure about; it does not make the model right.

## Sources

- Vaswani, A. et al. (2017). *Attention Is All You Need.* https://arxiv.org/abs/1706.03762
- Holtzman, A. et al. (2019). *The Curious Case of Neural Text Degeneration.* https://arxiv.org/abs/1904.09751
