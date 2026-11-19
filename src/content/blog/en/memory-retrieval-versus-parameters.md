---
title: "Memory: Retrieval Versus Parameters"
description: "Weights compress facts and update slowly, retrieval keeps them exact and stale - the bill lands on update frequency, latency and privacy."
pubDatetime: "2026-02-28T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Memory
  - Retrieval
categories:
  - Math
---

On a Tuesday the price list changed. On Thursday the assistant still quoted the old number, in a fluent sentence that read like it had checked.

The system knew something on Monday. The world moved. Nothing in the stack noticed. There were only two places the new number could have come from: weights written before the change, or a document fetched during the request. Most arguments about memory in language models are arguments about which of those two to trust, and how often.

## Two Ways to Know a Fact

A model answers from parameters by sampling

$$y \sim p_\theta(y \mid x),$$

and from retrieval by sampling $y \sim p_\theta(y \mid x, \mathcal{D}_x)$, where $\mathcal{D}_x$ is whatever the retriever placed in the window. Same function, different conditioning set. That framing dissolves the metaphor: neither path is memory in the database sense. One is a compressed function of the training distribution. The other is a lookup whose result happens to be concatenated into the prompt.

The compressed path does two things at once, and they are usually conflated. It generalises, because the weights carry the shape of a domain rather than a list of its facts, which is why a model can say something useful about a document it never saw. And it forgets, because a fact written into $\theta$ is not addressable. Correcting one fact means touching every parameter that took part in encoding it, and you cannot enumerate them.

That asymmetry, addressable versus not, drives most of the engineering decisions that follow.

## What Weights Give Up

Facts stored in weights share capacity. In the linear picture, associations are superimposed across the same $d$ dimensions, so interference is governed by load per dimension rather than by the raw count of facts. Push more associations into a fixed space and recall gets noisier, and the noise is not confined to the fact you just added.

Updates make it worse in a specific way. A gradient step on new data moves $\theta$ along $-\nabla_\theta \mathcal{L}_{\text{new}}$, and that direction is not orthogonal to the directions encoding older behaviour. The projection onto those directions is the forgetting term. Care in the schedule and the data mix shrinks it. Nothing makes it zero.

There is a second cost that is organisational rather than mathematical. Changing a weight means a training run, an evaluation pass and a rollout plan. Even when continued pretraining is cheap in FLOPs, it is expensive in calendar time and in the review it demands. An index edit is a write to a row.

It is tempting to treat external memory as the escape from all this. Katraouras and Rafailidis (2026) report in [arXiv:2601.00756](https://arxiv.org/abs/2601.00756) that compressing a memory bank cuts it to 0.3% of the size of the strongest baseline while keeping high retention during online adaptation. Read that carefully. The win is real and it is a result about memory banks, not about weights. The lesson I take is that any store which grows with the data stream eventually gets compressed, and compression reintroduces the lossiness you moved to retrieval to avoid.

## Retrieval Is Exact About Bytes

A retriever returns the chunk you indexed. It is exact about those bytes. It is exact about very little else.

Three things go stale independently. The source document changes and the index has not caught up. The embedding model is upgraded and every neighbour in the space shifts. The chunk boundaries are redrawn and a paragraph that used to carry an answer now carries half of it. None of these raise an error.

A crude model helps me size the exposure. If the underlying record changes at rate $\lambda$ and the index trails the source by $\Delta$, then under a Poisson assumption the probability that a returned chunk is current is

$$P(\text{current}) = e^{-\lambda \Delta}.$$

That is a modelling choice, not a measurement. It is still useful, because it makes the trade explicit: shortening the index lag lowers the exponent, and for large $\lambda$ the exponent is where the product fails.

Staleness is only half of it. Retrieved text enters the context as evidence, and the model then decides what to do with it, so retrieval quality and answer correctness are different quantities. Schuster, Gautam and Markert (2026) test 13 open-weight models on inter-context knowledge conflicts in [arXiv:2601.03746](https://arxiv.org/abs/2601.03746). They find a preference for institutionally corroborated sources over individuals and social media, and they find that the preference reverses when the less credible source repeats its claim. Their mitigation cuts that repetition bias by up to 79.2% while retaining at least 72.5% of the original preferences.

For anyone running a retriever, that should be uncomfortable. A document retrieved on every request is a repeated source. Repetition is a lever on belief, and the index decides which claims get pulled most often.

## The Cost Surface

Three variables decide the architecture: how often facts change, how tight the latency budget is, and what deletion obligations you carry. The first two are arithmetic. Write the parameter path as

$$C_p = \frac{c_{\text{train}}}{T} + n_q\, c_{\text{fwd}}$$

and the retrieval path as

$$C_r = f\, c_{\text{write}} + n_q\left(c_{\text{fwd}} + c_{\text{embed}} + c_{\text{search}}(N) + c_{\text{out}}\, k \bar{m}\right),$$

where $f$ is edits per unit time, $T$ is the retraining period, $n_q$ is query volume, $N$ is index size, $k$ is chunks retrieved and $\bar{m}$ is mean chunk length. Retraining on a schedule competes only when

$$\frac{c_{\text{train}}}{T} < f\, c_{\text{write}},$$

which says that parametric upkeep is affordable exactly when the facts are stable. Nothing about model quality enters that inequality. It is a statement about churn.

Latency behaves the same way. The retrieval hop is visible, so engineers budget for it, and then miss the larger term: prefilling $k\bar{m}$ extra tokens that every layer must attend to. In [arXiv:2602.07525](https://arxiv.org/abs/2602.07525), Hou et al. (2026) report that their retrieval framework needs an average of 6.3k tokens or more per query and a minimum of 3.0k, adapting to query difficulty, alongside reported gains of 4.8% EM and 5.0% F1 over the strongest baseline. The adaptive token cost is the honest part of that design. Retrieval spend tracks how hard the question is. A parametric answer has a flat bill, whatever it knows.

| Property | Weights | Retrieval |
|---|---|---|
| Unit of update | retrain or an edit | one record |
| Update latency | days, gated by evaluation | seconds |
| Exactness | lossy through interference | exact bytes, possibly stale |
| Serving cost | flat per query | scales with $k$ and context length |
| Deletion of one fact | not addressable | one row |
| Failure mode | confidently wrong after a change | confidently ignores the evidence |

## Privacy Is Not a Tie-Breaker

The usual claim is that retrieval is the private option because the data never enters the weights. That is partly true and it is the wrong axis.

Weights memorise. Membership inference works against them, and a fact baked into $\theta$ cannot be removed on request; the only remedy is a retrain, which does not guarantee removal. An index is also a second copy of the corpus, with its own access control, its own backups, and query logs recording what people asked for. Query logs are frequently more sensitive than the documents. Embeddings are not one-way functions either, so an index is not automatically the safer store.

What actually differs is blast radius and deletion latency. When a user invokes erasure, the retrieval side can honour it in seconds and prove it. The parametric side cannot honour it at all without a retrain. That asymmetry is the real privacy argument for retrieval, and note what it argues about: deletion, not storage. A team that adopts retrieval for privacy reasons and then keeps retrieval logs forever has moved the risk, not reduced it.

## The Contrarian Case: Retrieval Only Works If the Weights Agree to Listen

The current default advice is to stop teaching models facts and let retrieval supply them. I think that is correct about facts and wrong as a general claim about knowledge.

Start with a mechanism. In [arXiv:2602.08221](https://arxiv.org/abs/2602.08221), Ma, Zhang and Nie (2026) trace RAG unfaithfulness to what they call parametric suppression: in deeper layers, certain FFN layers overwrite context-sensitive representations with memorised priors. If that account holds, whether retrieved evidence gets used is a property of the weights, not of the retriever. You can build a clean index and serve it to a model whose priors outvote it.

The direct comparison points the same way, with an inconvenient detail. In [arXiv:2601.07054](https://arxiv.org/abs/2601.07054), Yang et al. (2026) compare unsupervised fine-tuning, supervised fine-tuning and RAG across three 7B-parameter models, on QASC and on more than 10,000 multi-hop questions built from 2024 Wikipedia events. They report that unsupervised fine-tuning gives limited gains over the base models, that RAG gives substantial and consistent improvements especially on temporally novel questions, and that supervised fine-tuning reaches the highest overall accuracy across models and datasets.

That last clause is the one people skip. My reading is a hypothesis, not their claim: supervised fine-tuning is not buying facts there. It is buying deference, the behaviour of answering from supplied evidence and saying when evidence is missing. Deference is cheap to train and expensive to fake, and no retriever supplies it.

So the contrarian position is narrower than "fine-tune your facts". It is this: retrieval is a transport mechanism, and transport is worthless if the model's priors veto the payload. The part of knowledge that belongs in weights is the disposition to use facts, not the facts themselves.

## What I Would Build

Three tiers, decided by churn and by what has to be deletable.

Volatile leaf facts, meaning prices, stock and policy versions, go to retrieval, with the document version stamped into the context and an eval that checks the model prefers the stamped evidence over its prior. Stable domain shape, meaning vocabulary, procedures, output formats and tool contracts, goes into weights through a small supervised pass, because that is where the flat latency bill and the generalisation are worth paying for. Anything with an erasure obligation goes to retrieval, because deletion latency is the one asymmetry that cannot be engineered around.

```python
def memory_route(updates_per_day, p95_budget_ms, must_delete):
    # thresholds are policy choices, not measurements
    if must_delete or updates_per_day > 24:
        return "retrieval"
    if updates_per_day < 1 and p95_budget_ms < 200:
        return "parameters"
    return "retrieval_plus_deference_training"
```

The thresholds are mine and they are policy, not findings. What matters is that both branches get evaluated, and that the evaluation separates two failures which look identical on a dashboard: the fact was never available, and the fact was available and the model overrode it. One is a retrieval bug. The other is a weights bug. Fixing the wrong one is how a team spends a quarter improving an index that was already correct.

The Tuesday price change is not solved by picking a side. It is solved by knowing which side is allowed to be wrong, and measuring that in production, on the questions that cost money.

## Sources

- "IGMiRAG: Intuition-Guided Retrieval-Augmented Generation with Adaptive Mining of In-Depth Memory", [arXiv:2602.07525](https://arxiv.org/abs/2602.07525) - Xingliang Hou, Yuyan Liu, Qi Sun et al., 2026. IGMiRAG: intuition-guided retrieval with adaptive mining of in-depth memory.
- "Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge", [arXiv:2601.07054](https://arxiv.org/abs/2601.07054) - Zhuoyi Yang, Yurun Song, Iftekhar Ahmed, Ian Harris, 2026. Fine-tuning versus RAG for multi-hop question answering with novel knowledge.
- "Whose Facts Win? LLM Source Preferences under Knowledge Conflicts", [arXiv:2601.03746](https://arxiv.org/abs/2601.03746) - Jakob Schuster, Vagrant Gautam, Katja Markert, 2026. Source preferences under knowledge conflicts in 13 open-weight models.
- "CoRect: Context-Aware Logit Contrast for Hidden State Rectification to Resolve Knowledge Conflicts", [arXiv:2602.08221](https://arxiv.org/abs/2602.08221) - Xuhua Ma, Richong Zhang, Zhijie Nie, 2026. CoRect: parametric suppression and hidden state rectification.
- "Memory Bank Compression for Continual Adaptation of Large Language Models", [arXiv:2601.00756](https://arxiv.org/abs/2601.00756) - Thomas Katraouras, Dimitrios Rafailidis, 2026. Memory bank compression for continual adaptation.
