---
title: "Data Curation as an Optimisation Problem"
description: "Deduplication, filtering and mixing are three ways of writing the same weight vector. Treat curation as a control problem with measurable effects."
pubDatetime: "2026-08-15T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Data
  - Curation
categories:
  - Math
---

You have 400 billion tokens of web text, a 1.3B parameter model, and a fixed compute budget. Someone on the team runs deduplication at a MinHash threshold of 0.8, cuts with a quality classifier at 0.6, mixes in 12 percent code, and starts training. The loss curve is smooth. The average across your eight downstream evaluations moves by less than the seed-to-seed spread. Nobody can say which of those four decisions did anything, and nobody can say whether a fifth decision, never tried, would have done more.

That is the ordinary state of data work. The pipeline is inherited, the constants arrive from someone else's ablation on a different corpus, and no stage is isolated. Because the stages run in sequence, their effects are confounded by construction. Deduplication changes what the filter sees. The filter changes what the mixer weighs. The mixer changes what the loss reads.

## A Corpus Is a Design Decision Nobody Signs

Write the weight vector down. Let $\mathcal{D}$ be the pool of candidate documents, and let each stage assign a non-negative weight $w(d)$ to document $d$. Deduplication sets $w$ to zero on all but a few copies of a duplicate group. Filtering sets $w$ to zero on anything below a score threshold. Mixing scales $w$ by domain. All three are the same object:

$$\hat{p}(x) = \frac{\sum_{d \in \mathcal{D}} w(d)\, \mathbb{1}\left[x \in d\right]}{\sum_{d \in \mathcal{D}} w(d)}$$

The corpus is $\hat{p}$. Whatever the model learns is a function of $\hat{p}$, and $\hat{p}$ is a vector you chose. Calling the result "the data" hides the authorship.

Once it is written that way, the question stops being which pipeline to run and becomes which objective to descend. The objective you want is the loss on a target distribution $q$ you cannot sample from, since the deployed distribution is the future. So you substitute a finite evaluation set $\hat{q}$, and the discipline of data curation becomes the approximation

$$\hat{p}^{*} = \arg\min_{\hat{p} \in \mathcal{P}} \; \mathbb{E}_{x \sim \hat{q}}\left[\ell(x; \hat{p})\right],$$

where $\mathcal{P}$ is the set of corpora reachable by your pipeline. Two things are wrong here on purpose. The evaluation set is finite, so it carries an idiosyncratic component. The reachable set $\mathcal{P}$ is described by whichever knobs you happened to implement. Both determine the answer more than the optimizer does.

## Filtering, Weighting and Deduplication Are Not Three Techniques

They differ in which part of $w$ they touch, and that difference predicts what they can and cannot do.

Filtering is monotone. It only removes support, so its ceiling is set by what was already in the pool, and a better filter cannot add a capability the pool lacks. Weighting is a reallocation. It changes relative share and can push a minority domain past its raw share, which means it can manufacture a capability the pool holds in small quantity. Deduplication is neither, in practice. It removes support at the copy level while keeping the group, so it reshapes the distribution inside a document family without touching other families.

The distinction matters because the three get tuned with the same instrument. A quality classifier is a filter, so raising its threshold cannot fix a domain imbalance. Continuing to raise it while a domain stays underweight is the most common wasted week in this work.

## Deduplication Is a Budget, Not a Boolean

Document-level exact and near deduplication is settled engineering. The live question is subdocument redundancy, and it is live because a fixed retention rule is a bad policy applied consistently.

Consider what "keep one copy" does to a corpus. Boilerplate appearing on four million pages gets one copy, which is correct. A specification quoted verbatim across thirty forum threads gets one copy, which discards thirty independent contexts where the quote was used to answer a question. Duplicate counts are heavy tailed and they correlate with content type, so a single constant is a systematic error with a direction.

H. Wang et al. (2026) make this the centre of their design. Their framework decouples duplicate detection from copy retention, then allocates an adaptive copy budget per duplicate group: more copies kept when a repetition is rare or short, more aggressive deletion when it is frequent or long ([arXiv:2608.03089](https://arxiv.org/abs/2608.03089)). The reported result is that models trained on the processed data land best overall among the settings they evaluated, on FineWeb-Edu and on a code-containing web corpus. The useful part is the framing rather than the number. Retention became a function of group statistics instead of a constant.

One such shape, not their formula:

```python
def copy_budget(group_size: int, mean_len: int) -> int:
    # keep more copies when the repetition is rare or short
    rarity = max(0, 3 - group_size.bit_length())
    brevity = max(0, 3 - (mean_len // 512).bit_length())
    return 1 << min(rarity + brevity, 4)
```

At trillion-token scale the same decision becomes a systems problem, and the constraint reshapes the design. Yang et al. (2026) report cumulative index maintenance of $O(N \log N \log_T K)$ against $\Theta(KN)$ for rebuilding a snapshot index $K$ times, 128 bytes per document against roughly 6.5 KB for a resident MinHash-LSH table, and one billion documents ingested in about 1.75 hours in a single process. They also report 11x and 35x throughput advantages over two comparison systems inside a ten-hour cap, and deployment on more than 30 billion documents ([arXiv:2608.28622](https://arxiv.org/abs/2608.28622)). Bore et al. (2026) take the online route, indexing only admitted documents and reporting recall of 0.94 to 0.97 relative to a strong LSH baseline at 30M documents, with up to 8x throughput ([arXiv:2606.03001](https://arxiv.org/abs/2606.03001)).

Those are observed results. My reading of why they matter is separate: once the honest cost of a curation policy is a cluster, the policy that survives is the cheap one, and that is an optimization pressure applied to the data itself rather than to the model.

## The Contrarian Case: Diversity Is the Wrong Variable

Almost every curation paper lists diversity among its objectives, and often it is the objective in the title. The evidence behind it is thinner than the frequency of the claim suggests.

Whetten et al. (2026) test it directly on self-supervised speech pre-training. They build subsets that optimize acoustic, speaker and linguistic diversity and report no clear improvement over random sampling on ASR. What does work is prioritizing the longest utterances, which beats the full dataset while using half of it and cuts pre-training time by 24 percent ([arXiv:2601.20896](https://arxiv.org/abs/2601.20896)).

My view is that diversity is a property of the pool you sample from, not of the subset you keep. Random sampling preserves the pool's diversity in expectation, up to sampling noise. Optimizing a diversity surrogate is therefore close to a no-op unless you are correcting a known skew, and when you do optimize it you usually trade something else away to buy a statistic. The variable that moved performance in that study was mundane: sequence length, and with it the amount of context and gradient per example. That mechanism is my hypothesis, not a finding of the paper, and it is one study in one modality, but it matches a pattern I have come to expect: the variables that move results rarely make good section headings.

## The Loop Between Data and Evaluation Closes on Itself

The moment selection uses validation feedback, curation stops being pre-processing and becomes a controller.

Cheng et al. (2026) build exactly that. They define a learnable value function over intrinsic quality, task relevance and distributional diversity, then refine the selection strategy with a feedback loop that reads proxy model performance across downstream tasks. They report competitive zero-shot results on ImageNet, CIFAR-100, MS-COCO and Flickr30K while using 36 percent of CC12M, ahead of their baselines by 1.0 percent on average ([arXiv:2602.05251](https://arxiv.org/abs/2602.05251)).

S. Wang et al. (2026) push the loop into the optimizer. Their selector scores a candidate by projecting the effective update it would induce onto a target direction taken from a stable, in-distribution proxy, so the criterion is not "is this document good" but "what does training on it do to the parameters." They report 4.7 percent additional compute overhead, GPT-2 Large and XL trained on 30B tokens outperforming a full 200B-token run on FineWeb and FineWeb-Edu, and a continued pre-training result on Qwen3-8B-Base reaching parity with a 3B-token run using 0.5B tokens ([arXiv:2602.05400](https://arxiv.org/abs/2602.05400)).

Both are real results and both architectures have the right shape. They are also a closed loop, and that is the part I would raise in a design review. The policy optimizes $\hat{q}$. Each iteration moves $\hat{p}$ toward whatever $\hat{q}$ rewards, including its idiosyncrasies. Contamination stops being an accident you screen for and becomes an outcome the search is incentivized to produce, because a document resembling the evaluation set scores well under any proxy derived from it. The fix is mechanical rather than clever: keep the selection proxy and the reported evaluation disjoint, and re-draw the proxy when it saturates.

There is a good counter-pattern from outside text. Zhu et al. (2026) curate heterogeneous battery aging data with agents that must ground every extracted value in verbatim evidence and abstain when nothing supports it, then gate admission through 26 schema, statistical and physical-plausibility rules, releasing a benchmark of 41 datasets from more than 25 institutions ([arXiv:2607.09762](https://arxiv.org/abs/2607.09762)). Abstention with a calibrated "no evidence" is a filter that reports its own uncertainty. Most text pipelines have no such state, which is why their errors stay silent.

## What I Would Instrument

One table, then the discipline.

| Intervention | What it changes | Signal to log |
|---|---|---|
| Deduplication | copies per duplicate group | retained fraction by group-size band |
| Filtering | support of $\hat{p}$ | rejection rate by score decile |
| Weighting | relative share of domains | token share against evaluation share |

The rule I would enforce: no curation stage ships without a retention rate and a marginal effect at fixed step count. Fixed step count matters because the usual comparison trains the smaller corpus for the same number of tokens, which conflates better data with more epochs over it. Train to a fixed token budget, then to a fixed step budget, and report both. If the win appears in only one, you have learned which effect you bought.

Two costs are honest. The measurement is expensive, since each stage needs its own run at a scale where the effect exceeds seed noise, which is often three seeds. And the answer is local, because optimal retention depends on what the model has already seen, so a threshold tuned on one corpus at one model size transfers worse than people assume. That is an argument for instrumenting a cheap proxy, not for skipping the measurement.

The reframe earns its keep. A corpus is not gathered, it is designed, and every filter, dedup rule and mixing weight is an intervention with a sign and a magnitude. Once the weight vector is written down, the argument about which pipeline is better turns into an argument about which objective you are approximating, and that argument has an answer.

## Sources

- "Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining", [arXiv:2608.03089](https://arxiv.org/abs/2608.03089) - Wang, H. et al. (2026). Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining.
- "PUFFER: Incremental Fuzzy Deduplication for Continuously Evolving Corpora", [arXiv:2608.28622](https://arxiv.org/abs/2608.28622) - Yang, X., Aldape, E. E., & Millidge, B. (2026). PUFFER: Incremental Fuzzy Deduplication for Continuously Evolving Corpora.
- "FOLD: Fuzzy Online Deduplication for Very Large Evolving Datasets via Approximate Nearest Neighbor Search", [arXiv:2606.03001](https://arxiv.org/abs/2606.03001) - Bore, N. et al. (2026). FOLD: Fuzzy Online Deduplication for Very Large Evolving Datasets via Approximate Nearest Neighbor Search.
- "A Study of Data Selection Strategies for Pre-training Self-Supervised Speech Models", [arXiv:2601.20896](https://arxiv.org/abs/2601.20896) - Whetten, R. et al. (2026). A Study of Data Selection Strategies for Pre-training Self-Supervised Speech Models.
- "TADS: Task-Aware Data Selection for Multi-Task Multimodal Pre-Training", [arXiv:2602.05251](https://arxiv.org/abs/2602.05251) - Cheng, G. et al. (2026). TADS: Task-Aware Data Selection for Multi-Task Multimodal Pre-Training.
- "OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration", [arXiv:2602.05400](https://arxiv.org/abs/2602.05400) - Wang, S. et al. (2026). OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration.
- "BatteryLake: Agentic, Physics-Grounded Curation of Heterogeneous Battery Aging Data and Benchmarking", [arXiv:2607.09762](https://arxiv.org/abs/2607.09762) - Zhu, T., Wang, H., & Wen, Y. (2026). BatteryLake: Agentic, Physics-Grounded Curation of Heterogeneous Battery Aging Data and Benchmarking.
