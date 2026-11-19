---
title: "One Number Never Tells the Whole Story"
description: "One accuracy number is a projection of a joint distribution over inputs, labels and costs, and whatever it drops tends to return as an incident."
pubDatetime: "2023-10-21T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Evaluation
  - Metrics
  - Engineering
categories:
  - AI
---

When a model moves from a notebook into a system, someone eventually asks for one number. It might be accuracy, F1, AUC, or a root-mean-squared error. The number then travels: into a slide, a pull request, a model card, a launch review. Nobody who reads it sees the test set, the label definitions, or the threshold that produced it. Most of the time that is fine. The trouble starts because the number looks like a property of the model, and it is not.

## A metric is a projection of a distribution

Formally, a metric is a scalar summary of a joint distribution over inputs, labels and costs. If $f$ is the model, $P$ the distribution of pairs $(x, y)$, and $\ell$ a loss, the quantity we claim to care about is the expected risk

$$ R(f) = \mathbb{E}_{(x,y)\sim P}\left[\ell(f(x), y)\right]. $$

What gets reported is an estimate on a finite sample,

$$ \hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i), y_i), $$

which is an estimate of risk under the test distribution, not under $P$. That distinction sounds like a cliché until you notice that both the loss $\ell$ and the distribution are choices, and the scalar carries neither of them. The number is a projection of a high-dimensional object onto one axis. Projections are useful, and they are lossy by construction. The lost part does not disappear. It moves somewhere later in the process.

## Accuracy is a cost matrix you never wrote down

Accuracy looks like the least opinionated metric available. It is the most opinionated one. Accuracy is expected risk under the zero-one loss, $\ell(\hat y, y) = \mathbb{1}[\hat y \neq y]$, which is exactly the assertion that every error costs the same. In a screening system, a missed positive and a false alarm rarely cost the same. In a ranking system, an error at the top of the list costs more than an error near the bottom.

The honest object is a cost matrix $C$, with $C_{ij}$ the cost of predicting class $i$ when the truth is class $j$, and the quantity to minimize is

$$ \mathbb{E}_{(x,y)\sim P}\left[C_{\hat y(x),\, y}\right]. $$

Accuracy is the special case where $C$ is zero on the diagonal and one everywhere else. Two models that differ by a point of accuracy can order differently once you multiply through by a real cost matrix, because the difference in expected cost is a linear functional of the confusion matrix, and the ordering of linear functionals depends on the direction you weight them. This is not a subtlety. It is the whole comparison, and the single number discards it. There is a second hidden choice as well: the threshold. A classifier produces scores, a decision requires a cut, and accuracy is the value at one cut, often inherited from a default rather than chosen.

| Metric | What it fixes | What it hides |
| --- | --- | --- |
| Accuracy | Uniform error cost, one threshold | The real cost matrix and the chosen cut |
| AUC | Ranking quality across all thresholds | Calibration, and the region you actually operate in |
| RMSE | A squared-error cost | The asymmetry between large and small mistakes, and the tails |

Each of these is a reasonable default and a bad universal. The table is not an argument against any of them. It is an argument against reporting one of them as if it were the thing.

## The test average is an average over a mixture

A test set is a mixture. If the data can be partitioned into groups $g$ with weights $w_g = P(g)$, the risk decomposes as

$$ R(f) = \sum_g w_g R_g(f), \qquad R_g(f) = \mathbb{E}\left[\ell(f(x), y) \mid g\right]. $$

Reporting $R$ discards every $R_g$ and every $w_g$. The algebraic consequence is easy to state and easy to forget: a change of $\Delta$ in a group's risk moves the total by $w_g \Delta$. Small groups can degrade severely while the aggregate improves, because the aggregate is weighted by mass and the harm is concentrated in the tail. A model that gets better on the majority and worse on a minority is a single number that went up.

The mixture itself is also a choice. A random split preserves the historical mixture, which is the mixture of the data you happened to collect, not the mixture of the traffic you will serve. So the aggregate is a summary of a sample drawn from a distribution that is itself an artifact.

```python
report = {}
for name, mask in slices.items():
    report[name] = {
        "n": int(mask.sum()),
        "metric": metric(y[mask], pred[mask]),
        "cost": cost_matrix[y[mask], pred[mask]].mean(),
    }
```

## Metrics correlate until the distribution moves

Inside one fixed benchmark, metrics tend to agree. Models that do better on accuracy usually do better on AUC, and the rank correlation across a leaderboard is often close to one. That agreement is a fact about the joint distribution of model-dataset pairs at a moment, not a law. Written as a correlation conditional on the distribution it is

$$ \rho_P(m_1, m_2) = \frac{\operatorname{Cov}_P(m_1, m_2)}{\sigma_P(m_1)\,\sigma_P(m_2)}, $$

and the subscript is the point. There is no reason for $\rho_P$ to equal $\rho_Q$ when the data moves from $P$ to $Q$. Under covariate shift the inputs move and a ranking metric may survive. Under label shift, or a shift concentrated near the decision boundary, calibration and accuracy can part company, and a model that wins on AUC can lose at the threshold you deploy.

The practical version is this: a proxy metric is a bet that the correlation survives your data. The bet is usually unstated, and it is tested by production rather than by the benchmark. Goodfellow, Bengio and Courville make the neighbouring point in their treatment of regularization. The training objective is a surrogate, and much of the machinery of learning exists to close the gap between the quantity optimized and the quantity wanted. A validation number is another surrogate, one layer further out.

## The contrarian case: leaderboards rank triples, not models

The standard framing is that a benchmark number is a property of a model, so a table of numbers is a comparison of models. I think that framing is wrong in a way that matters operationally. The number is a property of a triple: model, dataset and metric. In any real harness there is a fourth term, the implementation, which covers preprocessing, tokenization, decoding, thresholding and the evaluation script itself. A leaderboard ranks implementations of a triple. Nothing about that ordering is guaranteed to transfer to your data, your labels and your cost matrix, because those are different coordinates in the same triple.

The consequence is that reporting one number without a slice breakdown is not a summary. It is a loss of information, and the information does not evaporate. It reappears later as an incident, when the group that was averaged away turns out to be the group that matters. A scalar is a decision about what to ignore, made by whoever chose the metric, and inherited silently by everyone downstream.

I do not think this is mainly a failure of individual rigor. It is a failure of legibility. A leaderboard position is a single legible artifact; a slice table is not. Legible artifacts win attention, attention drives effort, and effort shapes what gets measured next. The familiar observation that a measure used as a target stops being a good measure is not mysticism. It is what happens when the projection is optimized instead of the thing it projected.

## The part that actually breaks is the pipeline

Sculley and colleagues described the surrounding system as the source of most of the debt in machine learning: glue code, pipeline jungles, configuration debt, undeclared consumers, and the rule that changing anything changes everything. The model is a small component of that system, and the metric is computed on a frozen snapshot of it.

This is why a system can break while its number holds. The evaluation set is a fixture; production is a stream. An upstream schema changes, a join key drifts, a feature computation is refactored to be faster and slightly different, and the model's accuracy on the fixture is unchanged because the fixture no longer resembles the input. The metric was a property of a snapshot, and the snapshot stopped describing the system.

The same logic applies to the precision of the estimate. A mean over $n$ examples has a standard error, and for a bounded loss it is roughly

$$ \mathrm{SE} \approx \sqrt{\frac{\hat{R}\,(1-\hat{R})}{n}}, $$

which is small enough at benchmark scale to make small differences look decisive, and large enough at slice scale to make them meaningless. A slice with few examples has a wide interval, and a wide interval reported as a point estimate is a claim the data does not support.

So I report the metric together with its cost matrix, its threshold, the slices with their counts, and an interval. It is more work to read. It is also the difference between a summary and a decision made on someone else's behalf. One number never tells the whole story, and the part it leaves out is usually the part that pages you.

## Sources

- D. Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html
- Ian Goodfellow, Yoshua Bengio and Aaron Courville (2016). *Deep Learning*, chapter 7. https://www.deeplearningbook.org/contents/regularization.html
