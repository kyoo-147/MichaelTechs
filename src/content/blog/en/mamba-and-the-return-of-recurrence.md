---
title: "Mamba and the Return of Recurrence"
description: "A state space layer is a linear recurrence whose parameters read the input, which breaks the convolution and rewrites what a model can remember."
pubDatetime: "2024-03-16T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - State Space Models
  - Recurrence
categories:
  - Math
---

## A cache with no ceiling

A decoder with $L$ layers, $H$ heads and head dimension $d$ holds, per token, two vectors of size $Hd$ in every layer. The cache is therefore

$$M_{\text{kv}} = 2 \cdot B \cdot L \cdot H \cdot d \cdot n \ \text{floats}$$

for a batch of $B$ sequences at length $n$. Nothing there can be made sublinear in $n$ without changing what the layer does, because the next query is compared against every key ever written. That memory is the attention operation, stored.

That is the pressure behind the state space layer. Instead of keeping every past token addressable, keep a fixed-size state and update it as tokens arrive. The state has the same shape at token ten and token one hundred thousand. The cost per step does not move.

## The layer is a recurrence

Write the layer the way control theory writes it:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t$$

with $x_t$ a scalar input for one channel, $h_t \in \mathbb{R}^{N}$ the state, and $N$ small. Bars denote discretisation. A continuous system $\dot{h} = Ah + Bx$ sampled at step size $\Delta$ becomes the above under a zero-order hold, with

$$\bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}\big(\exp(\Delta A) - I\big)\,\Delta B .$$

So far this is a linear recurrence with constant transition. Unroll it. If $\bar{A}$, $\bar{B}$ and $C$ do not depend on $t$, then

$$y_t = \sum_{k \le t} C \bar{A}^{\,t-k} \bar{B}\, x_k = (K * x)_t, \qquad K = \big(C\bar{B},\; C\bar{A}\bar{B},\; C\bar{A}^2\bar{B},\; \dots\big).$$

The output is a convolution with a kernel of length $n$, computed from the parameters rather than learned. That is the S4 result, and it is why the early papers were fast: one FFT takes the whole sequence at $O(n \log n)$ with no recurrence visible in the implementation. Time invariance is what buys the kernel. The same matrix power appears at every offset.

At inference the same object costs almost nothing. Keep $N$ floats per channel, read one token, write one token, emit one output. There is no cache to grow.

## Why a selective scan is not a convolution

The recurrence above has a property that turns out to be a defect. The map from input to output is fixed. Every token is filtered by the same kernel, so the layer cannot decide that this token matters more than the last one, and the state decays at a rate chosen during training rather than at a rate the content suggests.

Mamba's move is to make the step size and the input and output maps functions of the token:

$$\Delta_t = \text{softplus}(W_\Delta x_t), \qquad B_t = W_B x_t, \qquad C_t = W_C x_t .$$

Now the discretisation happens per step, so $\bar{A}_t$ varies with $t$, and the unrolled sum becomes

$$y_t = \sum_{k \le t} \Big( \prod_{j=k+1}^{t} \bar{A}_j \Big) \bar{B}_k x_k .$$

Read the product carefully. It is not a power of one matrix. It depends on both $t$ and $k$, and through them on the data. There is no single kernel, so no convolution, so the FFT route is gone. The layer still computes the same kind of sum, but the sum has to be evaluated as written.

This is what people blur when they call a selective state space model a convolution with a learned kernel. A convolution's kernel is a property of the weights. Here the kernel is a property of the sentence.

One thing survives: for a fixed input the recurrence is still linear in the state. Selectivity changes the coefficients, not the algebra.

## The scan is the trick

A sequential loop over $n$ tokens has depth $n$. On a GPU that is a serial chain with the parallelism sitting idle, which is why recurrent layers were painful.

The fix is to notice that each step is an affine map, $h \mapsto ah + b$, and composing two is another affine map:

```python
# The same recurrence, written twice.

# 1. Reference form: one step at a time, depth n.
h = zeros(N)
for t in range(n):
    A_bar, B_bar, C = step_params(x[t])   # input-dependent when selective
    h = A_bar * h + B_bar * x[t]
    y[t] = (C * h).sum()

# 2. What the scan evaluates: a prefix product over these pairs,
#    where a step t contributes (A_bar_t, B_bar_t * x[t]).
def combine(left, right):        # left happens earlier in time
    a1, b1 = left
    a2, b2 = right
    return (a2 * a1, a2 * b1 + b2)
```

`combine` is associative. Applying it left to right gives the same answer as applying it in a balanced tree, so the prefix can be computed in $O(\log n)$ depth. A naive tree costs $O(n \log n)$ work. A work-efficient scheme that reuses partial results brings that down to $O(n)$.

That is the whole algorithmic content of the name. The scan does not approximate the recurrence. It computes the identical values in a different order, which is why a model trained with the scan deploys as a plain loop.

What the fast implementations add is not mathematics but less memory traffic. Materialising the full prefix in global memory would throw the advantage away, so the kernels tile the sequence, hold the state on chip, and recompute rather than store. My reading is that the scan is a modest idea and the kernel is the hard part: an $O(n)$ algorithm with bad data movement loses to an $O(n^2)$ algorithm with good data movement at the sizes people actually run.

## What the matrix was doing

Remove the $n \times n$ matrix and three capabilities leave with it.

The first is full-resolution routing. In attention, token 900 reads token 3 directly, with a weight computed from their contents alone, and nothing has to survive a compression step. In a state space layer every token is written into a state of width $N$ and every token reads back out of it. The channel is narrow and shared.

The second is a memory that grows with the sequence. The matrix has $n^2$ entries, most doing very little, but the ones that matter are available. A state of $N$ numbers holds $N$ numbers' worth of anything.

The third is inspectability. You can plot the attention matrix and see what the layer did. A scan leaves no such object.

The empirical work in this period is a catalogue of what the bottleneck costs. Jelassi and coauthors prove that a two-layer transformer can copy strings of exponential length while a model with a fixed-size latent state is fundamentally limited by that state, and they find the gap on synthetic copying tasks and on pretrained models asked to copy and retrieve from context. Vision Mamba answers a narrower version: one forward scan over flattened patches cannot see the patch that follows, so the authors scan backwards as well and add position embeddings. They report 2.8 times the speed of DeiT with 86.8 percent less GPU memory on batched feature extraction at 1248 by 1248. LocalMamba argues that raster order stretches the distance between neighbours, and searches for a scan pattern per layer.

Put those together and a structural point falls out. Once the state is a bottleneck, the order in which you feed tokens stops being bookkeeping and becomes an architectural decision, the way kernel shape used to be for a convolutional network. Attention never faced this. Permuting tokens permutes rows and columns and changes nothing else. A scan is not permutation invariant, and that asymmetry is a design surface with no counterpart in the transformer.

| | Attention | LTI state space | Selective state space |
|---|---|---|---|
| Map from input to output | input-dependent, per pair | fixed | input-dependent, per step |
| Training algorithm | matmul over $n \times n$ | FFT convolution | associative scan |
| Inference memory | $O(n)$ cache | $O(1)$ state | $O(1)$ state |
| Effective memory | grows with $n$ | fixed $N$ | fixed $N$, content-gated |
| Characteristic failure | cost at large $n$ | cannot select content | bounded verbatim recall |

## The contrarian reading: capacity, not speed

The standard framing is that state space models are attention with linear cost, and the conclusion people draw is that this is a cheaper drop-in. That framing is wrong in practice, in two ways.

The first is that the asymptotic argument does not apply where most inference happens. $O(n)$ beats $O(n^2)$ for large $n$, and at the lengths that dominate production traffic both fit comfortably in memory and the attention kernels are very well tuned. Note where the vision results above report their wins: high resolution images, where $n$ is large. At short $n$ the linear term is not binding and the comparison is a wash. A complexity class is a claim about a regime, not a property of a model.

The second matters more. The layer no longer has anywhere to put unbounded information. A fixed state is a hard ceiling, and the ceiling $N$ is a hyperparameter chosen before training. Copying a context of length $n$ requires on the order of $n$ bits somewhere, and no constant $N$ does that as $n$ grows. That is a capacity argument, not a tuning problem. The copying result is an existence proof of the ceiling rather than a benchmark loss. You cannot fix it by training longer or with a better kernel.

I would also quibble with the word return. The recurrent layer that fell out of favour around 2016 was slow because the recurrence was serial in wall-clock time. The scan removes precisely that. What returned is the state, not the sequential dependency: training runs a parallel prefix, inference runs a loop, and those are different objects sharing parameters. Calling this a return of recurrence is defensible as description and misleading as diagnosis. Nothing that made the old RNNs hard came back.

## What I would measure

Fix the compute budget and sweep the state size $N$. My hypothesis is that copying and retrieval accuracy against sequence length is a cliff near a small multiple of $N$ rather than a gentle slope, and that the cliff moves predictably with $N$. I have not run this. It is where I would start, because it tests the capacity story instead of the speed story.

Compare scan orders at identical parameter count and identical FLOPs. LocalMamba's per-layer search suggests the gap between a raster and a windowed scan matters at fixed budget. If that holds, part of what gets reported as architectural improvement is really a search result, and ablations that fix the scan order measure the wrong thing.

My expectation, not a finding, is that the configuration which survives production keeps a few attention layers for exact retrieval and uses the scan for the rest. The capacity argument says something has to hold the verbatim content. The cost argument says it should be as little of the model as possible. A hybrid is the obvious compromise, and the open question is where the boundary belongs, which is about a task's recall requirements rather than about the architecture.

## Sources

Jelassi, Samy, David Brandfonbrener, Sham M. Kakade, and Eran Malach. "Repeat After Me: Transformers are Better than State Space Models at Copying." [arXiv:2402.01032](https://arxiv.org/abs/2402.01032), 2024. https://arxiv.org/abs/2402.01032

Zhu, Lianghui, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model." [arXiv:2401.09417](https://arxiv.org/abs/2401.09417), 2024. https://arxiv.org/abs/2401.09417

Huang, Tao, Xiaohuan Pei, Shan You, Fei Wang, Chen Qian, and Chang Xu. "LocalMamba: Visual State Space Model with Windowed Selective Scan." [arXiv:2403.09338](https://arxiv.org/abs/2403.09338), 2024. https://arxiv.org/abs/2403.09338
