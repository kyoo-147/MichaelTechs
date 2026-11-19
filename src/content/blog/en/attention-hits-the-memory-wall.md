---
title: "Attention Hits the Memory Wall"
description: "Attention kernels spend more time moving score matrices between HBM and SRAM than doing arithmetic, which is why tiling beats approximation."
pubDatetime: "2024-02-17T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - FlashAttention
  - GPU Memory
categories:
  - Math
---

A 16,384-token sequence, one attention head, fp16. The score matrix holds $16384^2$, or 268 million, entries. That is 512 MiB before softmax writes anything, and training keeps more than one such buffer alive. The arithmetic is modest by comparison: about 69 GFLOP for the head at $d = 64$. Most of the time goes to carrying intermediates between HBM and SRAM and back.

An A100 has about 312 TFLOP/s of fp16 tensor throughput and 2.0 TB/s of HBM bandwidth. Divide one by the other: the machine wants roughly 150 floating point operations for every byte it fetches. Below that ratio the tensor cores wait. Attention at long sequence lengths sits well below it, and that single fact explains most of the kernel engineering of the last few years.

## The matrix that never needs to exist

$\mathrm{softmax}(QK^\top/\sqrt{d})V$ is a composition, and the $N \times N$ score matrix is an intermediate rather than a result. Standard libraries materialise it because each step is a separate kernel: one GEMM writes the scores, a softmax kernel reads and writes them, a second GEMM reads the probabilities. Together they push $O(N^2)$ bytes through HBM several times, when the output only needs $O(Nd)$.

Materialising is not a mistake. It is what you get when every operation is a library call, and it is right when the matrix fits in cache. The problem begins when $N^2$ elements exceed the cache and every pass pays full bandwidth.

Saha and Ye's 2024 analysis makes the cost precise. They treat attention as a two-level memory problem, count accesses to slow memory, and show that FlashAttention's $N^2d^2/M$ I/O count is optimal for any SRAM size $M \ge d^2$, up to constant factors. The lower bound holds even if the matrix multiplications use fast algorithms. In the regime where GPUs operate, the tiled exact algorithm is not a heuristic.

## Arithmetic intensity decides what fast means

Arithmetic intensity is FLOPs divided by bytes moved, $I = F/W$. A kernel's time is bounded below by

$$T \;\ge\; \max\left(\frac{F}{P},\ \frac{W}{B}\right),$$

where $P$ is peak throughput and $B$ is bandwidth. The ratio $P/B$ is the machine balance. When $I < P/B$ the second term dominates and the kernel is bandwidth bound. When $I > P/B$ the first term dominates and it is compute bound.

The useful consequence is that flops are free on the wrong side of the line. Removing operations from a bandwidth-bound kernel does not shorten it, because the time was never spent in arithmetic. Adding operations can be a win if the additions remove bytes. Tiling spends SRAM to keep the intermediate on chip; recomputation spends FLOPs to avoid storing it. Both trade against bandwidth, and either can make an algorithm with more work finish first.

Capacity is a separate axis. A kernel that runs out of memory is not slow, it is impossible, and the fix is a smaller working set. A kernel that runs but misses bandwidth needs fewer bytes. Confusing the two leads to the wrong optimisation, usually an approximation that costs quality without touching traffic.

## What the naive kernel moves

Take one head with $N$ queries and keys, head dimension $d$, fp16. Materialising the scores and the softmax output moves $8N^2$ bytes: write $S$, read $S$, write $P$, read $P$, at two bytes per element. The fused arithmetic is $4N^2d$ FLOPs, from the two matrix products. The ratio is

$$I_{\mathrm{naive}} = \frac{4N^2d}{8N^2} = \frac{d}{2}.$$

With $d = 64$ that is 32 FLOP per byte, about five times below the A100 balance. Notice what the ratio does not contain: $N$. Longer sequences make the kernel slower in absolute terms but do not change its intensity. The quadratic term hurts because there are more bytes, not because each byte buys less arithmetic.

Fusing the softmax into the score GEMM helps, but only by a factor of two. If the scores still reach HBM once, the traffic is $4N^2$ bytes and the intensity is $d$. Still below the line. The only way to raise intensity is to stop writing the matrix at all.

## Tiling changes the denominator

Tiling keeps a block of queries in SRAM and streams keys and values past it. The score block is computed, normalised and multiplied by the value block while it is still in registers or shared memory, then discarded. The output tile accumulates across the stream. Softmax needs a correction because the maximum is unknown until every key has been seen, so the kernel carries a running maximum and rescales the accumulator:

$$m_i' = \max(m_i, \tilde m_i), \qquad
l_i' = e^{m_i - m_i'} l_i + e^{\tilde m_i - m_i'} \tilde l_i, \qquad
O_i' = e^{m_i - m_i'} O_i + e^{\tilde m_i - m_i'} \tilde O_i,$$

where the tilde quantities come from the current key block. Each rescale is a multiply on a tile already in SRAM. The exponentials are evaluated once per score block, not once per pass over HBM.

```python
for q in tiles(Q, Br):                  # stays in SRAM
    m = full(Br, -inf); l = zeros(Br); O = zeros(Br, d)
    for k, v in zip(tiles(K, Bc), tiles(V, Bc)):
        s = q @ k.T / sqrt(d) + mask    # score block, never stored
        m_new = maximum(m, s.max(axis=1))
        p = exp(s - m_new[:, None])
        alpha = exp(m - m_new)
        l = alpha * l + p.sum(axis=1)
        O = alpha[:, None] * O + p @ v
        m = m_new
    write(O / l[:, None])               # normalise at the end
```

With a query tile of $B_r$ rows, each element of $K$ and $V$ is read once per query tile, so traffic falls to $4N^2d/B_r$ bytes while the FLOP count stays at $4N^2d$. The intensity becomes

$$I_{\mathrm{tiled}} \approx B_r .$$

At $B_r = 128$ the intensity is 128 FLOP per byte, close to the A100 balance of about 150. The tile height, not the head dimension, sets the ratio, and the tile height is limited by SRAM. A 128 by 64 fp16 query tile is 16 KiB. The output tile and the key and value tiles each take 16 KiB at matching block sizes. What remains of the 192 KiB per SM goes to double buffering and the running statistics.

Nothing in that loop approximates the softmax. The result is the same function, computed in a different order, with the intermediate matrix never existing in HBM.

| Kernel | HBM bytes | FLOPs | Intensity |
| --- | --- | --- | --- |
| Scores and probabilities materialised | $8N^2$ | $4N^2d$ | $d/2$ |
| Softmax fused, scores still stored | $4N^2$ | $4N^2d$ | $d$ |
| Tiled, nothing materialised | $4N^2d/B_r$ | $4N^2d$ | $B_r$ |

## Recomputation buys bandwidth with flops

The backward pass needs the probabilities to compute gradients. The obvious choice is to save them from the forward pass, which costs $4N^2$ bytes to store and load. The alternative recomputes the score block from $Q$ and $K$ inside the backward tile. That adds $2N^2d$ FLOPs and $N^2$ exponentials, and removes the $4N^2$ byte round trip.

For the matrix work the exchange rate is $d/2$ FLOPs per byte saved. At $d = 64$ that is 32, while the machine can issue about 150 FLOPs in the time it takes to move one byte. Recomputing is cheaper than storing as long as the kernel stays below the balance point. The counterintuitive part: the backward pass with more FLOPs is the faster one.

The trade has a boundary. If the sequence is short and the batch large, the score matrix may fit in cache, recomputation only adds work, and the stored version wins. If arithmetic is already the bottleneck, extra FLOPs cost time that no saved byte can repay. The answer depends on $N$, $d$, the SRAM size and the batch shape, which is why kernels are benchmarked across shapes rather than argued about abstractly.

## The contrarian case: exact wins because the problem was mis-stated

The standard framing says attention is quadratic in compute, so the fix is to approximate it: sparsify the matrix, factorise it, replace the softmax with a kernel. That framing was reasonable when the compute term dominated. On current accelerators it names the wrong bottleneck.

An approximate method reduces FLOPs. If the kernel is bandwidth bound, that reduction does not shorten the critical path. Many approximations also add a selection step: compute scores, find the top $k$, gather the matching values. The selection pass reads the scores, which is the same $N^2$ traffic the exact kernel avoided, and the gather is irregular enough to waste bandwidth. The arithmetic gets cheaper and the kernel gets slower.

The FlashAttention papers report wall-clock gains of 2 to 4 times over optimised baselines with no approximation at all, and FlashAttention-2 reaches 50 to 73 percent of peak fp16 throughput on an A100. Those are published measurements, not a claim that exactness is always faster. The claim is narrower: when the bottleneck is bytes, reducing operations is not the lever.

The approximation program is not foolish. Zheng and coauthors treat the gap between linear-time random-feature attention and softmax attention as the object of study, decompose it with control variates, and shrink it while keeping linear complexity. That work makes the cost explicit: you buy asymptotics with a quality gap, and closing the gap is its own research problem.

My reading is that for prefill and training at moderate sequence lengths, the exact fused kernel should be the default, and approximation should be reserved for cases where capacity or extreme $N$ makes quadratic traffic dominant even after tiling. The exact kernel has a practical property that matters: it computes the same function, so no evaluation has to be rerun to prove quality survived.

## What to check before tuning anything

Start with the roofline arithmetic for your shape. Compute $F$ and $W$, divide, and compare with $P/B$ on the device. If the ratio is far below the balance, instruction-level tuning will not save you; the kernel needs fewer bytes.

Then separate capacity from bandwidth. An out-of-memory error is a working-set problem: smaller tiles, a sharded sequence, or less state. A kernel that runs at 40 percent of peak bandwidth is a traffic problem: fuse passes, keep intermediates in SRAM, and confirm the tile fits alongside double buffering.

Decode is a different regime and needs its own numbers. Generating one token reads the whole KV cache and does $O(Nd)$ arithmetic against it, so intensity stays low however the attention matrix is tiled. There is no $N^2$ intermediate to eliminate. The levers are fewer bytes per cached token, sharing keys and values across heads, and batching requests so weights are reused. Tiling helps prefill and training because they have a quadratic intermediate to remove. It does not create reuse in decode.

Measure achieved bandwidth, not only elapsed time. A kernel at 80 percent of HBM peak is close to done even if the FLOP counter looks low. A kernel at 30 percent has a layout, occupancy or tile-size problem, and the fix lives in the memory system.

None of this requires an approximation. It requires knowing which resource is scarce.

## Sources

- Barna Saha and Christopher Ye, [The I/O Complexity of Attention, or How Optimal is Flash Attention?](https://arxiv.org/abs/2402.07443) (2024).
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022).
- Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023).
- Lin Zheng, Jianbo Yuan, Chong Wang and Lingpeng Kong, [Efficient Attention via Control Variates](https://arxiv.org/abs/2302.04542) (2023).
