---
title: "On-Device Inference Budgets"
description: "A phone's token rate is set by bytes moved and heat shed, not flops, so precision, prompt length and p99 timing decide what ships."
pubDatetime: "2026-07-25T09:00:00.000Z"
locale: en
author: Michael
featured: false
tags:
  - Edge AI
  - Inference
categories:
  - Math
---

A 7B model at 4 bits occupies 3.5 GB. A phone with 4 GB of usable memory and 20 GB/s of bandwidth can hold it, and cannot run it quickly. Every decoded token reads the whole weight tensor once, so the floor on token time is 3.5 GB divided by 20 GB/s, or 175 ms. That is 5.7 tokens per second before kernel inefficiency, prompt processing or thermal throttling.

The flops do not enter that calculation. Decode is a matrix-vector product per token, about four operations per byte fetched. A device with 10 TOPS of integer throughput and 20 GB/s of memory can issue 500 operations in the time it takes to move one byte, so a kernel at 4 operations per byte is two orders of magnitude from needing the arithmetic units.

Three budgets bind an on-device model: bytes moved per token, joules per second the enclosure can shed, and the variance of both across a session. Model size, precision and pipeline shape are ways of spending them. Tune against FLOPs instead and you get something that passes on a cold phone and fails in a pocket.

## Decode reads the whole model, every token

A decoder generating one token touches every weight once. Counting arithmetic, the per-token work for $N$ parameters is about $2N$ FLOPs. Counting bytes, it is $Nw$ for $w$ bytes per weight. The ratio is

$$I = \frac{2N}{Nw} = \frac{2}{w}.$$

There is no term for context length, batch size or model size. Intensity depends only on how many bytes each weight occupies.

| Weight precision | $w$, bytes per weight | Decode intensity, $2/w$ | Share of a 500 FLOP/byte balance |
| --- | --- | --- | --- |
| FP16 | 2 | 1 | 0.2 percent |
| INT8 | 1 | 2 | 0.4 percent |
| INT4 | 0.5 | 4 | 0.8 percent |
| 3-bit | 0.375 | 5.33 | 1.1 percent |

No precision in that table approaches the ridge, so the bound is better written as a division. With $B$ bytes per second of usable bandwidth, the fastest possible token time is

$$t_{\text{tok}} \ge \frac{Nw}{B}.$$

Li and coauthors push this to the limit on an embedded FPGA. They describe typical edge hardware as 4 GB of capacity with under 20 GB/s of bandwidth, note that a 7B model at 4 bits needs 3.5 GB, and report roughly 5 tokens per second for LLaMA2-7B while reaching 85 percent of the theoretical bandwidth of the board's DDR4. Those figures give 175 ms per token as the floor. They land near it. The kernel sits against the wall, and the wall is the part they bought.

Prefill behaves differently. It multiplies a block of tokens against the weights, so each weight byte is reused across the block. Decode reuses nothing. Murthy and coauthors measure the split on a Snapdragon 8 Elite: the NPU gives 1.64x over CPU on prefill and 1.18x on decode. The accelerator helps where there is arithmetic to accelerate.

Context makes the ratio worse. Boroujeni and coauthors describe the key-value cache as growing linearly with context and often dominating decoding cost. Each cached element is read once per token and contributes one multiply-add, so at fp16 the cache runs near one FLOP per byte. It is a second bandwidth stream with a worse ratio than the weights.

## Thermal headroom is the second bandwidth

Every number above assumes the memory clock holds. On a phone it holds for a while. There is no fan, the case is the heatsink, and the SoC reduces frequency rather than exceed a skin temperature limit. A first-order model shows the shape:

$$\dot{Q} = \frac{T - T_{\text{amb}}}{R_{\text{th}}}, \qquad P_{\text{ss}} = \frac{T_{\max} - T_{\text{amb}}}{R_{\text{th}}}.$$

$P_{\text{ss}}$ is the power the device can dissipate indefinitely. Anything above it charges the thermal capacitance until the limit arrives, and the energy available beforehand is roughly $C_{\text{th}}(T_{\max} - T_{\text{amb}})$. Divide by joules per token and you have the tokens a burst can afford. Both constants belong to the enclosure, not the model, which is why the same network behaves differently in a tablet and a phone.

I have not measured them on a specific device and would not quote them from a spec sheet. The consequence is what carries: sustained token rate is bounded by $P_{\text{ss}} / E_{\text{tok}}$, so lowering energy per token raises the sustained rate even when peak bandwidth is unchanged. That is a different optimisation from making the kernel faster.

The measurements I have read fit this. Murthy and coauthors run 100 iterations for thermal stability and report that NPU placement lowers steady-state temperature by 10.47 degrees Celsius and energy by 2.52x, which keeps an always-on workload out of throttling. Gong and coauthors treat the resulting variation as the problem: thermal frequency changes on a Jetson Orin Nano and a Mi 11 Lite produce large swings in inference latency, and their controller scales CPU and GPU frequency to hold the variation down.

A hypothesis I hold without proof: throttling hurts bandwidth-bound kernels more than the peak FLOP figure suggests, because memory and compute clocks are governed together and a lower memory clock raises token time almost proportionally. If that is right, the effective bandwidth of a hot phone is a fraction of its cold bandwidth, and any budget computed cold is optimistic.

## Quantisation is a bandwidth trade priced in bits per weight

If bytes per token are the constraint, bits per weight are the price. Song and coauthors evaluate models from 0.5B to 14B across seven post-training quantisation methods on commodity hardware. Three findings are worth carrying into a design review. Heavily quantised larger models consistently beat smaller high-precision ones, with a threshold near 3.5 effective bits per weight. Resource use scales roughly linearly with bits per weight. And as the model shrinks, the binding constraint moves from data movement to computational latency.

The third finding changes plans. Quantisation buys bandwidth until arithmetic and launch overhead dominate, and then more bits saved stop helping. The 3.5 threshold is empirical and hardware-specific, so treat it as a hypothesis to test on the target device.

Below four bits the trade gets less clean. Weights still cross the bus at the lower width, so the saving is real, but the kernel has to unpack them into a format its datapath accepts. If the NPU is integer-only at 8 bits, 4-bit weights arrive cheap and are expanded on chip, spending registers and ALU to save traffic. That is usually right on a bandwidth-bound step and wrong on a compute-bound one.

Accuracy is where the choice gets expensive. Boroujeni and coauthors measure how large it is for the key-value cache. Instead of one fixed precision they assign bit width per token from 2, 4, 8 and 16 bits, and on one SmolLM-360M setting they report 17.75 percent lower decoding latency than static cache quantisation with 7.60 points higher accuracy, within 0.30 points of fp16. One benchmark, one model family. The structural point travels: cached tokens do not carry uniform value per byte, so a uniform bit width spends budget where it was not needed.

## The contrarian case: shrinking the parameter count is often the wrong first move

The reflex when a model does not fit is to pick a smaller model. On a bandwidth-bound device that reflex is often backwards, because the budget is bytes per request and parameter count is one factor in it, not the thing itself.

Two configurations from the arithmetic above. A 3B model at 8 bits moves 3.0 GB per decode step. A 7B model at 4 bits moves 3.5 GB. The larger model costs 17 percent more bytes and, per the threshold Song and coauthors report, is the one likely to score better. Choosing the 3B because it has fewer parameters is choosing worse quality per byte. Precision first and parameter count second is the order the budget implies.

Bytes per request is wider than weights. A request carries the prompt, the cache and any retrieved context. A 1B model with a 4,000-token prompt and a two-turn tool loop can move more bytes than a 7B model at 4 bits with a 200-token prompt and a single turn. That is arithmetic from the formulas above rather than a measurement, but the direction is not in doubt: token count multiplies the same bandwidth term the weights do.

Here I differ from the usual framing of on-device model selection. Cao and coauthors compare 16 models across five tasks with a ratio combining accuracy, throughput, memory and latency, and find that 0.5B to 3B models score best across all five. That is a real result, and it supports small models for a fixed task in a fixed pipeline. It does not establish that a small model is cheaper per unit of quality for an arbitrary request, because the metric holds request shape constant. Change the shape and the ranking can move.

My position is to fix the request, then the precision, then the parameter count. Most teams work in the reverse order and pay for it with a retrain when the pipeline changes.

## Distillation moves the budget from weights to tokens

Distillation sits on the same ledger. Kerpicci and coauthors distil speech representation models for edge deployment and report close to 75 percent reduction in model size with 0.1 percent accuracy and 0.9 percent equal error rate degradation against the full-size model. That is a weight-budget saving of the kind that makes a model fit at all.

The more interesting case is when the student also needs fewer bytes per request. McGinness crops handwritten exam responses with bounding boxes before passing them to models between 4B and 72B parameters, and reports that cropping improves grading accuracy and reduces computational cost measured in FLOPs across the range. Nothing is distilled there, yet the shape is the same: a change upstream of the model cut both the error and the work. My reading is that on a bandwidth-bound pipeline a tighter input is a budget move with no accuracy debt, since fewer pixels means fewer bytes into the vision encoder.

The failure mode is worth naming. A distilled model that still reads a 4,000-token prompt has moved the problem rather than solved it, because the token stream now dominates the bytes. The question to ask of any compression step is not how many parameters it removed but how many bytes per request it removed.

## Percentiles are the deliverable

An average latency is a statement about a device that does not exist. The device you ship has a cold state and a hot state, and the hot state is what users meet after a minute of use. A mean over both describes neither.

For a normal distribution the 99th percentile sits about 2.33 standard deviations above the mean:

$$p_{99} \approx \mu + 2.33\sigma.$$

Latency is not normal. It is right-skewed, with a long tail from scheduling, memory contention and frequency changes, so the approximation understates the tail. That asymmetry is the argument for reporting percentiles directly.

```python
import statistics, time

def measure(step, seconds=600, warmup=30):
    """step() runs one request end to end."""
    for _ in range(warmup):
        step()
    samples, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        start = time.perf_counter()
        step()
        samples.append(time.perf_counter() - start)
    samples.sort()
    n = len(samples)
    def pct(q):
        return samples[min(n - 1, int(q * n))]
    head = statistics.mean(samples[:100])
    tail = statistics.mean(samples[-100:])
    return {"n": n, "p50": pct(0.50), "p95": pct(0.95),
            "p99": pct(0.99), "drift": tail / head}
```

The loop matters less than the protocol. Warm up long enough to leave the burst regime, run for at least ten minutes, and report the first minute separately from the last. A model whose p50 is flat and whose p99 doubles between minute one and minute ten is throttling, and no kernel change fixes it. A model whose p99 is flat and whose p50 rises is contending for memory with something else on the device. Those are different failures with different fixes, and the average cannot tell them apart.

Three numbers are worth writing down before any model is chosen: bytes per token, joules per token, and p99 latency at thermal steady state. A change that does not move one of the three is not an optimisation for this device, whatever it does to the FLOP count.

## Sources

- Jinghan Cao, Yu Ma, Xinjin Li, Qingyang Ren and Xiangyun Chen, [arXiv:2603.21389](https://arxiv.org/abs/2603.21389), 2026.
- Lachlan McGinness, [arXiv:2607.18767](https://arxiv.org/abs/2607.18767), 2026.
- Jindong Li, Tenglong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang and Yi Zeng, [arXiv:2502.10659](https://arxiv.org/abs/2502.10659), 2025.
- Qingyu Song, Rui Liu, Wei Lin, Peiyu Liao, Wenqian Zhao, Yiwen Wang, Shoubo Hu, Yining Jiang, Mochun Long, Hui-Ling Zhen, Ning Jiang, Mingxuan Yuan, Qiao Xiang and Hong Xu, [arXiv:2505.15030](https://arxiv.org/abs/2505.15030), 2025.
- Aryama V Murthy, Yashas N Kotre, Prathmesh Sharma, Pragya Mishra, Sanjith Ganapathi and Priyesh Shukla, [arXiv:2606.27906](https://arxiv.org/abs/2606.27906), 2026.
- Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Patrick Woods, Gabriel Hillesheim and Abolfazl Razi, [arXiv:2604.04722](https://arxiv.org/abs/2604.04722), 2026.
- Yifan Gong, Yushu Wu, Zheng Zhan, Pu Zhao, Liangkai Liu, Chao Wu, Xulong Tang and Yanzhi Wang, [arXiv:2410.10847](https://arxiv.org/abs/2410.10847), 2024.
- Mine Kerpicci, Van Nguyen, Shuhua Zhang and Erik Visser, [arXiv:2210.16611](https://arxiv.org/abs/2210.16611), 2022.
