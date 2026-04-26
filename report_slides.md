---
marp: true
title: Phase-Adaptive Quantization
description: Report slides for hardware-aware 4-bit MMA rescale pipeline DSE
theme: default
paginate: true
size: 16:9
math: mathjax
style: |
  :root {
    --ink: #14213d;
    --muted: #5b657a;
    --blue: #2563eb;
    --teal: #0891b2;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --line: #d8dee9;
    --soft: #eef4ff;
  }
  section {
    font-family: "Aptos", "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    background: #fbfcff;
    padding: 54px 68px;
    letter-spacing: 0;
  }
  h1 {
    color: var(--ink);
    font-size: 48px;
    line-height: 1.02;
    margin: 0 0 18px 0;
    letter-spacing: 0;
  }
  h2 {
    color: var(--ink);
    font-size: 34px;
    margin: 0 0 24px 0;
    letter-spacing: 0;
  }
  h3 {
    color: var(--muted);
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 18px 0;
    letter-spacing: 0;
  }
  p, li {
    font-size: 22px;
    line-height: 1.32;
  }
  ul, ol {
    margin-top: 12px;
  }
  strong {
    color: var(--blue);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 17px;
  }
  th {
    color: var(--ink);
    border-bottom: 2px solid var(--ink);
    text-align: left;
    padding: 8px 10px;
  }
  td {
    border-bottom: 1px solid var(--line);
    padding: 8px 10px;
  }
  section.cover {
    background: linear-gradient(112deg, #f8fbff 0%, #eef4ff 52%, #fff7ed 100%);
  }
  section.cover h1 {
    font-size: 66px;
    max-width: 850px;
  }
  section.cover p {
    max-width: 760px;
    color: var(--muted);
    font-size: 25px;
  }
  .kicker {
    color: var(--teal);
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 18px;
  }
  .split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 42px;
    align-items: center;
  }
  .wide-split {
    display: grid;
    grid-template-columns: 0.88fr 1.12fr;
    gap: 36px;
    align-items: center;
  }
  .thirds {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    align-items: start;
  }
  .metric {
    font-size: 62px;
    font-weight: 800;
    color: var(--blue);
    line-height: 0.95;
  }
  .metric.small {
    font-size: 44px;
  }
  .label {
    color: var(--muted);
    font-size: 17px;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 0.06em;
  }
  .note {
    color: var(--muted);
    font-size: 18px;
  }
  .callout {
    border-left: 7px solid var(--blue);
    padding-left: 18px;
    margin-top: 22px;
  }
  .flow {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-top: 24px;
  }
  .step {
    background: white;
    border: 1px solid var(--line);
    border-radius: 8px;
    min-height: 112px;
    padding: 14px;
    font-size: 19px;
    line-height: 1.2;
  }
  .step strong {
    display: block;
    font-size: 15px;
    margin-bottom: 8px;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .chip {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    background: var(--soft);
    color: var(--blue);
    font-size: 17px;
    font-weight: 700;
    margin: 4px 6px 4px 0;
  }
  img.fit {
    width: 100%;
    max-height: 500px;
    object-fit: contain;
  }
  img.figure {
    display: block;
    width: 100%;
    max-height: 500px;
    object-fit: contain;
    margin: 0 auto;
  }
  img.tall-figure {
    display: block;
    width: auto;
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
    margin: 0 auto;
  }
  .foot {
    position: absolute;
    left: 68px;
    bottom: 28px;
    color: var(--muted);
    font-size: 14px;
  }
---

<!-- _class: cover -->

<div class="kicker">MIT Deep Learning Hardware Report</div>

# Phase-Adaptive Quantization

### Design space exploration of 4-bit MMA rescale pipelines

Yichong Zhang and Wenye Xiong  
Hardware-aware quantization for custom AI inference accelerators

---

## One-Line Thesis

**A single 4-bit quantization pipeline is not globally optimal.**

Modern inference has two very different phases:

- **Prefill:** large-$M$ forward pass, compute-bound, rescale arithmetic is expensive.
- **Decode:** $M=1$ autoregressive generation, memory-bandwidth-bound, 4-bit weight movement helps.

<div class="callout">
Our search shows that the best datapath changes across workload and phase: C1, C8, and C9 are each selected somewhere under a 0.98 cosine-similarity floor.
</div>

---

## Motivation: Same Format, Opposite Behavior

<div class="wide-split">
<div>

NVFP4 is optimized for general GPUs, but it bundles a fixed datapath:

<span class="chip">2 rescale levels</span>
<span class="chip">b=16 + tensor scale</span>
<span class="chip">FP32 rescale</span>
<span class="chip">FP32 accumulation</span>

That conservative pipeline is safe, but not always energy efficient for a known deployment target.

</div>
<div>

![Energy breakdown](project_proposal%20(1)/figures/energy_breakdown.png)

</div>
</div>

---

## Preliminary Bottleneck Result

| Phase | NVFP4 energy vs FP16 | NVFP4 latency vs FP16 | Rescale share | Interpretation |
|---|---:|---:|---:|---|
| Decode, $M=1$ | **0.90x** | **0.59x** | 64.5% | Bandwidth savings outweigh rescale cost |
| Prefill, $M=128$ | **1.27x** | **0.29x** | 74.0% | Rescale turns W4A4 energy-negative |

<div class="callout">
The same NVFP4-like pipeline helps decode energy but hurts prefill energy. That asymmetry is the reason to search phase-specific datapaths.
</div>

---

## Core Research Question

Can we decompose a 4-bit MMA rescale pipeline into hardware-realizable design dimensions and show that **prefill and decode have different Pareto-optimal configurations**?

The design space varies:

- **Pipeline structure:** 0, 1, or 2 rescale levels; block sizes such as 16 or 32.
- **Scale format:** FP32, FP16, or E8M0 power-of-two shifts.
- **Accumulator precision:** FP32 or FP16.
- **Workload shape:** LLM FFN, VLM vision GEMM, VLA action head.

---

## Search Space: Existing Formats Become Points

| Config | Role | Topology | Scale path | Accumulator |
|---|---|---|---|---|
| C0 | Ideal lower bound | raw FP4 | none | FP32 |
| **C1** | MXFP4-like | 1-level, b=32 | E8M0 shift | FP32 |
| C3/C4 | One-level baselines | 1-level, b=16 | FP16 / FP32 | FP32 |
| **C7** | NVFP4-like reference | 2-level, b=16 + tensor | FP32 + FP32 | FP32 |
| **C8** | Aggressive one-level | 1-level, b=16 | FP16 | FP16 |
| **C9** | Aggressive hybrid | 2-level, b=16 + tensor | E8M0 + FP16 | FP16 |

Existing schemes are not separate categories; they are fixed coordinates inside the same DSE.

---

## What `run_sweeps.py` Automates

<div class="flow">
<div class="step"><strong>Manifest</strong>10 configs x 3 workloads x 2 phases</div>
<div class="step"><strong>Workload YAML</strong>Generate AccelForge einsum graphs for each datapath</div>
<div class="step"><strong>Arch YAML</strong>Parametric PE array, memory, QuantMAC, FP4MAC, RescaleMAC</div>
<div class="step"><strong>Hardware</strong>Run or resume AccelForge mappings; record energy, latency, area</div>
<div class="step"><strong>Accuracy</strong>Pure-Python emulator computes cosine vs FP16 snapshots</div>
</div>

<div class="callout">
The proposal path models inference-time execution: weights are prequantized offline, while runtime activation quantization, FP4 matmul, and output rescale remain in the graph.
</div>

---

## Experiment Setup

| Workload | Representative shape | Phase model | Accuracy proxy |
|---|---:|---|---|
| LLM FFN | $(M, 11008, 4096)$ | Decode $M=1$, prefill $M=128$ | Layer-output cosine |
| VLM vision | $(M, 3072, 3072)$ | Decode $M=1$, prefill $M=128$ | Layer-output cosine |
| VLA action head | $(M, 256, 4096)$ | Decode $M=1$, prefill $M=128$ | Layer-output cosine |

Hardware model: parameterized 8x8 PE-array accelerator, 64 KB GLB, 128 B RF/PE, 45 nm component assumptions.

Completed run status: **60 hardware rows + 60 accuracy rows, all OK**.

---

## Result 1: Best Configurations Differ

Under cosine similarity $\ge 0.98$:

| Workload | Decode best | Decode energy / output | Prefill best | Prefill energy / output |
|---|---|---:|---|---:|
| LLM | **C8** | 7,767 pJ | **C1** | 5,262 pJ |
| VLM | **C1** | 5,557 pJ | **C1** | 3,952 pJ |
| VLA | **C8** | 7,891 pJ | **C9** | 6,263 pJ |

<div class="callout">
No single configuration wins all six workload-phase cells. VLM collapses to one simple mode; LLM and VLA benefit from phase-specific choices.
</div>

---

## Result 2: Pareto Frontiers Shift by Workload and Phase

<img class="tall-figure" src="workspace/lab_4/project4_m1/figures/pareto_panels.png" alt="Pareto panels">

<div class="foot">Each panel plots energy per output against cosine similarity. C7 is NVFP4-like; C1 is MXFP4-like.</div>

---

## Result 3: Phase-Adaptive Beats Fixed NVFP4-like C7

<div class="wide-split">
<div>

Across deployment weights $\alpha$:

- **LLM:** about 58% lower weighted energy vs C7.
- **VLM:** about 60% lower weighted energy vs C7.
- **VLA:** about 51% lower weighted energy vs C7.

Relative to best-fixed, phase adaptation is selective:

- LLM: about 0.6%
- VLM: 0%
- VLA: about 8.2%

</div>
<div>

![Adaptive savings](workspace/lab_4/project4_m1/figures/phase_adaptive_savings.png)

</div>
</div>

---

## Strategy Comparison at Default $\alpha$

<img class="figure" src="workspace/lab_4/project4_m1/figures/phase_adaptive_strategy_bar.png" alt="Strategy comparison">

<div class="foot">Phase-adaptive equals best-fixed for VLM because both phases select C1.</div>

---

## Interpretation

<div class="thirds">
<div>
<div class="metric small">C1</div>
<div class="label">MXFP4-like</div>
<p class="note">Often the low-energy knee. Selected for LLM prefill and both VLM phases.</p>
</div>
<div>
<div class="metric small">C8</div>
<div class="label">FP16 rescale + FP16 acc</div>
<p class="note">Spends a little more energy to clear accuracy in LLM/VLA decode.</p>
</div>
<div>
<div class="metric small">C9</div>
<div class="label">Hybrid two-level</div>
<p class="note">Wins VLA prefill, where a small action-head output dimension changes the tradeoff.</p>
</div>
</div>

<div class="callout">
The main contribution is not one universal format; it is the workload- and phase-aware search methodology.
</div>

---

## Contributions

1. **Structured search space** for 4-bit MMA rescale pipelines with hardware-realizable datapaths.
2. **Phase-aware evaluation** that separates prefill and decode instead of averaging them away.
3. **Reproducible sweep tooling** that generates AccelForge inputs, runs hardware cost models, emulates accuracy, and joins results.
4. **Completed evidence** that C7/NVFP4-like is consistently high energy, while the optimal choice depends on workload and phase.

---

## Limitations and Next Steps

Current limitations:

- Accuracy uses layer-output cosine on bounded tensor slices, not full end-to-end task metrics.
- AccelForge modeling uses analytical component assumptions rather than synthesized RTL.
- Phase switching overhead is discussed qualitatively; implementation cost needs a tighter area and control estimate.

Next work:

- Larger tensor slices and end-to-end perplexity / task accuracy / success-rate evaluation.
- Milestone-3 architecture saturation analysis across QuantMAC and RescaleMAC counts.
- Final write-up polish and a clearer recommendation for when to enable phase adaptation.

---

## Closing Takeaway

**General-purpose 4-bit formats bundle too many choices.**

When the accelerator target is known, those choices should be searched:

<div class="thirds">
<div>
<div class="metric">50-60%</div>
<div class="label">Energy reduction vs NVFP4-like C7</div>
</div>
<div>
<div class="metric">3</div>
<div class="label">Distinct selected datapaths</div>
</div>
<div>
<div class="metric">60 + 60</div>
<div class="label">Completed hardware and accuracy rows</div>
</div>
</div>

Phase-adaptive quantization is useful precisely because it can also tell us when not to adapt.
