# Phase-Adaptive Quantization for LLM Inference: A Design Space Exploration of 4-Bit MMA Rescale Pipelines

Yichong Zhang, [Partner Name]

---

## I. Project Overview

### Problem and Motivation

LLM inference consists of two fundamentally different computational phases. **Prefill** processes the entire input prompt in a single forward pass with large batch dimension (M >> 1), making it **compute-bound**: the energy and latency bottleneck lies in arithmetic operations, particularly in the multiply-accumulate datapath. **Decode** generates tokens autoregressively with M = 1, making it **memory-bandwidth-bound**: the bottleneck is streaming the full weight matrix from HBM for every generated token. These two phases have radically different hardware utilization profiles—yet current 4-bit quantization schemes apply a single, fixed pipeline to both.

All major 4-bit quantization formats—NVIDIA's NVFP4 (Blackwell), the Microscaling standard (MXFP4), and INT4 (GPTQ/AWQ)—are **bundled, one-size-fits-all designs**. Each scheme implicitly fixes the entire MMA datapath: how many times rescaling occurs, at what granularity, in what precision, and how accumulation is performed. These choices are conservative because GPU vendors must guarantee numerical safety across arbitrary workloads and both inference phases. But this conservatism creates a fundamental mismatch:

- **During prefill**, the rescale pipeline dominates energy. Our preliminary analysis of the NVFP4 MMA pipeline shows that FP32 rescale and accumulation stages contribute **>40% of total MMA energy**—an FP32 multiplier consumes ~4× the energy of FP16. Yet for compute-bound prefill, cheaper rescale formats (FP16, bit-shift) may preserve accuracy at substantially lower energy.
- **During decode**, memory bandwidth dominates. The primary benefit of quantization is reducing data movement, and the rescale pipeline's energy contribution is comparatively small. More aggressive quantization (fewer bits, coarser granularity) may be optimal because the savings in bandwidth outweigh any marginal accuracy loss.

**The key insight is that prefill and decode should use different quantization configurations**—but the design question of *which* configuration is best for each phase has never been systematically explored. The rescale pipeline itself—how many levels of scaling, what granularity, what precision—constitutes a structured, multi-dimensional design space. Recent works have made isolated improvements along individual dimensions: HiFloat4 proposes a three-level scaling hierarchy, AXE treats accumulator precision as a tunable parameter, MixPE explores dequantization placement, and "Is Finer Better?" analyzes block-size/scale-format interactions. However, each produces a single new hand-designed point in this space, and **none considers the prefill/decode dichotomy**. The problem has never been formulated as a joint DSE that decouples the two phases and searches for phase-specific optima.

### Key Questions

**Core question:** Can we define a structured search space over the 4-bit MMA rescale pipeline—decomposed into orthogonal dimensions (rescale levels, block granularity, scale format, accumulator precision)—and demonstrate that prefill and decode have *different* Pareto-optimal configurations within this space, thereby justifying phase-adaptive quantization over any single bundled scheme?

Our contribution is not a single better configuration, but **the search space definition and the phase-decoupling methodology**. The finding that prefill and decode prefer different configurations is the central validation—it proves that one-size-fits-all schemes are inherently leaving efficiency on the table, and that the ability to independently select quantization per phase is valuable.

Specifically:

- **Search space definition:** How should the 4-bit MMA rescale pipeline be decomposed into orthogonal design dimensions? What pruning rules reduce the combinatorial space to a tractable set of viable configurations?
- **Phase-aware evaluation:** Can we build a joint evaluation pipeline—AccelForge hardware cost modeling under both prefill (large M, compute-bound) and decode (M=1, memory-bound) regimes, plus PyTorch numerical emulation for accuracy—that enables fair comparison across all configurations?
- **Validation:** Do the results confirm that (a) the Pareto-optimal configuration for prefill differs from decode, (b) both phase-specific optima dominate existing bundled schemes, and (c) the gap between phase-adaptive and fixed-scheme design is significant?

### Hypotheses

1. **Prefill favors cheaper rescale.** In the compute-bound regime (large M), the rescale pipeline is the dominant energy contributor. Lower-precision rescale (FP16, bit-shift) can achieve comparable accuracy at significantly lower energy, because compute utilization is high and the energy savings from cheaper arithmetic are fully realized.
2. **Decode favors more aggressive quantization.** In the memory-bound regime (M=1), the primary benefit of quantization is reduced data movement. Configurations with coarser granularity or fewer rescale levels—even if slightly less accurate—may be Pareto-optimal because bandwidth savings dominate.
3. **The optimal configuration differs between prefill and decode.** The Pareto frontier for prefill occupies a different region of the accuracy–energy plane than for decode, and the configuration at each frontier's knee point is distinct. This phase-dependence is the key validation—it proves that no single bundled scheme can be universally optimal.
4. We target **≥2× energy reduction in the rescale/accumulation stages** relative to NVFP4 for the prefill-optimal configuration, and **measurable end-to-end throughput improvement** from phase-adaptive selection versus a single fixed scheme.

---

## II. Technical Contributions

### Existing Solutions

Current 4-bit quantization schemes each bundle a fixed rescale pipeline, applied uniformly to both prefill and decode:

| Scheme | Weight Fmt | # Rescale Levels | Scale Format | Scale Granularity | Accumulator | Rescale Precision |
|--------|-----------|-----------------|-------------|-------------------|-------------|-------------------|
| INT4 (GPTQ/AWQ) | INT4 | 1 | FP16 (scale + zero-point) | per-group (g=128) | FP32 | FP16 dequant |
| MXFP4 | E2M1 | 1 | E8M0 (power-of-2) | per-block (b=32) | FP32 | Bit-shift (free) |
| NVFP4 | E2M1 | 2 | FP8 (fine) + per-tensor (coarse) | per-block (b=16) + per-tensor | FP32 | FP32 multiply (×2) |
| HiFloat4 | E2M1 | 3 | E6M2 global + 1-bit micro-exponents (×2) | hierarchical | FP32 | Mixed |

Recent works have improved individual dimensions of this pipeline. HiFloat4 introduces a three-level scaling hierarchy but fixes it as a single format. AXE treats accumulator precision as a tunable parameter but operates purely in the PTQ software domain without hardware cost modeling. MixPE explores dequantization placement within PE arrays via Pareto analysis but does not touch the internal rescale pipeline structure. "Is Finer Better?" reveals that smaller block sizes can hurt when scale factor precision is insufficient, but stops at analysis without proposing a design methodology. M²XFP and MX+ enhance the MX format through metadata augmentation and outlier handling, each producing one new fixed scheme.

All of these works share a common limitation: **each produces a single fixed design applied uniformly to all inference phases.** None considers that prefill and decode have fundamentally different compute-vs-memory profiles, and therefore may benefit from different pipeline configurations. Hardware-aware quantization (HAQ, mixed-precision NAS) does search, but over per-layer bit-width on fixed hardware—not over the internal MMA pipeline structure, and not across inference phases.

**What's missing:** (1) A structured DSE formulation that decomposes the rescale pipeline into orthogonal dimensions and defines a tractable search space. (2) Phase-aware evaluation that separately models hardware cost under prefill and decode regimes. (3) The insight that the optimal configuration is phase-dependent, making phase-adaptive quantization both feasible and necessary.

### Our Approach: Phase-Adaptive Quantization via Search Space Decomposition

We decompose the 4-bit MMA datapath into independently tunable design dimensions, define a structured search space, and evaluate each configuration under **both prefill and decode regimes** to identify phase-specific optima.

#### Dimension 1: Weight Operand Format

| Option | Description |
|--------|-------------|
| INT4 | 4-bit uniform integer, 16 evenly spaced levels |
| E2M1 (FP4) | 4-bit float, log-spaced, better for bell-shaped distributions |
| E3M0 | 4-bit float with wider exponent range, no mantissa bit |
| E1M2 | 4-bit float with narrower range but finer resolution near zero |

#### Dimension 2: Rescale Pipeline Structure

This is the core of our search space. We jointly explore **the number of rescale levels** and **the block granularity at each level**:

| # Levels | Granularity | Description | Rescale Ops |
|----------|------------|-------------|-------------|
| 0 | — | Raw 4-bit, no scale factors | 0 |
| 1 | b=16 | One fine-grained scale per 16 elements | 1 per block |
| 1 | b=32 | One scale per 32 elements | 1 per block |
| 1 | b=64 | One coarse scale per 64 elements | 1 per block |
| 1 | per-channel | One scale per output channel | 1 per channel |
| 2 | b=16, B=256 | Fine block + intermediate group | 2 |
| 2 | b=16, B=per-tensor | Fine block + global (NVFP4-style) | 2 |
| 2 | b=16, B=per-channel | Fine block + per-channel | 2 |
| 2 | b=32, B=per-channel | Coarser fine block + per-channel | 2 |

The key tradeoff: finer granularity and more levels → higher accuracy (tighter per-block scaling) but more rescale operations and more scale-factor storage overhead.

#### Dimension 3: Scale Factor Format (per level, independently)

Each rescale level independently chooses its precision:

| Option | Rescale Cost | Dynamic Range | Notes |
|--------|-------------|---------------|-------|
| FP32 | FP32 multiply (baseline, expensive) | Full | NVFP4 uses this |
| FP16 | ~4× cheaper than FP32 | Adequate for most ranges | |
| BF16 | ~4× cheaper than FP32 | FP32 range, less mantissa | |
| FP8 E4M3 | ~8× cheaper | Limited, may suffice for fine-level | |
| E8M0 (power-of-2) | Bit-shift (nearly free) | Quantized to powers of 2 | MXFP4 uses this |
| Fixed-point | Shift + small multiply | Tunable | |

In a 2-level pipeline, the fine and coarse levels can use *different* formats. For example: E8M0 shift at the fine level + FP16 multiply at the coarse level—combining cheap per-block rescale with accurate global correction.

#### Dimension 4: Accumulator Precision

| Option | Description |
|--------|-------------|
| FP32 | Full precision, safe for any K (all baselines use this) |
| FP16 | ~4× cheaper, risk of swamping for large K |
| BF16 | FP32 range but less mantissa precision |
| INT32 | Cheap for integer operands, exact for INT4×INT8 |
| Segmented (FP16 → FP32) | Accumulate every N partial products in FP16, then reduce to FP32. N is a design parameter |

#### Dimension 5: Activation Format

FP8 E4M3 (default); INT8 as alternative. Held constant in most experiments.

#### Locating Existing Schemes in the Space

Every existing 4-bit scheme is a single hand-designed point in this multi-dimensional space:

- **NVFP4** = {E2M1, 2-level (b=16, B=tensor), FP8 fine + per-tensor coarse, FP32 acc, FP32 rescale at both levels}
- **MXFP4** = {E2M1, 1-level (b=32), E8M0, FP32 acc, bit-shift rescale}
- **INT4-GPTQ** = {INT4, 1-level (g=128), FP16 scale+zp, FP32 acc, FP16 dequant}
- **HiFloat4** = {E2M1, 3-level (micro-exponent hierarchy), E6M2 + E1 + E1, FP32 acc, mixed rescale}

Our contribution is not any single configuration, but **the search space definition and phase-decoupling methodology**: decomposing the bundled pipeline into this search space, evaluating under both prefill and decode regimes, and demonstrating that (a) existing bundled schemes are suboptimal for both phases, and (b) the optimal configuration differs between prefill and decode—validating the need for phase-adaptive quantization.

### Workload Characterization: Prefill vs. Decode

We focus on **weight × activation GEMMs in transformer FFN/MLP layers**. FFN layers account for approximately **two-thirds of total model parameters** and a comparable share of FLOPs, making them the dominant target for quantization optimization.

The critical variable distinguishing prefill from decode is **M (batch/token dimension)**:

| Phase | M | Compute Profile | Energy Bottleneck | Quantization Priority |
|-------|---|----------------|-------------------|----------------------|
| **Decode** | 1 | Memory-bandwidth-bound | Data movement (weight streaming) | Minimize data footprint: fewer bits, coarser granularity |
| **Prefill** | 128–2048 | Compute-bound | Arithmetic (MAC + rescale) | Minimize compute energy: cheaper rescale, lower-precision accumulation |

Beyond M, each GEMM is further characterized by:

- **K (reduction dim):** Determines accumulation chain length and required dynamic range. Larger K stresses the accumulator and may require more conservative rescale.
- **Weight distribution features:** kurtosis, dynamic range ratio, per-block variance ratio—these influence whether single-level or multi-level scaling is needed.

This characterization enables conclusions driven by (phase, shape, distribution) rather than model identity.

### Tradeoffs and Challenges

- **Search space tractability:** After pruning (e.g., accumulator precision ≥ operand precision; 0-level scaling only for near-uniform distributions), we estimate ~10–20 viable configurations per evaluation.
- **Phase switching cost:** Supporting two quantization modes (prefill vs. decode) requires either (a) two sets of quantized weights in memory, or (b) runtime requantization. We account for this overhead in our evaluation. In practice, the weight storage overhead is modest since scale factors are small relative to weight tensors.
- **Scale factor storage overhead:** More rescale levels and finer blocks mean more scale factors, reducing effective compression ratio. This is included in our total efficiency metric.
- **Design-time commitment:** The accelerator's compute units are fixed at design time. Phase-adaptive quantization requires the datapath to support at least two rescale modes—we show this is feasible with minimal area overhead.

---

## III. Evaluation

### Baselines and Fair Comparison

We compare against three baselines: INT4 (GPTQ-style), MXFP4, and NVFP4—each evaluated under both prefill and decode regimes.

**Held constant across all configurations:**
- All configurations use 4-bit weights. Effective compression ratio (including scale factor overhead) is reported.
- Activation precision is FP16 or FP8 E4M3 unless stated otherwise.
- GEMM shapes (N, K) are identical; M varies to represent prefill (M=128) vs. decode (M=1).

**Accuracy metric:** Per-layer output MSE and SQNR relative to FP16 baseline.

### Experiment 1: Phase-Dependent Energy Bottleneck Analysis

**Goal:** Demonstrate that prefill and decode have qualitatively different energy bottleneck structures, establishing the motivation for phase-adaptive quantization.

**Method:** Model the NVFP4 MMA pipeline in AccelForge/Timeloop with parameterized batch dimension. For each baseline scheme (Baseline FP16, W4A16, W4A4), run hardware evaluation under M=1 (decode) and M=128 (prefill). Decompose total energy into compute vs. memory components, and further break down by pipeline stage (quantize, matmul, rescale, accumulate).

**Class concepts:** Compute-bound vs. memory-bound regimes; energy cost hierarchy; the role of arithmetic intensity in determining bottleneck location.

**Metrics:** Per-stage energy (pJ) and latency (cycles); compute-to-memory energy ratio; fraction of total energy from rescale stages.

**Visualization:** Side-by-side stacked bar charts for prefill vs. decode. Each bar segmented by pipeline stage. Annotate the dominant bottleneck for each phase.

**Expected trend:** Under prefill (M=128), rescale stages dominate energy (>40% for NVFP4), making rescale precision the key optimization lever. Under decode (M=1), memory bandwidth dominates, and rescale energy is a small fraction—quantization granularity and data footprint matter more.

### Experiment 2: Phase-Specific Pareto Frontiers

**Goal:** Sweep the search space and show that the Pareto-optimal configuration for prefill differs from decode, validating phase-adaptive design.

**Method:**
1. **Accuracy:** For each candidate configuration, implement a PyTorch emulator simulating the quantized MMA with configurable rescale pipeline. Measure per-layer MSE/SQNR. Accuracy is phase-independent (same mathematical pipeline).
2. **Hardware cost:** For each configuration, evaluate in AccelForge under both M=1 (decode) and M=128 (prefill). Extract energy per output element.
3. **Joint plot:** For each phase, overlay all configurations on the accuracy–energy plane. Draw separate Pareto frontiers.

**Class concepts:** Design space exploration; Pareto optimality; phase-aware hardware-software co-design.

**Metrics:** X-axis: energy per output element (pJ). Y-axis: SQNR (dB). Each point is one configuration. Two Pareto frontier curves (prefill, decode) on the same plot.

**Visualization:** Scatter plot with two Pareto frontiers overlaid (different colors for prefill vs. decode). Existing schemes (INT4, MXFP4, NVFP4) as labeled reference points. Highlight the knee-point configuration for each phase.

**Expected trend:** The prefill Pareto frontier favors configurations with cheaper rescale (FP16/bit-shift) since compute energy dominates. The decode frontier favors configurations with coarser granularity or fewer rescale levels since bandwidth dominates. The knee-point configuration differs between the two frontiers. Existing bundled schemes are interior to both frontiers.

### Experiment 3: Phase-Adaptive Gain Quantification

**Goal:** Quantify the end-to-end benefit of phase-adaptive quantization (different config for prefill vs. decode) over a single fixed scheme.

**Method:** For each existing baseline and for the best fixed configuration from the search space, compute weighted energy = α × E_prefill + (1−α) × E_decode, where α reflects the prefill/decode time ratio for a given deployment scenario (e.g., α=0.3 for chatbot, α=0.8 for batch summarization). Compare against phase-adaptive: E_adaptive = α × E_prefill_opt + (1−α) × E_decode_opt, where each phase uses its own Pareto-optimal configuration.

**Class concepts:** Workload characterization; design-time specialization vs. one-size-fits-all; system-level optimization.

**Metrics:** Energy reduction (%) of phase-adaptive vs. best fixed scheme, as a function of prefill/decode ratio α. Area overhead of supporting two modes.

**Visualization:**
- (a) Line plot: X = α (prefill fraction), Y = energy savings of phase-adaptive vs. best fixed. Shows that phase-adaptive wins across all deployment scenarios.
- (b) Table: best configuration for each phase, with per-stage energy breakdown.
- (c) Sensitivity analysis: vary K to show how the optimal per-phase configuration shifts with GEMM shape.

**Expected trend:** Phase-adaptive quantization yields 15–30% energy reduction over the best single fixed scheme across typical deployment scenarios. The benefit is largest for mixed workloads (α ≈ 0.5) where neither phase's optimal alone is satisfactory.

---

## IV. Timeline and Work Distribution

| Week | Tasks | Owner |
|------|-------|-------|
| 1–2 | Finalize search space (prune to ~10 configs); build AccelForge workload/arch variants for all configs; run Experiment 1 (prefill vs. decode bottleneck analysis) | [TBD] |
| 3–4 | Implement PyTorch quantization emulator; accuracy evaluation across all configs; run AccelForge sweep for both M=1 and M=128 | [TBD] |
| 5–6 | Experiment 2 (phase-specific Pareto frontiers); Experiment 3 (phase-adaptive gain quantification); sensitivity analysis over K | Joint |
| 7–8 | Final figures; write-up and presentation | Joint |

**Work split:** One member focuses on AccelForge hardware modeling (workload/arch YAML generation, mapping evaluation, energy/latency extraction). The other focuses on PyTorch accuracy evaluation (quantization emulator, MSE/SQNR measurement). Both collaborate on Pareto analysis and writing.
