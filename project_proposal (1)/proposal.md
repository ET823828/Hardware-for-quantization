# Precision Assignment Design Space Exploration for 4-Bit Quantized Matrix Multiply-Accumulate in FFN Layers

Yichong [Surname], [Partner Name]

---

## I. Project Overview

### Problem and Motivation

LLM inference is fundamentally memory-bandwidth bound: during autoregressive decoding, the entire weight matrix must be streamed from HBM for every generated token. 4-bit weight quantization achieves a 4× reduction over BF16, making it one of the most impactful deployment optimizations. All major hardware vendors have responded: NVIDIA introduced NVFP4 with Blackwell, and the Microscaling (MXFP4) standard is gaining cross-vendor adoption.

However, these 4-bit schemes are **bundled, one-size-fits-all solutions** designed for general-purpose GPUs. Each scheme implicitly fixes the entire MMA datapath: operand format, how many times rescaling occurs, at what granularity, in what precision, and how accumulation is performed. These choices must be conservative to guarantee numerical safety across arbitrary workloads.

Our preliminary analysis of the NVFP4 MMA pipeline reveals that the **rescale and accumulation stages contribute disproportionately to energy and latency**. NVFP4 employs a two-level block scaling architecture where rescale operations are performed in FP32—an FP32 multiplier consumes roughly 4× the energy of FP16 and 16× that of INT8. This conservatism exists because NVIDIA cannot assume anything about the workload. But for a **domain-specific accelerator targeting known workload characteristics**, this represents a significant efficiency opportunity.

The key insight is that the rescale pipeline itself—how many levels of scaling, what granularity each level operates at, and what precision each rescale computation uses—constitutes a structured, multi-dimensional design space. Recent works have made isolated improvements along individual dimensions: HiFloat4 proposes a fixed three-level scaling hierarchy, AXE treats accumulator precision as a tunable parameter, MixPE explores dequantization placement within the PE array, and "Is Finer Better?" analyzes the block-size/scale-format interaction. However, each of these produces a single new hand-designed point in the space, fixing all other dimensions. **The problem has never been formulated as a joint DSE**: no prior work decomposes these dimensions into an orthogonal search space and systematically evaluates the accuracy–hardware-cost tradeoff across configurations. Existing schemes—INT4, MXFP4, NVFP4, and even newer proposals like HiFloat4—each occupy one sparse point in this much richer space.

### Key Questions

**Core question:** Can we formulate the design of a 4-bit MMA rescale pipeline as a structured design-space exploration problem—defining the search space (orthogonal pipeline dimensions), the search method (systematic sweep with pruning), and the evaluation metric (joint accuracy–hardware-cost)—such that this formulation is both valid (contains configurations dominating all existing bundled schemes) and necessary (the optimal configuration is workload-specific, so no single bundled scheme can be universally optimal)?

Our contribution is not a single better configuration, but **the problem formulation itself**: the decomposition of the bundled pipeline into a tractable search space, paired with a hardware-aware evaluation framework. The finding that optimal configurations are workload-specific is what validates this formulation—it proves that hand-designed one-size-fits-all schemes are inherently leaving efficiency on the table.

Specifically:

- **Search space definition:** How should the 4-bit MMA rescale pipeline be decomposed into orthogonal design dimensions (rescale levels, block granularity, scale format, accumulator precision)? What pruning rules reduce the combinatorial space to a tractable set of viable configurations?
- **Evaluation framework:** Can we build a joint evaluation pipeline—PyTorch numerical emulation for accuracy, AccelForge/Timeloop for hardware cost—that enables fair, apples-to-apples comparison across all configurations and existing schemes?
- **Validation:** Do the results confirm that (a) the space contains Pareto-optimal configurations dominating INT4, MXFP4, NVFP4, and HiFloat4, and (b) the optimal configuration varies with workload characteristics, thereby justifying the need for this DSE framework over any fixed bundled design?

### Hypotheses

1. **FP32 rescale is over-provisioned** for many FFN GEMM workloads, particularly those with moderate accumulation chain lengths (K ≤ 4096) and well-behaved weight distributions. Lower-precision rescale (FP16, or even shift-only) can preserve accuracy at substantially lower energy.
2. **Two-level scaling is not universally necessary.** For layers with low dynamic range ratio, single-level scaling with an appropriately chosen block size and scale precision may achieve comparable accuracy while eliminating an entire rescale stage. Conversely, some extreme distributions may benefit from multi-level scaling with cheaper per-level precision.
3. **The optimal configuration is workload-specific.** Layers with small K and low kurtosis tolerate aggressive precision reduction; layers with large K and heavy tails require more conservative accumulation. **This workload-dependence is the key validation of our framework**—it proves that no single bundled scheme can be universally optimal, and that the ability to search the space per-workload is inherently valuable.
4. We target **≥2× energy reduction in the rescale/accumulation stages with <1% accuracy degradation** relative to NVFP4 for the workload-optimal configuration.

---

## II. Technical Contributions

### Existing Solutions

Current 4-bit quantization schemes each bundle a fixed rescale pipeline:

| Scheme | Weight Fmt | # Rescale Levels | Scale Format | Scale Granularity | Accumulator | Rescale Precision |
|--------|-----------|-----------------|-------------|-------------------|-------------|-------------------|
| INT4 (GPTQ/AWQ) | INT4 | 1 | FP16 (scale + zero-point) | per-group (g=128) | FP32 | FP16 dequant |
| MXFP4 | E2M1 | 1 | E8M0 (power-of-2) | per-block (b=32) | FP32 | Bit-shift (free) |
| NVFP4 | E2M1 | 2 | FP8 (fine) + per-tensor (coarse) | per-block (b=16) + per-tensor | FP32 | FP32 multiply (×2) |
| HiFloat4 | E2M1 | 3 | E6M2 global + 1-bit micro-exponents (×2) | hierarchical | FP32 | Mixed |

Recent works have improved individual dimensions of this pipeline. HiFloat4 introduces a three-level scaling hierarchy but fixes it as a single format. AXE treats accumulator precision as a tunable parameter but operates purely in the PTQ software domain without hardware cost modeling. MixPE explores dequantization placement within PE arrays via Pareto analysis but does not touch the internal rescale pipeline structure. "Is Finer Better?" reveals that smaller block sizes can hurt when scale factor precision is insufficient, but stops at analysis without proposing a design methodology. M²XFP and MX+ enhance the MX format through metadata augmentation and outlier handling, each producing one new fixed scheme.

All of these works share a common pattern: **each proposes or optimizes one new fixed-point design, without formulating the problem as a joint search over the full pipeline configuration space.** Hardware-aware quantization (HAQ, mixed-precision NAS) does search, but over per-layer bit-width on fixed hardware—not over the internal MMA pipeline structure.

**What's missing:** The problem has never been formulated as a structured DSE: decomposing the rescale pipeline into orthogonal dimensions, defining a tractable search space, and systematically evaluating the joint accuracy–hardware-cost tradeoff across configurations.

### Our Approach: Unbundling the Rescale Pipeline

We decompose the 4-bit MMA datapath into independently tunable design dimensions, with the **rescale pipeline structure** as the central focus.

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

Our contribution is not any single configuration, but **the formulation itself**: decomposing the bundled pipeline into this search space, building the evaluation infrastructure (accuracy emulation + hardware cost modeling), and demonstrating through systematic exploration that (a) all existing schemes are suboptimal, and (b) the optimal configuration is workload-dependent—validating the necessity of this framework.

### Workload Characterization

We focus on **weight × activation GEMMs in transformer FFN/MLP layers**. This choice is deliberate: in standard transformer architectures, FFN layers account for approximately **two-thirds of total model parameters** and a comparable share of FLOPs. For example, in Llama-2-7B, each transformer block contains two FFN projections (gate/up and down) whose weight matrices are 2–3× larger than those in the attention layer. During autoregressive decoding, these large weight matrices must be streamed from memory on every token, making FFN the dominant **memory-bandwidth and energy bottleneck**. Optimizing the quantization pipeline for FFN GEMMs therefore captures the majority of the efficiency gains available from 4-bit quantization.

Each GEMM is characterized by a compact feature vector combining shape and data distribution:

- **Shape features:** M (batch/tokens), N (output dim), K (reduction dim). K is the most critical: it determines accumulation chain length and thus the required accumulator dynamic range.
- **Weight distribution features:** kurtosis (tail heaviness → FP vs INT preference), dynamic range ratio max/mean (→ scale factor precision requirement), per-block variance ratio (→ whether single-level scaling suffices or multi-level is needed).
- **Activation distribution features:** variance, outlier ratio (fraction beyond 3σ). Measured over a calibration set; treated as stable statistical properties following standard PTQ methodology.

This joint (shape, distribution) characterization enables workload-feature-driven conclusions rather than model-specific ones.

### Tradeoffs and Challenges

- **Search space tractability:** The full combinatorial space is large, but many configurations can be pruned (e.g., accumulator precision ≥ operand precision; 0-level scaling only viable for near-uniform distributions). After pruning, we estimate ~100–200 viable configurations.
- **Design-time vs. runtime:** Our accelerator fixes the pipeline at design time (no reconfigurable overhead). A single configuration must cover all FFN layers in the target model—or we allow 2–3 switchable modes with accounted switching cost.
- **Scale factor storage overhead:** More rescale levels and finer blocks mean more scale factors stored alongside weights, reducing effective compression ratio. This overhead must be included in the total efficiency metric.
- **Activation distribution variability:** Weight distributions are fixed post-training, but activation distributions vary with input. We rely on calibration-set statistics being representative.

---

## III. Evaluation

### Baselines and Fair Comparison

We compare against three baselines: INT4 (GPTQ-style), MXFP4, and NVFP4.

**Held constant:**
- All configurations use 4-bit weights. Effective compression ratio (including scale factor overhead) is reported.
- Activation precision is FP8 E4M3 unless stated otherwise.
- GEMM shapes (M, N, K) are identical across all configurations.
- Area is either held constant or included via energy-delay-area product.

**Accuracy metric:** Per-layer output MSE relative to FP16 baseline; end-to-end task metric (perplexity for LLMs, top-1 accuracy for ViT) when feasible.

### Experiment 1: Energy Breakdown and Rescale Overhead Quantification

**Goal:** Establish the motivation by quantifying the energy cost of each MMA pipeline stage for existing schemes, demonstrating that rescale/accumulation overhead is significant and worth optimizing.

**Method:** Model a systolic-array MMA unit in Timeloop/Accelergy for each baseline (INT4, MXFP4, NVFP4). Decompose total energy into: (1) operand fetch from buffers, (2) multiply, (3) accumulation, (4) rescale—per level, (5) scale-factor fetch, (6) result writeback.

**Class concepts:** Energy cost hierarchy (compute vs. data movement); multiplier energy scaling with bit-width; the role of on-chip buffers in amortizing data movement cost.

**Metrics:** Energy per output element (pJ), broken down by pipeline stage; area per MAC unit (µm²).

**Visualization:** Stacked bar chart. X-axis: scheme (INT4, MXFP4, NVFP4). Y-axis: energy per output element. Each bar segmented by pipeline stage.

**Expected trend:** NVFP4 shows the highest rescale energy due to two FP32 multiplications. MXFP4's shift-based rescale is nearly free. Rescale + accumulation constitutes >40% of total MMA energy for NVFP4. INT4 falls in between. This motivates exploring whether cheaper rescale can preserve accuracy.

### Experiment 2: Accuracy–Efficiency Pareto Frontier

**Goal:** Sweep the rescale pipeline design space and show that configurations exist which dominate all existing bundled schemes.

**Method:**
1. **Accuracy:** For each candidate configuration, implement a PyTorch emulator simulating the quantized MMA with configurable rescale pipeline (number of levels, block sizes, scale formats, accumulator precision). Evaluate on FFN layers from Llama-2-7B, Llama-2-70B, and ViT-L. Measure per-layer MSE and end-to-end metric.
2. **Hardware cost:** For each configuration, build a Timeloop/Accelergy model. Extract energy per output element and area.
3. **Joint plot:** Overlay all configurations and baselines on the accuracy–energy plane.

**Class concepts:** Design space exploration; Pareto optimality; tradeoff between numerical precision and hardware resource cost.

**Metrics:** X-axis: energy per output element (pJ). Y-axis: accuracy (negative MSE or task metric). Each point is one configuration.

**Visualization:** Scatter plot with Pareto frontier curve. Existing schemes (INT4, MXFP4, NVFP4) as labeled reference points. Pareto-optimal discovered configurations highlighted.

**Expected trend:** Configurations such as {E2M1, 1-level b=16, FP16 scale, FP16 acc, FP16 rescale} or {E2M1, 2-level b=16+channel, E8M0 fine + FP16 coarse, BF16 acc} will match NVFP4 accuracy at substantially lower energy. The frontier will show a knee where further energy reduction causes steep accuracy drops. All three existing bundled schemes will be dominated (interior to the frontier), demonstrating the value of unbundled pipeline design.

### Experiment 3: Workload-Feature Sensitivity Analysis

**Goal:** Show that the optimal pipeline configuration depends on workload features (K, kurtosis, dynamic range), validating the need for workload-driven design-time specialization.

**Method:** Select 3 configurations from the Pareto frontier spanning aggressive/moderate/conservative. Profile every FFN GEMM layer from Llama-2-7B, Llama-2-70B, and ViT-L, extracting (K, kurtosis, dynamic_range_ratio, per_block_variance_ratio). For each layer × configuration, measure accuracy. Map the optimal configuration to the feature space.

**Class concepts:** Workload characterization; accumulation chain length vs. required dynamic range; design-time specialization vs. one-size-fits-all.

**Metrics:** Per-layer accuracy degradation for each configuration; energy savings under per-layer-optimal assignment vs. single fixed configuration.

**Visualization:**
- (a) 2D scatter: X = K, Y = kurtosis, color/marker = optimal configuration. Reveals decision boundaries in feature space.
- (b) Table: percentage of layers where each configuration is optimal, by model.
- (c) Line plot: X = K (binned), Y = maximum tolerable rescale precision reduction. Shows how K governs precision requirements.

**Expected trend:** Small-K layers (K ≤ 2048) with low kurtosis tolerate single-level scaling with FP16 or shift-only rescale. Large-K layers (K > 4096) with heavy tails need FP32 accumulation or segmented accumulation. K will be the strongest predictor of required precision. Cross-model analysis will show that (K, distribution) features—not model identity—predict optimal configuration, confirming generalizability.

---

## IV. Timeline and Work Distribution

| Week | Tasks | Owner |
|------|-------|-------|
| 1–2 | Define and prune search space; implement configurable PyTorch MMA emulator (rescale levels, block sizes, scale formats, accumulator precision); profile FFN layers from target models to extract (shape, distribution) features | [TBD] |
| 3–4 | Build Timeloop/Accelergy models for candidate configurations; energy/area estimation; Experiment 1 (energy breakdown of baselines) | [TBD] |
| 5–6 | Full design-space sweep: accuracy × hardware cost; Pareto frontier (Experiment 2); identify Pareto-optimal and dominated configurations | Joint |
| 7–8 | Workload-feature sensitivity analysis (Experiment 3); final figures; write-up and presentation | Joint |

**Work split:** One member focuses on the software accuracy pipeline (PyTorch emulation, model profiling, workload feature extraction). The other focuses on hardware cost modeling (Timeloop/Accelergy configuration, energy/area estimation, datapath design). Both collaborate on joint Pareto analysis and writing.
