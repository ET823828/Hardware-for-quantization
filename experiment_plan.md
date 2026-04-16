# Experiment Plan — Rescale Pipeline DSE

## 现有基础

已完成：
- `arch_template.yaml`：参数化架构，4 个计算单元（QuantMAC/FP4MAC/RescaleMAC/FP16MAC），用 `enabled` 约束路由
- 3 个 workload + mapping：Baseline (FP16), W4A16, W4A4 (NVFP4 full)
- 能耗/延迟 breakdown 数据已验证

## 设计空间（精简版）

固定：weight format = E2M1, activation = FP8/FP16, GEMM shape 参数化

### 3 个搜索维度

| 维度 | 选项 |
|------|------|
| D1: Rescale 级数 × 粒度 | 0-level, 1-level (b=16), 1-level (b=32), 2-level (b=16, B=tensor) |
| D2: Scale factor 格式 → rescale 能耗 | FP32 (3.7pJ), FP16 (1.0pJ), E8M0 bit-shift (0.03pJ) |
| D3: 累加器精度 | FP32, FP16 |

### 10 个候选配置

| ID | 级数 | 粒度 | Scale 格式 | Acc 精度 | 对标 | 需要新建 |
|----|------|------|-----------|---------|------|---------|
| C0 | 0 | — | — | FP32 | (raw 4-bit, lower bound) | workload + arch |
| C1 | 1 | b=32 | E8M0 shift | FP32 | **MXFP4** | workload + arch |
| C2 | 1 | b=16 | E8M0 shift | FP32 | (cheapest fine-grained) | workload + arch |
| C3 | 1 | b=16 | FP16 | FP32 | (中档 1-level) | workload + arch |
| C4 | 1 | b=16 | FP32 | FP32 | (贵的 1-level) | workload + arch |
| C5 | 2 | b=16+tensor | E8M0+FP16 | FP32 | (hybrid: shift fine + FP16 coarse) | workload + arch |
| C6 | 2 | b=16+tensor | FP16+FP16 | FP32 | (全 FP16 rescale) | workload + arch |
| C7 | 2 | b=16+tensor | FP32+FP32 | FP32 | **NVFP4** (已有) | ✓ 已有 |
| C8 | 1 | b=16 | FP16 | FP16 | (激进: 全低精度) | workload + arch |
| C9 | 2 | b=16+tensor | E8M0+FP16 | FP16 | (最激进 2-level) | workload + arch |

共 10 个配置，其中 C7 已有（= 现有 NVFP4 full），需要新建 9 个。

---

## 实验 1: Hardware Cost Modeling (AccelForge)

### 要做的事

对每个配置，需要：
1. **Workload YAML** — 定义该 pipeline 结构的 einsums 和 tensor accesses
2. **Arch YAML** — 修改 RescaleMAC 的 energy（根据 scale format）和累加器精度
3. **Auto-map + evaluate** — 用 FFM mapper 生成 mapping，提取 energy/latency breakdown

### Workload 模板分类

实际上只有 3 种 workload 结构（按 pipeline 拓扑）：

**结构 A: 0-level**（C0）
```
Aq × Wq → Y    (1 einsum, 无 scale factor)
```

**结构 B: 1-level block-only**（C1, C2, C3, C4, C8）
```
BlockScale(A/W) → BlockQuant(A/W) → MatMul → RescaleBlock(A/W)
(6 einsums, 类似 W4A16 但 rescale 在 matmul 之后)
```
不同配置只需改：
- `ki` 维度大小（b=16 → ki<16; b=32 → ki<32）
- arch 里 RescaleMAC 的 energy
- 中间张量的 bits_per_value（acc 精度）

**结构 C: 2-level block+tensor**（C5, C6, C7, C9）
```
TensorScale/Quant(A/W) → BlockScale/Quant(A/W) → MatMul → RescaleBlock(A/W) → RescaleTensor(A/W)
(13 einsums, = 现有 NVFP4 full 结构)
```
不同配置只需改 arch 里 RescaleMAC 的 energy。
可能需要两个不同 energy 的 rescale 单元（fine vs coarse）。

### 实际工作量

| 任务 | 工作量 |
|------|--------|
| 写结构 A workload (0-level) | 新写 ~30 行 YAML |
| 写结构 B workload (1-level) | 从 W4A16 改造，~60 行 |
| 结构 C workload | 已有（= nvfp4_full），只需改 arch |
| 每个 arch 变体 | 改 RescaleMAC energy 一行 + acc bits |
| 每个配置 auto-map + evaluate | 跑一次 notebook cell |

**总计：写 2 个新 workload 模板 + 9 个 arch 变体 → 跑 9 次 mapping**

### 输出

| 配置 | Energy (pJ) | Latency (cycles) | Area (m²) | 各阶段 breakdown |
|------|-------------|-------------------|-----------|------------------|

画一张 **stacked bar chart**，10 个配置并排，能耗按阶段分色。

---

## 实验 2: Accuracy Evaluation (PyTorch)

### 方法

写一个 **单文件 PyTorch 脚本**（~200 行），实现 configurable 量化 MMA emulator：

```python
def quantized_matmul(A, W, config):
    """
    config = {
        'block_size': 16 or 32,
        'n_levels': 0 or 1 or 2,
        'scale_format': 'fp32' | 'fp16' | 'e8m0',
        'acc_dtype': torch.float32 | torch.float16,
    }
    """
    # 1. Quantize W to 4-bit E2M1 with block scaling
    # 2. Quantize A similarly (if W4A4)
    # 3. MatMul in low precision
    # 4. Rescale with specified precision
    # 5. Return output
```

### 精度指标

- `MSE = mean((Y_quant - Y_fp16)^2)`
- `SQNR = 10 * log10(mean(Y_fp16^2) / MSE)` (dB, 越高越好)

### 数据源（2 选 1，按可行性）

**选项 A（推荐，最简单）：** 用随机高斯权重 + 随机激活，控制分布参数
- 生成 W ~ N(0, σ²)，改变 kurtosis 用 t-分布
- 优点：不需要下载模型，纯 CPU 就能跑
- 缺点：不是真实权重分布

**选项 B（更有说服力）：** 从 HuggingFace 加载一个小模型的 FFN 权重
- `facebook/opt-125m`（125M 参数，~250MB）或 `TinyLlama-1.1B`
- 取 2-3 层的 gate_proj/down_proj 权重矩阵
- 优点：真实分布
- 缺点：需要 GPU 或较大内存

### 要评估的 GEMM shapes

| Shape | M | N | K | 对应 |
|-------|---|---|---|------|
| S1 | 1 | 4096 | 4096 | Llama FFN gate_proj, batch=1 decode |
| S2 | 1 | 4096 | 11008 | Llama FFN down_proj, batch=1 decode |
| S3 | 128 | 4096 | 4096 | Prefill, batch=128 |

注意：K 是最关键的变量，因为它决定累加链长度。

### 工作量

| 任务 | 工作量 |
|------|--------|
| 实现 E2M1 量化函数 | ~50 行 |
| 实现 block scaling + rescale | ~80 行 |
| 实现 configurable pipeline | ~50 行 |
| 跑 10 configs × 3 shapes | 一个 for 循环，CPU 几分钟 |

### 输出

10 configs × 3 shapes 的 MSE/SQNR 表格。

---

## 实验 3: 合并 → Pareto + Workload-Specificity

### Pareto Frontier

把实验 1 的 energy 和实验 2 的 SQNR 合并：

- X 轴：Energy per output element (pJ)
- Y 轴：SQNR (dB)
- 每个点 = 一个配置
- 标注 NVFP4 (C7) 和 MXFP4 (C1) 的位置
- 画 Pareto frontier

**预期：** C5 或 C6（hybrid rescale）在 Pareto 前沿上，NVFP4 被支配。

### Workload-Specificity

对 3 个 GEMM shapes，分别找各自的 Pareto-optimal 配置：
- 如果 S1 (K=4096) 的最优是 C3，S2 (K=11008) 的最优是 C6 → 证明 workload-specific
- 如果所有 shape 的最优都是同一个 → 也是有意义的结论（说明某个配置 robust）

**可视化：** 一张图叠 3 条 Pareto frontier（不同 K），用不同颜色/线型。

---

## 时间线（4 周精简版）

| 周 | 硬件侧 | 软件侧 | 产出 |
|----|--------|--------|------|
| 1 | 写 2 个 workload 模板（0-level, 1-level）+ 9 个 arch 变体 | 实现 PyTorch 量化 emulator 骨架 | workload/arch YAMLs, quant_emulator.py |
| 2 | 跑 10 个配置的 AccelForge mapping + breakdown | 跑 10 configs × 3 shapes 的精度评估 | energy/latency 表格, MSE/SQNR 表格 |
| 3 | 整理数据，画 energy breakdown + Pareto 图 | Workload-specificity 分析 | 所有 figures |
| 4 | — | — | 写 report + presentation |

### 最小可行版本（如果时间紧）

砍掉的东西：
- 只做 **6 个配置**：C0, C1, C3, C5, C7, C8（覆盖 0/1/2 level 和关键变体）
- 只做 **1 个 GEMM shape**（K=4096）
- 精度评估用随机权重（选项 A）
- Workload-specificity 用 K=1024 vs K=4096 两个 shape 对比

这样硬件侧只需要写 2 个 workload + 5 个 arch 变体，软件侧只需一个 ~150 行 Python 脚本。
