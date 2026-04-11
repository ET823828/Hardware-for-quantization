# Project 4 Milestone 1 Progress Report

## Current assessment

The current notebook and YAML files completed the task **pick a workload and augment the spec by adding Einsums to represent the quantization**. 

The baseline workload is a dense GEMM with one `MatMul` einsum, and the quantized workload introduces explicit NVFP4-style tensors and an added `DequantW` einsum before the quantized matmul. This is visible in the baseline workload and the NVFP4 workload YAMLs. 

They also show a valid AccelForge-style evaluation flow around that workload augmentation. The quantized mapping explicitly schedules both `DequantW` and `MatMulQ`, which is consistent with the goal of representing quantization overhead as explicit work rather than assuming it is free. 

## What is already done

### 1. Baseline workload established

A baseline GEMM workload was defined with iteration dimensions `m`, `n`, and `k`, 16-bit tensors `A`, `W`, and `Y`, and a single `MatMul` einsum. 

### 2. NVFP4-style workload augmentation added

The workload was modified to represent weight-only quantization by:

- splitting `k` into `kb` and `ki`, with `ki = 16`,
- introducing quantized weights `Wq` at 4 bits,
- introducing scale tensor `Sw`,
- introducing dequantized weights `Wdq`, and
- adding a new `DequantW` einsum before `MatMulQ`. 

This is a good first Milestone 1 implementation of the workload-side requirement because it makes quantization overhead explicit in the spec.

### 3. Mapping support added for both baseline and quantized cases

A baseline mapping exists for `MatMul`, and a quantized mapping exists that includes both `DequantW` and `MatMulQ`. 

### 4. Minimal architecture available for initial validation

A simple architecture with `DRAM`, `Buffer`, `PEArray`, and `MAC` was provided so the workload and mapping can be parsed and evaluated in AccelForge. 




| Item                        | Baseline   | NVFP4 weight-only    |
| --------------------------- | ---------- | -------------------- |
| Main op                     | `MatMul`   | `DequantW + MatMulQ` |
| Weight representation       | 16-bit `W` | 4-bit `Wq`           |
| Scale tensor                | None       | `Sw[n,kb]`           |
| Weight blocking             | None       | `k -> kb, ki=16`     |
| Extra quantization overhead | None       | explicit `DequantW`  |
| Energy_pj                   | 5902336.0  | 6230016.0            |
| Latency cycles              | 524288.0.  | 524288.0             |
