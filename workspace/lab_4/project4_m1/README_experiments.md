# Project 4 Sweep Toolkit

This directory now contains a scriptable experiment flow that keeps
`project4_m1_quantization_workload.ipynb` as the prototype, but moves the
actual proposal and milestone-3 sweeps into repeatable Python entrypoints.
The generated CSV/JSON artifacts under `results/` are the source of truth for
proposal readiness; the notebook should be treated as a thin frontend for
launching commands and previewing those saved outputs.

## Files

- `experiment_defs.py`
  Shared workload/config/architecture definitions and the checked-in default
  manifest layout.
- `run_sweeps.py`
  Generates workload + architecture YAML files, runs AccelForge hardware sweeps
  when available, runs the pure-Python accuracy emulator, and replays the
  notebook's legacy baselines for validation.
- `analyze_results.py`
  Joins hardware and accuracy summaries, computes Pareto frontiers, selects
  best configs per `(workload, phase)`, computes phase-adaptive savings, and
  summarizes milestone-3 saturation points.
- `dump_accuracy_snapshot.py`
  Creates proposal accuracy `.npz` snapshots either directly from a Hugging
  Face checkpoint or from pre-extracted tensor files. It now includes local
  checkpoint helpers for Qwen2-VL and OpenVLA plus a module-inspection mode.
- `experiment_manifest.json`
  Default checked-in manifest covering:
  - proposal sweep: `10 configs x 3 workloads x 2 phases`
  - milestone-3 sweep: `3 configs x 3 workloads x 2 phases x 16 arch variants`
  - legacy validation: the original `4096 x 4096` notebook cases

## Usage

From the repo root:

```bash
python3 workspace/lab_4/project4_m1/run_sweeps.py write-manifest
python3 workspace/lab_4/project4_m1/dump_accuracy_snapshot.py from-hf-causal-lm \
  --model-id meta-llama/Llama-2-7b-hf \
  --module-path model.layers.0.mlp.up_proj \
  --prompt "Write a short summary of quantization-aware accelerator design." \
  --output workspace/lab_4/project4_m1/accuracy_inputs/llm_ffn_layer.npz
python3 workspace/lab_4/project4_m1/dump_accuracy_snapshot.py inspect-modules \
  --model-type qwen2-vl \
  --model-path workspace/lab_4/project4_m1/accuracy_inputs/Qwen2-VL-2B \
  --rows 3072 --cols 3072 --limit 20
python3 workspace/lab_4/project4_m1/dump_accuracy_snapshot.py from-qwen2-vl \
  --model-path workspace/lab_4/project4_m1/accuracy_inputs/Qwen2-VL-2B \
  --module-path PATH_FOUND_FROM_INSPECT \
  --image-path /path/to/example_image.jpg \
  --prompt "Describe the important objects in this scene." \
  --output workspace/lab_4/project4_m1/accuracy_inputs/vlm_vision_gemm.npz
python3 workspace/lab_4/project4_m1/dump_accuracy_snapshot.py inspect-modules \
  --model-type openvla \
  --model-path workspace/lab_4/project4_m1/accuracy_inputs/openvla-7b \
  --rows 256 --limit 20
python3 workspace/lab_4/project4_m1/dump_accuracy_snapshot.py from-openvla \
  --model-path workspace/lab_4/project4_m1/accuracy_inputs/openvla-7b \
  --module-path PATH_FOUND_FROM_INSPECT \
  --image-path /path/to/robot_scene.jpg \
  --instruction "pick up the red block" \
  --output workspace/lab_4/project4_m1/accuracy_inputs/vla_action_head.npz
python3 workspace/lab_4/project4_m1/run_sweeps.py run-accuracy --suite proposal
python3 workspace/lab_4/project4_m1/run_sweeps.py run-hardware --suite proposal --jobs 4
python3 workspace/lab_4/project4_m1/run_sweeps.py run-hardware --suite milestone3 --jobs 4
python3 workspace/lab_4/project4_m1/run_sweeps.py verify-legacy
python3 workspace/lab_4/project4_m1/analyze_results.py
```

## Notes

- The hardware runner generates concrete `arch.yaml` and `workload.yaml`
  files under `generated/<suite>/<run_id>/`.
- The hardware sweep requires an environment with `accelforge` installed.
  If it is missing, the runner still generates inputs and records
  `generated_only` rows in the CSV summaries.
- Hardware progress is printed per case, and each completed case is appended
  to the CSV summary immediately. `--jobs N` runs independent hardware cases
  in parallel; start with `--jobs 2` or `--jobs 4` and adjust based on memory
  pressure in your AccelForge environment.
- Hardware runs resume by default: rows already recorded as `ok` in the target
  summary CSV are skipped. Pass `--rerun-ok` when you intentionally want to
  recompute successful cases.
- The accuracy pipeline is dependency-free and uses a deterministic
  pure-Python quantization emulator in `--input-mode debug`. The default
  proposal path (`--input-mode proposal`) requires representative tensor
  snapshots configured in the manifest under `accuracy_inputs`.
- For Qwen2-VL and OpenVLA, use `dump_accuracy_snapshot.py inspect-modules`
  first to discover a real module path from your local checkpoint, then use
  `from-qwen2-vl` or `from-openvla` to write the `.npz` snapshot.
- Proposal accuracy runs now fail closed: missing tensor snapshots are
  recorded as `missing_inputs` rows instead of silently falling back to
  synthetic Gaussian data.
- The default manifest points `accuracy_inputs` to relative paths under
  `project4_m1/accuracy_inputs/` so the same manifest works on the host and
  inside a containerized workspace mount.
- Proposal hardware workloads now use mapper-safe encodings: two-level
  tensor-granular coarse scales are emitted as singleton ranks instead of
  zero-rank tensors, and proposal one-level/two-level sweeps split large
  output-channel dimensions into small ranks to avoid AccelForge rank-width
  overflows on large LLM cases.
- `C7` is the corrected proposal NVFP4-like reference with tensor-granular
  coarse scaling. `LEGACY_NVFP4_FULL` preserves the historical notebook
  row-wise coarse-scale behavior for validation only.
- `analyze_results.py` writes `results/analysis_status.json`. The notebook
  uses this status file to decide whether proposal-derived CSVs and figures
  should be shown or marked as skipped.
