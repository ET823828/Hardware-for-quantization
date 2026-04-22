# Project 4 Sweep Toolkit

This directory now contains a scriptable experiment flow that keeps
`project4_m1_quantization_workload.ipynb` as the prototype, but moves the
actual proposal and milestone-3 sweeps into repeatable Python entrypoints.

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
- `experiment_manifest.json`
  Default checked-in manifest covering:
  - proposal sweep: `10 configs x 3 workloads x 2 phases`
  - milestone-3 sweep: `3 configs x 3 workloads x 2 phases x 16 arch variants`
  - legacy validation: the original `4096 x 4096` notebook cases

## Usage

From the repo root:

```bash
python3 workspace/lab_4/project4_m1/run_sweeps.py write-manifest
python3 workspace/lab_4/project4_m1/run_sweeps.py run-accuracy
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
- The accuracy pipeline is dependency-free and uses a deterministic
  pure-Python quantization emulator. It samples small representative
  matrices while preserving the full reduction dimension `K`, which keeps
  the runtime manageable while still exercising accumulator stress.
- `C7` is wired to stay compatible with the current notebook's NVFP4-style
  storage assumptions: fine scales remain 8-bit metadata, while rescale
  computation is modeled as FP32.
