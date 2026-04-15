from __future__ import annotations

import json
import os
import traceback
import types
from pathlib import Path

try:
    import accelforge as af
    from accelforge.mapper import Metrics

    AF_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import depends on docker env
    af = None
    Metrics = None
    AF_AVAILABLE = False
    AF_IMPORT_ERROR = exc


ARCH_PARAMS = {
    "GLB_SIZE": 524288,
    "RF_SIZE": 1024,
    "PE_X": 8,
    "PE_Y": 8,
    "DRAM_BW": 8,
    "GLB_BW": 32,
}

KI = 16
MAX_SAFE_RANK_EXTENT = 128

SHAPES = {
    "decode_up": {"m": 1, "n": 11008, "k": 4096},
    "decode_down": {"m": 1, "n": 4096, "k": 11008},
    "prefill_up": {"m": 128, "n": 11008, "k": 4096},
    "prefill_down": {"m": 128, "n": 4096, "k": 11008},
}

CONFIGS = [
    "fp16_baseline",
    "w4a16_prequant",
    "w4a4_nvfp4_inference",
    "w4a4_ideal_no_overhead",
]


def find_lab_root() -> Path:
    cwd = Path.cwd()
    env_root = os.environ.get("AF_LAB4_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    if (cwd / "project4_m1_quantization_workload.ipynb").exists() or (cwd / "project4_m1").exists():
        return cwd
    nested = cwd / "workspace" / "lab_4"
    if nested.exists():
        return nested
    return cwd


ROOT = find_lab_root()
OUT_DIR = ROOT / "project4_m1"
SWEEP_DIR = OUT_DIR / "sweeps"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    SWEEP_DIR.mkdir(exist_ok=True)


def dump_yaml(path: Path, obj: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        if isinstance(obj, str):
            handle.write(obj)
        else:
            yaml.safe_dump(obj, handle, sort_keys=False)


def _extract_breakdown(df, best_idx: int, einsum_names: list[str]) -> tuple[dict, dict]:
    import pandas as pd

    row = df.iloc[best_idx]
    energy_breakdown = {}
    latency_breakdown = {}
    for col in df.columns:
        parts = col.split("<SEP>")
        if len(parts) < 2 or parts[0] not in einsum_names:
            continue
        einsum_name, metric = parts[0], parts[1]
        value = row[col]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(value):
            continue
        if metric == "energy":
            energy_breakdown[einsum_name] = energy_breakdown.get(einsum_name, 0.0) + value
        elif metric == "latency":
            latency_breakdown[einsum_name] = max(latency_breakdown.get(einsum_name, 0.0), value)
    return energy_breakdown, latency_breakdown


def _dominant_entry(breakdown: dict[str, float]) -> tuple[str | None, float]:
    if not breakdown:
        return None, 0.0
    name = max(breakdown, key=breakdown.get)
    return name, float(breakdown[name])


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _factor_rank(rank_name: str, size: int, max_extent: int = MAX_SAFE_RANK_EXTENT) -> tuple[list[str], dict[str, int]]:
    if size <= max_extent:
        return [rank_name], {rank_name: size}

    preferred_inner = [128, 64, 32, 16, 8, 4, 2]
    for inner in preferred_inner:
        if inner <= max_extent and size % inner == 0:
            outer = size // inner
            if outer <= max_extent:
                return [f"{rank_name}o", f"{rank_name}i"], {f"{rank_name}o": outer, f"{rank_name}i": inner}

    raise ValueError(
        f"Cannot split rank {rank_name}={size} into mapper-safe extents <= {max_extent}. "
        "Please choose a different shape or add another factoring rule."
    )


def _iteration_space(rank_sizes: dict[str, int]) -> dict[str, str]:
    return {rank: f"0 <= {rank} < {extent}" for rank, extent in rank_sizes.items()}


def make_architecture(rescale_energy_pj: float = 3.7, rescale_latency: int = 2) -> str:
    return f"""arch:
  nodes:
  - !Memory
    name: DRAM
    size: 99999999999
    leak_power: 0
    area: 0
    total_latency: "ceil(max((read_actions + metadata_read_actions) / {{{{ DRAM_BW }}}}, (write_actions + metadata_write_actions) / {{{{ DRAM_BW }}}}))"
    tensors: {{keep: ~Intermediates, may_keep: All}}
    actions:
    - {{name: read, energy: 10.0, bits_per_action: 64, latency: 0}}
    - {{name: write, energy: 10.0, bits_per_action: 64, latency: 0}}
    - {{name: metadata_read, energy: 2.0, bits_per_action: 16, latency: 0}}
    - {{name: metadata_write, energy: 2.0, bits_per_action: 16, latency: 0}}

  - !Memory
    name: GLB
    size: {{{{ GLB_SIZE }}}}
    leak_power: 0
    area: {{{{ GLB_SIZE * 7.4e-13 }}}}
    total_latency: "ceil(max(total_read_actions / {{{{ GLB_BW }}}}, total_write_actions / {{{{ GLB_BW }}}}))"
    tensors: {{keep: ~DRAM, may_keep: All}}
    actions:
    - {{name: read, energy: 5.0, bits_per_action: 32, latency: 0}}
    - {{name: write, energy: 5.0, bits_per_action: 32, latency: 0}}
    - {{name: metadata_read, energy: 2.0, bits_per_action: 16, latency: 0}}
    - {{name: metadata_write, energy: 2.0, bits_per_action: 16, latency: 0}}

  - !Memory
    name: RF
    size: {{{{ RF_SIZE }}}}
    leak_power: 0
    area: {{{{ RF_SIZE * 1e-12 }}}}
    total_latency: "ceil(max(total_read_actions / 2, total_write_actions / 2))"
    tensors: {{may_keep: All}}
    spatial:
    - {{name: X, fanout: {{{{ PE_X }}}}}}
    - {{name: Y, fanout: {{{{ PE_Y }}}}}}
    actions:
    - {{name: read, energy: 1.0, bits_per_action: 32, latency: 0}}
    - {{name: write, energy: 1.0, bits_per_action: 32, latency: 0}}

  - !Compute
    name: QuantMAC
    enabled: "'Ascl' in All or 'Sw' in All or ('A' in All and 'Sga' in All)"
    leak_power: 0
    area: 9e-11
    actions:
    - {{name: compute, energy: 1.0, latency: 1}}

  - !Compute
    name: FP4MAC
    enabled: "('Aq' in All and 'Wq' in All)"
    leak_power: 0
    area: 6e-11
    actions:
    - {{name: compute, energy: 0.2, latency: 1}}

  - !Compute
    name: RescaleMAC
    enabled: "('Yraw' in All and 'Sba' in All) or ('Ytmp' in All and 'Sbw' in All) or ('Yblk' in All and 'Sga' in All) or ('Ytmp2' in All and 'Sgw' in All)"
    leak_power: 0
    area: 7.2e-10
    actions:
    - {{name: compute, energy: {float(rescale_energy_pj)}, latency: {int(rescale_latency)}}}

  - !Compute
    name: FP16MAC
    enabled: "'A' in All and 'Y' in All"
    leak_power: 0
    area: 9e-11
    actions:
    - {{name: compute, energy: 1.0, latency: 1}}
"""


def _baseline_workload(m: int, n: int, k: int) -> dict:
    m_ranks, m_sizes = _factor_rank("m", m)
    n_ranks, n_sizes = _factor_rank("n", n)
    k_ranks, k_sizes = _factor_rank("k", k)
    rank_sizes = {}
    rank_sizes.update(m_sizes)
    rank_sizes.update(n_sizes)
    rank_sizes.update(k_sizes)
    return {
        "workload": {
            "iteration_space_shape": _iteration_space(rank_sizes),
            "bits_per_value": {"A": 16, "W": 16, "Y": 16},
            "einsums": [
                {
                    "name": "MatMul",
                    "tensor_accesses": [
                        {"name": "A", "projection": m_ranks + k_ranks, "density": 1.0},
                        {"name": "W", "projection": n_ranks + k_ranks, "density": 1.0},
                        {"name": "Y", "projection": m_ranks + n_ranks, "output": True},
                    ],
                }
            ],
        }
    }


def _w4a16_prequant_workload(m: int, n: int, k: int) -> dict:
    kb = k // KI
    m_ranks, m_sizes = _factor_rank("m", m)
    n_ranks, n_sizes = _factor_rank("n", n)
    kb_ranks, kb_sizes = _factor_rank("kb", kb)
    rank_sizes = {}
    rank_sizes.update(m_sizes)
    rank_sizes.update(n_sizes)
    rank_sizes.update(kb_sizes)
    rank_sizes["ki"] = KI
    return {
        "workload": {
            "iteration_space_shape": _iteration_space(rank_sizes),
            "bits_per_value": {"A": 16, "Wq": 4, "Sw": 16, "Wdq": 16, "Y": 16},
            "einsums": [
                {
                    "name": "DequantW",
                    "tensor_accesses": [
                        {"name": "Wq", "projection": n_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Sw", "projection": n_ranks + kb_ranks, "density": 1.0},
                        {"name": "Wdq", "projection": n_ranks + kb_ranks + ["ki"], "output": True},
                    ],
                },
                {
                    "name": "MatMulQ",
                    "tensor_accesses": [
                        {"name": "A", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Wdq", "projection": n_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Y", "projection": m_ranks + n_ranks, "output": True},
                    ],
                },
            ],
        }
    }


def _w4a4_inference_workload(m: int, n: int, k: int) -> dict:
    kb = k // KI
    m_ranks, m_sizes = _factor_rank("m", m)
    n_ranks, n_sizes = _factor_rank("n", n)
    kb_ranks, kb_sizes = _factor_rank("kb", kb)
    rank_sizes = {}
    rank_sizes.update(m_sizes)
    rank_sizes.update(n_sizes)
    rank_sizes.update(kb_sizes)
    rank_sizes["ki"] = KI
    return {
        "workload": {
            "iteration_space_shape": _iteration_space(rank_sizes),
            "bits_per_value": {
                "A": 16,
                "Sga": 32,
                "Ascl": 16,
                "Sba": 8,
                "Aq": 4,
                "Wq": 4,
                "Sbw": 8,
                "Sgw": 32,
                "Yraw": 32,
                "Ytmp": 32,
                "Yblk": 32,
                "Ytmp2": 32,
                "Y": 16,
            },
            "einsums": [
                {
                    "name": "TensorScaleA",
                    "tensor_accesses": [
                        {"name": "A", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Sga", "projection": m_ranks, "output": True},
                    ],
                },
                {
                    "name": "TensorQuantA",
                    "tensor_accesses": [
                        {"name": "A", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Sga", "projection": m_ranks, "density": 1.0},
                        {"name": "Ascl", "projection": m_ranks + kb_ranks + ["ki"], "output": True},
                    ],
                },
                {
                    "name": "BlockScaleA",
                    "tensor_accesses": [
                        {"name": "Ascl", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Sba", "projection": m_ranks + kb_ranks, "output": True},
                    ],
                },
                {
                    "name": "BlockQuantA",
                    "tensor_accesses": [
                        {"name": "Ascl", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Sba", "projection": m_ranks + kb_ranks, "density": 1.0},
                        {"name": "Aq", "projection": m_ranks + kb_ranks + ["ki"], "output": True},
                    ],
                },
                {
                    "name": "MatMulNVFP4",
                    "tensor_accesses": [
                        {"name": "Aq", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Wq", "projection": n_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Yraw", "projection": m_ranks + n_ranks + kb_ranks, "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockA",
                    "tensor_accesses": [
                        {"name": "Yraw", "projection": m_ranks + n_ranks + kb_ranks, "density": 1.0},
                        {"name": "Sba", "projection": m_ranks + kb_ranks, "density": 1.0},
                        {"name": "Ytmp", "projection": m_ranks + n_ranks + kb_ranks, "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockW",
                    "tensor_accesses": [
                        {"name": "Ytmp", "projection": m_ranks + n_ranks + kb_ranks, "density": 1.0},
                        {"name": "Sbw", "projection": n_ranks + kb_ranks, "density": 1.0},
                        {"name": "Yblk", "projection": m_ranks + n_ranks + kb_ranks, "output": True},
                    ],
                },
                {
                    "name": "RescaleTensorA",
                    "tensor_accesses": [
                        {"name": "Yblk", "projection": m_ranks + n_ranks + kb_ranks, "density": 1.0},
                        {"name": "Sga", "projection": m_ranks, "density": 1.0},
                        {"name": "Ytmp2", "projection": m_ranks + n_ranks, "output": True},
                    ],
                },
                {
                    "name": "RescaleTensorW",
                    "tensor_accesses": [
                        {"name": "Ytmp2", "projection": m_ranks + n_ranks, "density": 1.0},
                        {"name": "Sgw", "projection": n_ranks, "density": 1.0},
                        {"name": "Y", "projection": m_ranks + n_ranks, "output": True},
                    ],
                },
            ],
        }
    }


def _w4a4_ideal_workload(m: int, n: int, k: int) -> dict:
    kb = k // KI
    m_ranks, m_sizes = _factor_rank("m", m)
    n_ranks, n_sizes = _factor_rank("n", n)
    kb_ranks, kb_sizes = _factor_rank("kb", kb)
    rank_sizes = {}
    rank_sizes.update(m_sizes)
    rank_sizes.update(n_sizes)
    rank_sizes.update(kb_sizes)
    rank_sizes["ki"] = KI
    return {
        "workload": {
            "iteration_space_shape": _iteration_space(rank_sizes),
            "bits_per_value": {"Aq": 4, "Wq": 4, "Yraw": 32},
            "einsums": [
                {
                    "name": "MatMulNVFP4",
                    "tensor_accesses": [
                        {"name": "Aq", "projection": m_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Wq", "projection": n_ranks + kb_ranks + ["ki"], "density": 1.0},
                        {"name": "Yraw", "projection": m_ranks + n_ranks + kb_ranks, "output": True},
                    ],
                }
            ],
        }
    }


def make_workload(shape_name: str, m: int, n: int, k: int, config_name: str) -> dict:
    if k % KI != 0:
        raise ValueError(f"{shape_name}: k={k} must be divisible by ki={KI}")
    builders = {
        "fp16_baseline": _baseline_workload,
        "w4a16_prequant": _w4a16_prequant_workload,
        "w4a4_nvfp4_inference": _w4a4_inference_workload,
        "w4a4_ideal_no_overhead": _w4a4_ideal_workload,
    }
    if config_name not in builders:
        raise ValueError(f"Unknown config: {config_name}")
    return builders[config_name](m=m, n=n, k=k)


def auto_map(arch_file: Path, workload_file: Path, mapping_out_file: Path) -> types.SimpleNamespace:
    import numpy as np

    if not AF_AVAILABLE:
        raise RuntimeError(f"accelforge unavailable: {AF_IMPORT_ERROR}")
    spec = af.Spec.from_yaml(str(arch_file), str(workload_file), jinja_parse_data=ARCH_PARAMS)
    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
    all_mappings = spec.map_workload_to_arch()
    df = all_mappings.data
    einsum_names = list(all_mappings.einsum_names)

    energy_cols = [col for col in df.columns if "energy" in col.lower()]
    latency_cols = [col for col in df.columns if "latency" in col.lower()]
    best_idx = 0
    if energy_cols and latency_cols:
        edp = (df[energy_cols[0]] * df[latency_cols[0]]).values
        best_idx = int(np.argmin(edp))

    energy_total = float(df[energy_cols[0]].values[best_idx]) if energy_cols else None
    latency_total = float(df[latency_cols[0]].values[best_idx]) if latency_cols else None
    energy_breakdown, latency_breakdown = _extract_breakdown(df, best_idx, einsum_names)

    mapping_exported = False
    mapping_export_error = None
    try:
        mapping_out_file.write_text(all_mappings[best_idx].mapping().to_yaml())
        mapping_exported = True
    except Exception as exc:
        mapping_export_error = f"{type(exc).__name__}: {exc}"
        error_path = mapping_out_file.with_suffix(".export_error.txt")
        error_path.write_text(
            "AccelForge mapping export failed, but numeric results were still captured.\n\n"
            f"{mapping_export_error}\n\n"
            f"{traceback.format_exc()}"
        )

    breakdown_path = mapping_out_file.with_suffix(".breakdown.json")
    with open(breakdown_path, "w") as handle:
        json.dump(
            {
                "energy_total": energy_total,
                "latency_total": latency_total,
                "einsum_names": einsum_names,
                "energy_per_einsum": energy_breakdown,
                "latency_per_einsum": latency_breakdown,
                "mapping_exported": mapping_exported,
                "mapping_export_error": mapping_export_error,
            },
            handle,
            indent=2,
        )

    return types.SimpleNamespace(
        energy_pj=energy_total,
        latency_cycles=latency_total,
        energy_breakdown=energy_breakdown,
        latency_breakdown=latency_breakdown,
        einsum_names=einsum_names,
        mapping_exported=mapping_exported,
        mapping_export_error=mapping_export_error,
    )


def _load_saved_breakdown(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def run_case(shape_name: str, dims: dict[str, int], config_name: str) -> dict:
    ensure_dirs()
    case_dir = SWEEP_DIR / shape_name / config_name
    case_dir.mkdir(parents=True, exist_ok=True)

    arch = make_architecture()
    workload = make_workload(shape_name, dims["m"], dims["n"], dims["k"], config_name)

    arch_file = case_dir / "arch.yaml"
    workload_file = case_dir / "workload.yaml"
    mapping_file = case_dir / "mapping.yaml"
    breakdown_file = case_dir / "mapping.breakdown.json"

    dump_yaml(arch_file, arch)
    dump_yaml(workload_file, workload)

    energy_total = None
    latency_total = None
    energy_breakdown = {}
    latency_breakdown = {}
    einsum_names = [einsum["name"] for einsum in workload["workload"]["einsums"]]
    status = "generated_only"
    mapping_exported = None
    mapping_export_error = None

    if AF_AVAILABLE:
        result = auto_map(arch_file, workload_file, mapping_file)
        energy_total = result.energy_pj
        latency_total = result.latency_cycles
        energy_breakdown = result.energy_breakdown
        latency_breakdown = result.latency_breakdown
        einsum_names = result.einsum_names
        mapping_exported = result.mapping_exported
        mapping_export_error = result.mapping_export_error
        status = "mapped" if result.mapping_exported else "mapped_metrics_only"
    else:
        saved = _load_saved_breakdown(breakdown_file)
        if saved:
            energy_total = saved.get("energy_total")
            latency_total = saved.get("latency_total")
            energy_breakdown = saved.get("energy_per_einsum", {})
            latency_breakdown = saved.get("latency_per_einsum", {})
            einsum_names = saved.get("einsum_names", einsum_names)
            mapping_exported = saved.get("mapping_exported")
            mapping_export_error = saved.get("mapping_export_error")
            status = "loaded_saved_breakdown"

    dominant_energy_name, dominant_energy_value = _dominant_entry(energy_breakdown)
    dominant_latency_name, dominant_latency_value = _dominant_entry(latency_breakdown)

    return {
        "shape": shape_name,
        "config": config_name,
        "m": dims["m"],
        "n": dims["n"],
        "k": dims["k"],
        "kb": dims["k"] // KI,
        "ki": KI,
        "status": status,
        "arch_file": str(arch_file),
        "workload_file": str(workload_file),
        "mapping_file": str(mapping_file),
        "breakdown_file": str(breakdown_file),
        "mapping_exported": mapping_exported,
        "mapping_export_error": mapping_export_error,
        "energy_pj": energy_total,
        "latency_cycles": latency_total,
        "dominant_energy_einsum": dominant_energy_name,
        "dominant_energy_pj": dominant_energy_value,
        "dominant_latency_einsum": dominant_latency_name,
        "dominant_latency_cycles": dominant_latency_value,
        "energy_breakdown": energy_breakdown,
        "latency_breakdown": latency_breakdown,
        "einsum_names": einsum_names,
    }


def run_sweep(shapes: dict[str, dict[str, int]] | None = None, configs: list[str] | None = None):
    import pandas as pd

    ensure_dirs()
    shapes = shapes or SHAPES
    configs = configs or CONFIGS

    records = []
    breakdown_records = []
    for shape_name, dims in shapes.items():
        for config_name in configs:
            case_record = run_case(shape_name, dims, config_name)
            breakdown_records.append(
                {
                    "shape": shape_name,
                    "config": config_name,
                    "status": case_record["status"],
                    "energy_breakdown": case_record["energy_breakdown"],
                    "latency_breakdown": case_record["latency_breakdown"],
                    "einsum_names": case_record["einsum_names"],
                    "breakdown_file": case_record["breakdown_file"],
                }
            )
            records.append({k: v for k, v in case_record.items() if k not in {"energy_breakdown", "latency_breakdown", "einsum_names"}})

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        baseline_df = (
            results_df.loc[results_df["config"] == "fp16_baseline", ["shape", "energy_pj", "latency_cycles"]]
            .rename(columns={"energy_pj": "baseline_energy_pj", "latency_cycles": "baseline_latency_cycles"})
        )
        results_df = results_df.merge(baseline_df, on="shape", how="left")
        results_df["energy_norm_vs_fp16"] = results_df.apply(
            lambda row: _safe_ratio(row["energy_pj"], row["baseline_energy_pj"]),
            axis=1,
        )
        results_df["latency_norm_vs_fp16"] = results_df.apply(
            lambda row: _safe_ratio(row["latency_cycles"], row["baseline_latency_cycles"]),
            axis=1,
        )

        ideal_df = (
            results_df.loc[results_df["config"] == "w4a4_ideal_no_overhead", ["shape", "energy_pj", "latency_cycles"]]
            .rename(columns={"energy_pj": "ideal_energy_pj", "latency_cycles": "ideal_latency_cycles"})
        )
        results_df = results_df.merge(ideal_df, on="shape", how="left")
        results_df["energy_norm_vs_ideal"] = results_df.apply(
            lambda row: _safe_ratio(row["energy_pj"], row["ideal_energy_pj"]),
            axis=1,
        )
        results_df["latency_norm_vs_ideal"] = results_df.apply(
            lambda row: _safe_ratio(row["latency_cycles"], row["ideal_latency_cycles"]),
            axis=1,
        )
        results_df["energy_over_ideal_pj"] = results_df.apply(
            lambda row: None if row["energy_pj"] is None or row["ideal_energy_pj"] is None else float(row["energy_pj"]) - float(row["ideal_energy_pj"]),
            axis=1,
        )
        results_df["latency_over_ideal_cycles"] = results_df.apply(
            lambda row: None if row["latency_cycles"] is None or row["ideal_latency_cycles"] is None else float(row["latency_cycles"]) - float(row["ideal_latency_cycles"]),
            axis=1,
        )

    summary_csv = SWEEP_DIR / "results_summary.csv"
    results_df.to_csv(summary_csv, index=False)

    breakdown_json = SWEEP_DIR / "results_breakdowns.json"
    with open(breakdown_json, "w") as handle:
        json.dump(breakdown_records, handle, indent=2)

    return results_df


def plot_summary(results_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import numpy as np

    if results_df.empty:
        print("No sweep results to plot.")
        return {}

    mapped_df = results_df.dropna(subset=["energy_pj", "latency_cycles"])
    if mapped_df.empty:
        print("No mapped metrics found yet. Run in the Docker environment with AccelForge enabled.")
        return {}

    figures = {}

    configs = ["w4a16_prequant", "w4a4_nvfp4_inference", "w4a4_ideal_no_overhead"]
    metrics = ["energy_norm_vs_fp16", "latency_norm_vs_fp16"]
    config_labels = {
        "w4a16_prequant": "W4A16 prequant",
        "w4a4_nvfp4_inference": "W4A4 inference",
        "w4a4_ideal_no_overhead": "W4A4 ideal",
    }

    pivot = mapped_df[mapped_df["config"].isin(configs)].pivot(index="shape", columns="config", values=metrics)
    if not pivot.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        x = np.arange(len(pivot.index))
        width = 0.24
        for ax, metric, title in zip(axes, metrics, ["Energy vs FP16", "Latency vs FP16"]):
            for idx, config in enumerate(configs):
                values = pivot[(metric, config)].values
                ax.bar(x + (idx - 1) * width, values, width=width, label=config_labels[config])
            ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=15, ha="right")
            ax.set_ylabel("Normalized")
            ax.set_title(title)
        axes[0].legend()
        fig.tight_layout()
        figures["normalized_summary"] = fig

    inference_df = mapped_df[mapped_df["config"] == "w4a4_nvfp4_inference"].copy()
    if not inference_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        categories = [
            ("Activation quant", ["TensorScaleA", "TensorQuantA", "BlockScaleA", "BlockQuantA"]),
            ("Core matmul", ["MatMulNVFP4"]),
            ("Rescale", ["RescaleBlockA", "RescaleBlockW", "RescaleTensorA", "RescaleTensorW"]),
        ]
        bottoms = np.zeros(len(inference_df))
        for label, names in categories:
            values = []
            for _, row in inference_df.iterrows():
                breakdown = _load_saved_breakdown(Path(row["breakdown_file"])) or {}
                energy_breakdown = breakdown.get("energy_per_einsum", {})
                values.append(sum(float(energy_breakdown.get(name, 0.0)) for name in names))
            ax.bar(inference_df["shape"], values, bottom=bottoms, label=label)
            bottoms = bottoms + np.array(values)
        ax.set_ylabel("Energy (pJ)")
        ax.set_title("W4A4 inference energy breakdown by shape")
        ax.legend()
        fig.tight_layout()
        figures["inference_breakdown"] = fig

    gap_df = mapped_df[mapped_df["config"] == "w4a4_nvfp4_inference"].copy()
    if not gap_df.empty and gap_df["energy_over_ideal_pj"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(gap_df["shape"], gap_df["energy_over_ideal_pj"])
        axes[0].set_title("Energy overhead vs ideal")
        axes[0].set_ylabel("pJ")
        axes[1].bar(gap_df["shape"], gap_df["latency_over_ideal_cycles"])
        axes[1].set_title("Latency overhead vs ideal")
        axes[1].set_ylabel("cycles")
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        figures["overhead_vs_ideal"] = fig

    return figures


def plot_breakdowns(breakdown_records: list[dict]):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    inference_records = [record for record in breakdown_records if record["config"] == "w4a4_nvfp4_inference"]
    if not inference_records:
        print("No inference breakdown records available.")
        return None

    rows = []
    for record in inference_records:
        energy_breakdown = record.get("energy_breakdown", {})
        if not energy_breakdown and record.get("breakdown_file"):
            saved = _load_saved_breakdown(Path(record["breakdown_file"])) or {}
            energy_breakdown = saved.get("energy_per_einsum", {})
        for einsum_name, energy_value in energy_breakdown.items():
            rows.append({"shape": record["shape"], "einsum": einsum_name, "energy_pj": energy_value})

    if not rows:
        print("Inference breakdown records do not contain mapped energy values yet.")
        return None

    breakdown_df = pd.DataFrame(rows)
    pivot = breakdown_df.pivot(index="shape", columns="einsum", values="energy_pj").fillna(0.0)
    fig, ax = plt.subplots(figsize=(14, 6))
    bottoms = np.zeros(len(pivot.index))
    for einsum_name in pivot.columns:
        ax.bar(pivot.index, pivot[einsum_name].values, bottom=bottoms, label=einsum_name)
        bottoms = bottoms + pivot[einsum_name].values
    ax.set_ylabel("Energy (pJ)")
    ax.set_title("Per-einsum W4A4 inference energy breakdown")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def load_breakdown_records() -> list[dict]:
    breakdown_json = SWEEP_DIR / "results_breakdowns.json"
    if not breakdown_json.exists():
        return []
    with open(breakdown_json) as handle:
        return json.load(handle)


def sanity_check_workloads() -> dict:
    dims = SHAPES["decode_up"]
    inference = make_workload("decode_up", dims["m"], dims["n"], dims["k"], "w4a4_nvfp4_inference")
    ideal = make_workload("decode_up", dims["m"], dims["n"], dims["k"], "w4a4_ideal_no_overhead")

    inference_names = [einsum["name"] for einsum in inference["workload"]["einsums"]]
    assert "TensorScaleW" not in inference_names
    assert "TensorQuantW" not in inference_names
    assert "BlockScaleW" not in inference_names
    assert "BlockQuantW" not in inference_names

    inference_bits = inference["workload"]["bits_per_value"]
    assert "Wq" in inference_bits and "Sbw" in inference_bits and "Sgw" in inference_bits
    assert "W" not in inference_bits

    ideal_names = [einsum["name"] for einsum in ideal["workload"]["einsums"]]
    assert ideal_names == ["MatMulNVFP4"]

    return {
        "inference_einsums": inference_names,
        "ideal_einsums": ideal_names,
        "inference_bits": inference_bits,
        "shapes": SHAPES,
        "configs": CONFIGS,
    }
