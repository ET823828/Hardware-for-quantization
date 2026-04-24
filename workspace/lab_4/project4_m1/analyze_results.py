from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiment_defs import (
    DEFAULT_ALPHA_BY_WORKLOAD,
    DEFAULT_ACCURACY_FLOOR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SATURATION_TOLERANCE,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dir,
    get_quant_config,
    manifest_or_default,
    manifest_runs,
)


PROPOSAL_HARDWARE_CSV = RESULTS_DIR / "proposal_hardware_summary.csv"
MILESTONE3_HARDWARE_CSV = RESULTS_DIR / "milestone3_hardware_summary.csv"
ACCURACY_CSV = RESULTS_DIR / "accuracy_summary.csv"
COMBINED_CSV = RESULTS_DIR / "proposal_combined_summary.csv"
BEST_CONFIGS_CSV = RESULTS_DIR / "proposal_best_configs.csv"
PARETO_CSV = RESULTS_DIR / "proposal_pareto_frontier.csv"
ADAPTIVE_CSV = RESULTS_DIR / "phase_adaptive_savings.csv"
M3_SATURATION_CSV = RESULTS_DIR / "milestone3_saturation.csv"
ANALYSIS_STATUS_JSON = RESULTS_DIR / "analysis_status.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: str | float | int | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_status(payload: dict[str, Any]) -> None:
    ensure_dir(ANALYSIS_STATUS_JSON.parent)
    ANALYSIS_STATUS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def output_elements(row: dict[str, Any]) -> int:
    return int(row["m"]) * int(row["n"])


def effective_bits_per_weight(row: dict[str, Any]) -> float:
    config = get_quant_config(row["config_id"])
    if isinstance(config, dict):
        raise ValueError(f"Effective bits are only defined for proposal configs, got {row['config_id']}")
    if config.topology == "zero_level":
        return 4.0
    if config.topology == "one_level":
        assert config.block_size is not None and config.fine_scale_bits is not None
        return 4.0 + config.fine_scale_bits / config.block_size
    if config.topology == "two_level":
        assert config.block_size is not None
        assert config.fine_scale_bits is not None
        assert config.coarse_scale_bits is not None
        fine_overhead = config.fine_scale_bits / config.block_size
        if config.coarse_granularity == "tensor":
            coarse_overhead = config.coarse_scale_bits / (int(row["n"]) * int(row["k"]))
        elif config.coarse_granularity == "row":
            coarse_overhead = config.coarse_scale_bits / int(row["k"])
        else:
            coarse_overhead = 0.0
        return 4.0 + fine_overhead + coarse_overhead
    raise ValueError(f"Unsupported topology for effective bits: {config.topology}")


def energy_axis_value(row: dict[str, Any]) -> float | None:
    return to_float(row.get("energy_pj_per_output"))


def proposal_rows() -> list[dict[str, Any]]:
    hardware_rows = [row for row in load_csv(PROPOSAL_HARDWARE_CSV) if row.get("status") == "ok"]
    accuracy_rows = [row for row in load_csv(ACCURACY_CSV) if row.get("status") == "ok"]
    accuracy_index = {
        (row["workload_id"], row["phase_id"], row["config_id"]): row
        for row in accuracy_rows
    }
    combined: list[dict[str, Any]] = []
    for hardware in hardware_rows:
        accuracy = accuracy_index.get((hardware["workload_id"], hardware["phase_id"], hardware["config_id"]))
        row = dict(hardware)
        if accuracy:
            row["cosine_similarity"] = accuracy["cosine_similarity"]
            row["sqnr_db"] = accuracy["sqnr_db"]
            row["m_eval"] = accuracy["m_eval"]
            row["n_eval"] = accuracy["n_eval"]
            row["k_eval"] = accuracy["k_eval"]
            row["input_source"] = accuracy.get("input_source", "")
        else:
            row["cosine_similarity"] = ""
            row["sqnr_db"] = ""
            row["m_eval"] = ""
            row["n_eval"] = ""
            row["k_eval"] = ""
            row["input_source"] = ""
        energy_total = to_float(row.get("energy_total_pj")) or to_float(row.get("energy_pj"))
        row["energy_total_pj"] = energy_total if energy_total is not None else ""
        if energy_total is not None:
            row["energy_pj_per_output"] = energy_total / output_elements(row)
        else:
            row["energy_pj_per_output"] = ""
        bits = effective_bits_per_weight(row)
        row["effective_bits_per_weight"] = bits
        row["compression_vs_fp16"] = 16.0 / bits if bits else ""
        combined.append(row)
    return combined


def is_pareto_row(target: dict[str, Any], peers: list[dict[str, Any]]) -> bool:
    target_energy = energy_axis_value(target)
    target_cos = to_float(target["cosine_similarity"])
    if target_energy is None or target_cos is None:
        return False
    for peer in peers:
        if peer is target:
            continue
        peer_energy = energy_axis_value(peer)
        peer_cos = to_float(peer["cosine_similarity"])
        if peer_energy is None or peer_cos is None:
            continue
        dominates = (
            peer_energy <= target_energy
            and peer_cos >= target_cos
            and (peer_energy < target_energy or peer_cos > target_cos)
        )
        if dominates:
            return False
    return True


def select_best_phase_configs(rows: list[dict[str, Any]], accuracy_floor: float) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    keys = sorted({(row["workload_id"], row["phase_id"]) for row in rows})
    for workload_id, phase_id in keys:
        candidates = [row for row in rows if row["workload_id"] == workload_id and row["phase_id"] == phase_id]
        passing = [row for row in candidates if (to_float(row["cosine_similarity"]) or 0.0) >= accuracy_floor]
        pool = passing or candidates
        best = min(
            pool,
            key=lambda row: (
                to_float(row["energy_total_pj"]) if to_float(row["energy_total_pj"]) is not None else float("inf"),
                -(to_float(row["cosine_similarity"]) if to_float(row["cosine_similarity"]) is not None else 0.0),
            ),
        )
        best_rows.append(
            {
                "workload_id": workload_id,
                "phase_id": phase_id,
                "config_id": best["config_id"],
                "energy_total_pj": best["energy_total_pj"],
                "energy_pj_per_output": best["energy_pj_per_output"],
                "latency_cycles": best["latency_cycles"],
                "area_m2": best["area_m2"],
                "cosine_similarity": best["cosine_similarity"],
                "sqnr_db": best["sqnr_db"],
                "effective_bits_per_weight": best["effective_bits_per_weight"],
                "compression_vs_fp16": best["compression_vs_fp16"],
                "meets_accuracy_floor": "yes" if best in passing else "no",
            }
        )
    return best_rows


def choose_best_fixed(workload_rows: list[dict[str, Any]], alpha: float, accuracy_floor: float) -> dict[str, Any] | None:
    configs = sorted({row["config_id"] for row in workload_rows})
    candidates: list[dict[str, Any]] = []
    for config_id in configs:
        decode = next((row for row in workload_rows if row["config_id"] == config_id and row["phase_id"] == "decode"), None)
        prefill = next((row for row in workload_rows if row["config_id"] == config_id and row["phase_id"] == "prefill"), None)
        if not decode or not prefill:
            continue
        decode_energy = to_float(decode["energy_total_pj"])
        prefill_energy = to_float(prefill["energy_total_pj"])
        decode_cos = to_float(decode["cosine_similarity"]) or 0.0
        prefill_cos = to_float(prefill["cosine_similarity"]) or 0.0
        if decode_energy is None or prefill_energy is None:
            continue
        candidates.append(
            {
                "config_id": config_id,
                "weighted_energy": alpha * prefill_energy + (1.0 - alpha) * decode_energy,
                "min_cosine": min(decode_cos, prefill_cos),
                "meets_floor": decode_cos >= accuracy_floor and prefill_cos >= accuracy_floor,
            }
        )
    if not candidates:
        return None
    pool = [candidate for candidate in candidates if candidate["meets_floor"]] or candidates
    return min(pool, key=lambda item: (item["weighted_energy"], -item["min_cosine"]))


def proposal_completion_status() -> dict[str, Any]:
    manifest = manifest_or_default(DEFAULT_MANIFEST_PATH)
    proposal_runs = manifest_runs(manifest, "proposal")
    hardware_rows = load_csv(PROPOSAL_HARDWARE_CSV)
    accuracy_rows = load_csv(ACCURACY_CSV)
    expected_hardware = len({row["run_id"] for row in proposal_runs})
    expected_accuracy = len({(row["workload_id"], row["phase_id"], row["config_id"]) for row in proposal_runs})
    observed_hardware = len({row["run_id"] for row in hardware_rows})
    observed_accuracy = len({(row["workload_id"], row["phase_id"], row["config_id"]) for row in accuracy_rows})

    hardware_status_counts: dict[str, int] = {}
    for row in hardware_rows:
        status = row.get("status", "unknown")
        hardware_status_counts[status] = hardware_status_counts.get(status, 0) + 1

    accuracy_status_counts: dict[str, int] = {}
    for row in accuracy_rows:
        status = row.get("status", "unknown")
        accuracy_status_counts[status] = accuracy_status_counts.get(status, 0) + 1

    hardware_complete = (
        bool(hardware_rows)
        and observed_hardware == expected_hardware
        and all(row.get("status") == "ok" for row in hardware_rows)
    )
    accuracy_complete = bool(accuracy_rows) and all(
        row.get("status") == "ok"
        and bool(row.get("input_source"))
        and not str(row.get("input_source", "")).startswith("debug::")
        and Path(str(row.get("input_source", ""))).exists()
        for row in accuracy_rows
    ) and observed_accuracy == expected_accuracy
    issues: list[str] = []
    if not hardware_rows:
        issues.append("No proposal hardware summary rows were found.")
    elif not hardware_complete:
        issues.append(f"Proposal hardware rows are incomplete: {hardware_status_counts}")
        if observed_hardware != expected_hardware:
            issues.append(f"Proposal hardware row count mismatch: expected {expected_hardware}, found {observed_hardware}.")
    if not accuracy_rows:
        issues.append("No proposal accuracy summary rows were found.")
    elif not accuracy_complete:
        issues.append("Proposal accuracy rows are incomplete or still using stale/debug inputs.")
        issues.append(f"Proposal accuracy status counts: {accuracy_status_counts}")
        if observed_accuracy != expected_accuracy:
            issues.append(f"Proposal accuracy row count mismatch: expected {expected_accuracy}, found {observed_accuracy}.")

    return {
        "proposal_ready": hardware_complete and accuracy_complete,
        "hardware_status_counts": hardware_status_counts,
        "accuracy_status_counts": accuracy_status_counts,
        "hardware_row_count": len(hardware_rows),
        "accuracy_row_count": len(accuracy_rows),
        "observed_hardware_rows": observed_hardware,
        "observed_accuracy_rows": observed_accuracy,
        "issues": issues,
        "expected_hardware_rows": expected_hardware,
        "expected_accuracy_rows": expected_accuracy,
    }


def analyze_proposal(accuracy_floor: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    completion = proposal_completion_status()
    if not completion["proposal_ready"]:
        write_status(
            {
                "proposal_ready": False,
                "issues": completion["issues"],
                "hardware_status_counts": completion["hardware_status_counts"],
                "accuracy_status_counts": completion["accuracy_status_counts"],
                "generated_artifacts": [],
            }
        )
        return [], [], []

    rows = proposal_rows()
    write_csv(COMBINED_CSV, rows)
    pareto_rows: list[dict[str, Any]] = []
    for workload_id in sorted({row["workload_id"] for row in rows}):
        for phase_id in sorted({row["phase_id"] for row in rows}):
            peers = [row for row in rows if row["workload_id"] == workload_id and row["phase_id"] == phase_id]
            for row in peers:
                row_with_frontier = dict(row)
                row_with_frontier["is_pareto"] = "yes" if is_pareto_row(row, peers) else "no"
                pareto_rows.append(row_with_frontier)
    write_csv(PARETO_CSV, pareto_rows)

    best_rows = select_best_phase_configs(rows, accuracy_floor)
    write_csv(BEST_CONFIGS_CSV, best_rows)

    adaptive_rows: list[dict[str, Any]] = []
    for workload_id in sorted({row["workload_id"] for row in rows}):
        workload_rows = [row for row in rows if row["workload_id"] == workload_id]
        prefill_best = next(row for row in best_rows if row["workload_id"] == workload_id and row["phase_id"] == "prefill")
        decode_best = next(row for row in best_rows if row["workload_id"] == workload_id and row["phase_id"] == "decode")
        prefill_energy = float(prefill_best["energy_total_pj"])
        decode_energy = float(decode_best["energy_total_pj"])

        nvfp4_decode = next(row for row in workload_rows if row["config_id"] == "C7" and row["phase_id"] == "decode")
        nvfp4_prefill = next(row for row in workload_rows if row["config_id"] == "C7" and row["phase_id"] == "prefill")
        for alpha in (0.2, 0.5, 0.8):
            adaptive_energy = alpha * prefill_energy + (1.0 - alpha) * decode_energy
            fixed_nvfp4_energy = alpha * float(nvfp4_prefill["energy_total_pj"]) + (1.0 - alpha) * float(nvfp4_decode["energy_total_pj"])
            best_fixed = choose_best_fixed(workload_rows, alpha, accuracy_floor)
            best_fixed_energy = best_fixed["weighted_energy"] if best_fixed else None
            adaptive_rows.append(
                {
                    "workload_id": workload_id,
                    "alpha_prefill": alpha,
                    "adaptive_prefill_config": prefill_best["config_id"],
                    "adaptive_decode_config": decode_best["config_id"],
                    "adaptive_energy_pj": adaptive_energy,
                    "fixed_nvfp4_energy_pj": fixed_nvfp4_energy,
                    "best_fixed_config": best_fixed["config_id"] if best_fixed else "",
                    "best_fixed_energy_pj": best_fixed_energy if best_fixed is not None else "",
                    "adaptive_vs_nvfp4_pct": 100.0 * (fixed_nvfp4_energy - adaptive_energy) / fixed_nvfp4_energy,
                    "adaptive_vs_best_fixed_pct": (
                        100.0 * (best_fixed_energy - adaptive_energy) / best_fixed_energy
                        if best_fixed_energy not in (None, 0.0)
                        else ""
                    ),
                }
            )
    write_csv(ADAPTIVE_CSV, adaptive_rows)
    write_status(
        {
            "proposal_ready": True,
            "issues": [],
            "hardware_status_counts": completion["hardware_status_counts"],
            "accuracy_status_counts": completion["accuracy_status_counts"],
            "generated_artifacts": [
                str(COMBINED_CSV),
                str(BEST_CONFIGS_CSV),
                str(PARETO_CSV),
                str(ADAPTIVE_CSV),
            ],
        }
    )
    return rows, best_rows, adaptive_rows


def analyze_milestone3(saturation_tolerance: float) -> list[dict[str, Any]]:
    rows = [row for row in load_csv(MILESTONE3_HARDWARE_CSV) if row.get("status") == "ok"]
    saturation_rows: list[dict[str, Any]] = []
    keys = sorted({(row["workload_id"], row["phase_id"], row["config_id"]) for row in rows})
    for workload_id, phase_id, config_id in keys:
        peers = [
            row
            for row in rows
            if row["workload_id"] == workload_id
            and row["phase_id"] == phase_id
            and row["config_id"] == config_id
        ]
        latencies = [to_float(row["latency_cycles"]) for row in peers if to_float(row["latency_cycles"]) is not None]
        if not latencies:
            continue
        min_latency = min(latencies)
        near_min = [
            row
            for row in peers
            if to_float(row["latency_cycles"]) is not None
            and to_float(row["latency_cycles"]) <= min_latency * (1.0 + saturation_tolerance)
        ]
        saturated = min(
            near_min,
            key=lambda row: (
                to_float(row["area_m2"]) if to_float(row["area_m2"]) is not None else float("inf"),
                int(row["num_quantmac"]),
                int(row["num_rescalemac"]),
            ),
        )
        saturation_rows.append(
            {
                "workload_id": workload_id,
                "phase_id": phase_id,
                "config_id": config_id,
                "min_latency_cycles": min_latency,
                "saturated_arch_id": saturated["arch_id"],
                "saturated_num_quantmac": saturated["num_quantmac"],
                "saturated_num_rescalemac": saturated["num_rescalemac"],
                "saturated_area_m2": saturated["area_m2"],
                "saturated_latency_cycles": saturated["latency_cycles"],
                "bottleneck_component": saturated["bottleneck_component"],
            }
        )
    write_csv(M3_SATURATION_CSV, saturation_rows)
    return saturation_rows


def maybe_plot_pareto(rows: list[dict[str, Any]], adaptive_rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return

    ensure_dir(FIGURES_DIR)

    workloads = sorted({row["workload_id"] for row in rows})
    phases = ["decode", "prefill"]
    fig, axes = plt.subplots(len(workloads), len(phases), figsize=(12, 10), squeeze=False)
    for workload_index, workload_id in enumerate(workloads):
        for phase_index, phase_id in enumerate(phases):
            ax = axes[workload_index][phase_index]
            peers = [row for row in rows if row["workload_id"] == workload_id and row["phase_id"] == phase_id]
            pareto = [row for row in peers if is_pareto_row(row, peers)]
            ax.scatter(
                [float(row["energy_pj_per_output"]) for row in peers],
                [float(row["cosine_similarity"]) for row in peers],
                alpha=0.6,
                label="Configs",
            )
            pareto_sorted = sorted(pareto, key=lambda row: float(row["energy_pj_per_output"]))
            ax.plot(
                [float(row["energy_pj_per_output"]) for row in pareto_sorted],
                [float(row["cosine_similarity"]) for row in pareto_sorted],
                color="tab:red",
                linewidth=2,
                label="Pareto",
            )
            for row in peers:
                if row["config_id"] in {"C1", "C7"}:
                    ax.annotate(row["config_id"], (float(row["energy_pj_per_output"]), float(row["cosine_similarity"])))
            ax.set_title(f"{workload_id} / {phase_id}")
            ax.set_xlabel("Energy per output (pJ)")
            ax.set_ylabel("Cosine Similarity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pareto_panels.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    workloads = sorted({row["workload_id"] for row in adaptive_rows})
    for workload_id in workloads:
        peers = sorted(
            [row for row in adaptive_rows if row["workload_id"] == workload_id],
            key=lambda row: float(row["alpha_prefill"]),
        )
        ax.plot(
            [float(row["alpha_prefill"]) for row in peers],
            [float(row["adaptive_vs_nvfp4_pct"]) for row in peers],
            marker="o",
            label=f"{workload_id} vs NVFP4",
        )
    ax.set_xlabel("Prefill weight alpha")
    ax.set_ylabel("Adaptive Energy Savings (%)")
    ax.set_title("Phase-Adaptive vs Fixed NVFP4")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase_adaptive_savings.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    workloads = sorted({row["workload_id"] for row in adaptive_rows})
    x_positions = list(range(len(workloads)))
    width = 0.25
    adaptive_values = []
    fixed_values = []
    best_fixed_values = []
    for workload_id in workloads:
        target_alpha = DEFAULT_ALPHA_BY_WORKLOAD[workload_id]
        match = next(row for row in adaptive_rows if row["workload_id"] == workload_id and float(row["alpha_prefill"]) == target_alpha)
        adaptive_values.append(float(match["adaptive_energy_pj"]))
        fixed_values.append(float(match["fixed_nvfp4_energy_pj"]))
        best_fixed_values.append(float(match["best_fixed_energy_pj"]))
    ax.bar([x - width for x in x_positions], adaptive_values, width=width, label="Phase-adaptive")
    ax.bar(x_positions, fixed_values, width=width, label="Fixed NVFP4")
    ax.bar([x + width for x in x_positions], best_fixed_values, width=width, label="Best fixed")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(workloads)
    ax.set_ylabel("Weighted total energy (pJ)")
    ax.set_title("Default-alpha Strategy Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase_adaptive_strategy_bar.png", dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze proposal and milestone-3 sweep results.")
    parser.add_argument("--accuracy-floor", type=float, default=DEFAULT_ACCURACY_FLOOR)
    parser.add_argument("--saturation-tolerance", type=float, default=DEFAULT_SATURATION_TOLERANCE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, best_rows, adaptive_rows = analyze_proposal(args.accuracy_floor)
    milestone3_rows = analyze_milestone3(args.saturation_tolerance)
    maybe_plot_pareto(rows, adaptive_rows)
    if rows:
        print(f"Wrote combined proposal summary to {COMBINED_CSV}")
        print(f"Wrote best-config table to {BEST_CONFIGS_CSV}")
        print(f"Wrote Pareto table to {PARETO_CSV}")
        print(f"Wrote adaptive savings table to {ADAPTIVE_CSV}")
    else:
        print(f"Proposal analysis incomplete; see {ANALYSIS_STATUS_JSON}")
    print(f"Wrote milestone-3 saturation table to {M3_SATURATION_CSV}")
    print(f"Best rows: {len(best_rows)} | milestone3 saturation rows: {len(milestone3_rows)}")


if __name__ == "__main__":
    main()
