"""Generate proposal figures from inference-faithful sweep outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    }
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
SUMMARY_CSV = ROOT / "workspace" / "lab_4" / "project4_m1" / "sweeps" / "results_summary.csv"
BREAKDOWN_JSON = ROOT / "workspace" / "lab_4" / "project4_m1" / "sweeps" / "results_breakdowns.json"

OUT.mkdir(exist_ok=True)


def load_summary() -> list[dict]:
    with open(SUMMARY_CSV, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in [
            "m",
            "n",
            "k",
            "kb",
            "ki",
            "energy_pj",
            "latency_cycles",
            "baseline_energy_pj",
            "baseline_latency_cycles",
            "energy_norm_vs_fp16",
            "latency_norm_vs_fp16",
            "ideal_energy_pj",
            "ideal_latency_cycles",
            "energy_norm_vs_ideal",
            "latency_norm_vs_ideal",
            "energy_over_ideal_pj",
            "latency_over_ideal_cycles",
        ]:
            if row.get(key):
                row[key] = float(row[key])
        if row.get("mapping_exported") in {"True", "False"}:
            row["mapping_exported"] = row["mapping_exported"] == "True"
    return rows


def load_breakdowns() -> list[dict]:
    with open(BREAKDOWN_JSON) as handle:
        return json.load(handle)


def _save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    print(f"Saved {stem}.pdf/png")
    plt.close(fig)


def fig1_energy_breakdown(summary_rows: list[dict], breakdown_rows: list[dict]) -> None:
    """Stacked energy breakdown for W4A4 inference, averaged by decode vs prefill."""

    grouped = {"decode": [], "prefill": []}
    for record in breakdown_rows:
        if record["config"] != "w4a4_nvfp4_inference":
            continue
        bucket = "decode" if record["shape"].startswith("decode") else "prefill"
        grouped[bucket].append(record)

    categories = [
        ("Activation quant", ["TensorScaleA", "TensorQuantA", "BlockScaleA", "BlockQuantA"]),
        ("Core matmul", ["MatMulNVFP4"]),
        ("Rescale", ["RescaleBlockA", "RescaleBlockW", "RescaleTensorA", "RescaleTensorW"]),
    ]
    colors = {
        "Activation quant": "#A0CBE8",
        "Core matmul": "#4E79A7",
        "Rescale": "#E15759",
    }

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x_labels = ["Decode", "Prefill"]
    x = np.arange(len(x_labels))
    bottoms = np.zeros(len(x_labels))
    category_values = []

    for label, einsums in categories:
        vals = []
        for bucket in ["decode", "prefill"]:
            per_case_vals = []
            for record in grouped[bucket]:
                energy_breakdown = record.get("energy_breakdown", {})
                val = sum(float(energy_breakdown.get(name, 0.0)) for name in einsums)
                per_case_vals.append(val)
            vals.append(np.mean(per_case_vals))
        category_values.append(vals)
        ax.bar(x, vals, bottom=bottoms, color=colors[label], width=0.58, label=label)
        bottoms = bottoms + np.array(vals)

    totals = bottoms
    for idx, total in enumerate(totals):
        ax.text(x[idx], total * 1.01, f"{total/1e9:.1f}B", ha="center", va="bottom", fontsize=8, fontweight="bold")

    rescale_vals = np.array(category_values[2])
    for idx, pct in enumerate(rescale_vals / totals * 100):
        ax.text(x[idx], totals[idx] * 0.58, f"Rescale\n{pct:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Energy per GEMM (pJ)")
    ax.set_title("W4A4 Inference Energy Breakdown by Workload Regime")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    _save(fig, "energy_breakdown")


def fig2_sweep_summary(summary_rows: list[dict]) -> None:
    """Normalized energy/latency vs FP16 across shapes."""

    shape_order = ["decode_up", "decode_down", "prefill_up", "prefill_down"]
    shape_labels = ["Decode up", "Decode down", "Prefill up", "Prefill down"]
    config_order = ["w4a16_prequant", "w4a4_nvfp4_inference"]
    config_labels = {"w4a16_prequant": "W4A16 prequant", "w4a4_nvfp4_inference": "W4A4 inference"}
    colors = {"w4a16_prequant": "#F28E2B", "w4a4_nvfp4_inference": "#4E79A7"}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharex=True)
    x = np.arange(len(shape_order))
    width = 0.34

    for ax, metric, title in zip(
        axes,
        ["energy_norm_vs_fp16", "latency_norm_vs_fp16"],
        ["Energy vs FP16", "Latency vs FP16"],
    ):
        for idx, config in enumerate(config_order):
            vals = [
                next(row[metric] for row in summary_rows if row["shape"] == shape and row["config"] == config)
                for shape in shape_order
            ]
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, vals, width=width, color=colors[config], label=config_labels[config])
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(shape_labels, rotation=15, ha="right")
        ax.set_ylabel("Normalized")
        ax.set_title(title)

    axes[0].legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    _save(fig, "sweep_summary")


def fig3_overhead_vs_ideal(summary_rows: list[dict]) -> None:
    """Realized W4A4 inference cost relative to ideal no-overhead W4A4."""

    shape_order = ["decode_up", "decode_down", "prefill_up", "prefill_down"]
    shape_labels = ["Decode up", "Decode down", "Prefill up", "Prefill down"]
    inference_rows = {
        row["shape"]: row for row in summary_rows if row["config"] == "w4a4_nvfp4_inference"
    }

    energy_ratios = [inference_rows[shape]["energy_norm_vs_ideal"] for shape in shape_order]
    latency_ratios = [inference_rows[shape]["latency_norm_vs_ideal"] for shape in shape_order]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharex=True)
    bars_e = axes[0].bar(shape_labels, energy_ratios, color="#E15759")
    bars_l = axes[1].bar(shape_labels, latency_ratios, color="#76B7B2")

    axes[0].set_title("W4A4 Energy Gap vs Ideal")
    axes[0].set_ylabel("Realized / Ideal")
    axes[1].set_title("W4A4 Latency Gap vs Ideal")
    axes[1].set_ylabel("Realized / Ideal")
    for ax in axes:
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.tick_params(axis="x", rotation=15)

    for bars, vals, ax in [(bars_e, energy_ratios, axes[0]), (bars_l, latency_ratios, axes[1])]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}x", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, "overhead_vs_ideal")


if __name__ == "__main__":
    summary = load_summary()
    breakdowns = load_breakdowns()
    fig1_energy_breakdown(summary, breakdowns)
    fig2_sweep_summary(summary)
    fig3_overhead_vs_ideal(summary)
