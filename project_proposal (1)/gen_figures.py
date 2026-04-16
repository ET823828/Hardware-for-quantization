"""Generate proposal figures from AccelForge breakdown JSON files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

OUT = "figures"

# ── Load breakdown data from JSON files ───────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "workspace" / "lab_4" / "project4_m1"

def load_breakdown(name):
    path = DATA_DIR / name
    with open(path) as f:
        return json.load(f)

bk_baseline = load_breakdown("mapping_baseline.breakdown.json")
bk_w4a16    = load_breakdown("mapping_nvfp4_weight_only.breakdown.json")
bk_w4a4     = load_breakdown("mapping_nvfp4_full_auto.breakdown.json")


# ── Figure 1: W4A4 Energy Breakdown (grouped into categories) ────────

def fig1_energy_breakdown():
    e = bk_w4a4["energy_per_einsum"]

    quant_e = (e["TensorScaleA"] + e["TensorScaleW"]
             + e["TensorQuantA"] + e["TensorQuantW"]
             + e["BlockScaleA"]  + e["BlockScaleW"]
             + e["BlockQuantA"]  + e["BlockQuantW"])
    matmul_e = e["MatMulNVFP4"]
    rescale_block_e = e["RescaleBlockA"] + e["RescaleBlockW"]
    rescale_tensor_e = e["RescaleTensorA"] + e["RescaleTensorW"]

    total = quant_e + matmul_e + rescale_block_e + rescale_tensor_e

    categories = [
        "Quant &\nScale",
        "FP4 MatMul\n(core)",
        "Rescale\n(block-level)",
        "Rescale\n(tensor-level)",
    ]
    values = np.array([quant_e, matmul_e, rescale_block_e, rescale_tensor_e])
    pcts = values / total * 100

    colors = ["#A0CBE8", "#4E79A7", "#F28E2B", "#E15759"]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.barh(categories, values / 1e9, color=colors, edgecolor="white", linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold")

    rescale_pct = (rescale_block_e + rescale_tensor_e) / total * 100
    ax.set_xlabel("Energy (×10⁹ pJ)")
    ax.set_title(f"NVFP4 W4A4 Pipeline Energy Breakdown — Rescale Dominates ({rescale_pct:.0f}%)")
    ax.set_xlim(0, max(values/1e9) * 1.25)

    plt.tight_layout()
    fig.savefig(f"{OUT}/energy_breakdown.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/energy_breakdown.png", bbox_inches="tight")
    print("Saved energy_breakdown.pdf/png")
    plt.close(fig)


# ── Figure 2: Latency comparison with breakdown ─────────��────────────

def fig2_latency_comparison():
    bl_l = bk_baseline["latency_per_einsum"]
    w16_l = bk_w4a16["latency_per_einsum"]
    w4_l = bk_w4a4["latency_per_einsum"]

    # Baseline: all MatMul
    bl_matmul = bl_l["MatMul"]

    # W4A16: MatMulQ + DequantW
    w16_matmul = w16_l["MatMulQ"]
    w16_dequant = w16_l["DequantW"]

    # W4A4 breakdown
    w4_quant = sum(w4_l[k] for k in [
        "TensorScaleA", "TensorQuantA", "BlockScaleA", "BlockQuantA",
        "TensorScaleW", "TensorQuantW", "BlockScaleW", "BlockQuantW",
    ])
    w4_matmul = w4_l["MatMulNVFP4"]
    w4_rescale = sum(w4_l[k] for k in [
        "RescaleBlockA", "RescaleBlockW", "RescaleTensorA", "RescaleTensorW",
    ])

    configs = ["Baseline\n(FP16)", "W4A16\n(weight-only)", "W4A4\n(NVFP4 full)"]

    matmul_vals = np.array([bl_matmul, w16_matmul, w4_matmul]) / 1e6
    quant_vals = np.array([0, w16_dequant, w4_quant]) / 1e6
    rescale_vals = np.array([0, 0, w4_rescale]) / 1e6

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(configs))
    width = 0.55

    ax.bar(x, matmul_vals, width, label="MatMul (core)", color="#4E79A7")
    ax.bar(x, quant_vals, width, bottom=matmul_vals, label="Quant/Scale/Dequant", color="#A0CBE8")
    ax.bar(x, rescale_vals, width, bottom=matmul_vals + quant_vals, label="Rescale", color="#E15759")

    # Add total labels
    totals = matmul_vals + quant_vals + rescale_vals
    for i, t in enumerate(totals):
        ax.text(i, t + 15, f"{t:,.0f}M", ha="center", fontsize=8, fontweight="bold")

    # Overhead annotation
    overhead = (totals[2] - totals[0]) / totals[0] * 100
    ax.annotate(
        f"+{overhead:.0f}%\nlatency",
        xy=(2, totals[2]), xytext=(2.45, totals[2] * 0.75),
        fontsize=9, fontweight="bold", color="#E15759",
        arrowprops=dict(arrowstyle="->", color="#E15759", lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Latency (×10⁶ cycles)")
    ax.set_title(f"Latency Comparison: Rescale Adds {overhead:.0f}% Overhead in W4A4")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(f"{OUT}/latency_comparison.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/latency_comparison.png", bbox_inches="tight")
    print("Saved latency_comparison.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_energy_breakdown()
    fig2_latency_comparison()
