"""Generate proposal figures from AccelForge breakdown data."""
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

# ── Raw data from AccelForge breakdown ──────────────────────────────

# NVFP4 Full (W4A4) per-einsum energy (pJ)
w4a4_energy = {
    "TensorScale\n(A+W)":   61_362_176 * 2,
    "TensorQuant\n(A+W)":  101_974_016 * 2,
    "BlockScale\n(A+W)":    60_030_976 * 2,
    "BlockQuant\n(A+W)":    70_516_736 * 2,
    "MatMul\nNVFP4":     78_168_404_787,
    "Rescale\nBlock(A+W)": 59_008_824_115 * 2,
    "Rescale\nTensor(A+W)": 42_986_582_835 + 189_256_499,
}

# W4A4 per-einsum latency (cycles)
w4a4_latency = {
    "TensorScale\n(A+W)":   524_544 * 2,
    "TensorQuant\n(A+W)":  540_672 * 2,
    "BlockScale\n(A+W)":  1_048_576 * 2,
    "BlockQuant\n(A+W)":  1_048_576 * 2,
    "MatMul\nNVFP4":   1_073_741_824,
    "Rescale\nBlock(A+W)": 268_435_456 * 2,
    "Rescale\nTensor(A+W)": 134_217_728 + 1_573_120,
}

# Totals for the 3 configs
configs_total = {
    "Baseline\n(FP16)":  {"energy": 240_560_111_616, "latency": 1_073_741_824},
    "W4A16\n(weight-only)": {"energy": 240_548_052_992, "latency": 1_074_790_400},
    "W4A4\n(NVFP4 full)":  {"energy": 239_949_660_160, "latency": 1_752_728_320},
}

# ── Figure 1: W4A4 Energy Breakdown (grouped into categories) ──────

def fig1_energy_breakdown():
    # Group into: Quant/Scale, Core MatMul, Rescale
    quant_e = (61_362_176*2 + 101_974_016*2 + 60_030_976*2 + 70_516_736*2)
    matmul_e = 78_168_404_787
    rescale_block_e = 59_008_824_115 * 2
    rescale_tensor_e = 42_986_582_835 + 189_256_499

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

    ax.set_xlabel("Energy (×10⁹ pJ)")
    ax.set_title("NVFP4 W4A4 Pipeline Energy Breakdown — Rescale Dominates (67%)")
    ax.set_xlim(0, max(values/1e9) * 1.25)

    plt.tight_layout()
    fig.savefig(f"{OUT}/energy_breakdown.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/energy_breakdown.png", bbox_inches="tight")
    print("Saved energy_breakdown.pdf/png")
    plt.close(fig)


# ── Figure 2: Latency comparison with breakdown ──────────────────

def fig2_latency_comparison():
    # Baseline: all MatMul
    bl_matmul = 1_073_741_824

    # W4A16: MatMulQ + DequantW
    w16_matmul = 1_073_741_824
    w16_dequant = 1_048_576

    # W4A4 breakdown
    w4_quant = (524_544*2 + 540_672*2 + 1_048_576*2 + 1_048_576*2)
    w4_matmul = 1_073_741_824
    w4_rescale = 268_435_456*2 + 134_217_728 + 1_573_120

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
    ax.set_title("Latency Comparison: Rescale Adds 63% Overhead in W4A4")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(f"{OUT}/latency_comparison.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/latency_comparison.png", bbox_inches="tight")
    print("Saved latency_comparison.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_energy_breakdown()
