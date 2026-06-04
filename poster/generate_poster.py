from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "poster"
RESULTS_DIR = ROOT / "workspace/lab_4/project4_m1/results"
FIGURES_DIR = ROOT / "workspace/lab_4/project4_m1/figures"
PROPOSAL_FIGURES_DIR = ROOT / "project_proposal (1)/figures"

WIDTH, HEIGHT = 7200, 5400
MARGIN = 190
GAP = 58

BG = "#f7f9fc"
INK = "#132238"
MUTED = "#607086"
LINE = "#d6dee8"
CARD = "#ffffff"
NAVY = "#172554"
BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
SOFT_BLUE = "#eaf2ff"
SOFT_CYAN = "#e6f6fb"
SOFT_GREEN = "#eaf8ef"
SOFT_AMBER = "#fff5e6"
SOFT_RED = "#fdecec"


def font_path(*names: str) -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return name
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No suitable font found")


FONT_REG = font_path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = font_path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


BEST_ROWS = read_csv(RESULTS_DIR / "proposal_best_configs.csv")
SAVINGS_ROWS = read_csv(RESULTS_DIR / "phase_adaptive_savings.csv")


CONFIG_LABELS = {
    "C1": "C1: MXFP4-like",
    "C8": "C8: FP16 rescale + FP16 acc",
    "C9": "C9: hybrid 2-level",
}


def fmt_int(value: str | float) -> str:
    return f"{float(value):,.0f}"


def fmt_pct(value: str | float) -> str:
    return f"{float(value):.0f}%"


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.getbbox(candidate)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    line_gap: int = 10,
) -> int:
    x, y = xy
    for line in wrap_text(text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int = 28,
    fill: str = CARD,
    outline: str | None = LINE,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, accent: str = BLUE) -> tuple[int, int, int, int]:
    rounded_rect(draw, box, radius=34, fill=CARD, outline=LINE, width=3)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 20), radius=34, fill=accent)
    draw.rectangle((x0, y0 + 10, x1, y0 + 22), fill=accent)
    draw.text((x0 + 48, y0 + 48), title.upper(), font=f(42, True), fill=INK)
    return (x0 + 48, y0 + 122, x1 - 48, y1 - 42)


def draw_pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, fg: str = INK) -> int:
    x, y = xy
    font = f(31, True)
    pad_x, pad_y = 24, 13
    w = font.getbbox(text)[2] + pad_x * 2
    h = font.size + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=23, fill=fill)
    draw.text((x + pad_x, y + pad_y - 2), text, font=font, fill=fg)
    return x + w + 18


def paste_image_fit(
    canvas: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
    bg: str = CARD,
    padding: int = 0,
) -> None:
    x0, y0, x1, y1 = box
    img = Image.open(path).convert("RGBA")
    max_w = x1 - x0 - 2 * padding
    max_h = y1 - y0 - 2 * padding
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), bg)
    px = ((x1 - x0) - img.width) // 2
    py = ((y1 - y0) - img.height) // 2
    layer.alpha_composite(img, (px, py))
    canvas.alpha_composite(layer, (x0, y0))


def draw_metric(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str, label: str, color: str) -> None:
    x0, y0, x1, y1 = box
    rounded_rect(draw, box, radius=30, fill="#ffffff", outline="#c9d6e6", width=3)
    draw.text((x0 + 30, y0 + 28), value, font=f(78, True), fill=color)
    draw_wrapped(draw, (x0 + 34, y0 + 122), label, f(31, True), MUTED, x1 - x0 - 68, line_gap=6)


def draw_header(canvas: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    header = (MARGIN, 120, WIDTH - MARGIN, 760)
    draw.rounded_rectangle(header, radius=48, fill="#ffffff", outline=LINE, width=3)
    x0, y0, x1, y1 = header
    draw.rectangle((x0, y0, x1, y0 + 24), fill=NAVY)
    draw.text((x0 + 58, y0 + 60), "PHASE-ADAPTIVE QUANTIZATION", font=f(114, True), fill=INK)
    draw.text(
        (x0 + 62, y0 + 198),
        "Design space exploration of 4-bit MMA rescale pipelines for AI accelerators",
        font=f(47, False),
        fill=MUTED,
    )
    draw.text((x0 + 62, y0 + 286), "Yichong Zhang · Wenye Xiong", font=f(43, True), fill=CYAN)
    draw.text(
        (x0 + 62, y0 + 354),
        "Hardware-aware quantization for LLM, VLM, and VLA inference workloads",
        font=f(36, False),
        fill=MUTED,
    )
    draw_wrapped(
        draw,
        (x0 + 62, y0 + 446),
        "Thesis: one bundled 4-bit format is not globally optimal. A known accelerator target should choose the rescale datapath by workload and inference phase.",
        f(38, True),
        INK,
        4100,
        line_gap=10,
    )

    metric_w = 550
    metric_gap = 28
    mx = x1 - 3 * metric_w - 2 * metric_gap - 56
    my = y0 + 82
    draw_metric(draw, (mx, my, mx + metric_w, my + 240), "60+60", "completed hardware + accuracy rows", BLUE)
    draw_metric(
        draw,
        (mx + metric_w + metric_gap, my, mx + 2 * metric_w + metric_gap, my + 240),
        "50-60%",
        "weighted energy cut vs NVFP4-like C7",
        GREEN,
    )
    draw_metric(
        draw,
        (mx + 2 * metric_w + 2 * metric_gap, my, mx + 3 * metric_w + 2 * metric_gap, my + 240),
        "3",
        "selected datapaths across six cells",
        AMBER,
    )


def draw_motivation(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "1. Motivation", BLUE)
    y = draw_wrapped(
        draw,
        (x0, y0),
        "LLM inference alternates between prefill and decode, but common 4-bit formats bundle a fixed rescale pipeline for both.",
        f(35, False),
        INK,
        x1 - x0,
        line_gap=10,
    )
    y += 18
    items = [
        ("Decode, M=1", "memory-bandwidth-bound; weight movement dominates, so compression and metadata overhead are central."),
        ("Prefill, M=128", "compute/rescale-heavy; FP32 rescale and accumulation can dominate MMA energy."),
        ("Design gap", "NVFP4/MXFP4 are useful reference points, but each fixes levels, block size, scale format, and accumulator precision."),
    ]
    colors = [SOFT_BLUE, SOFT_CYAN, SOFT_AMBER]
    for idx, (head, body) in enumerate(items):
        draw.rounded_rectangle((x0, y, x1, y + 148), radius=24, fill=colors[idx], outline=None)
        draw.text((x0 + 26, y + 20), head, font=f(33, True), fill=INK)
        draw_wrapped(draw, (x0 + 26, y + 68), body, f(29, False), MUTED, x1 - x0 - 52, line_gap=7)
        y += 170


def draw_principles(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "2. Underlying Principles", CYAN)
    flow_y = y0
    steps = [
        ("Quantize", "A, W -> FP4"),
        ("MatMul", "FP4 products"),
        ("Rescale", "block / tensor"),
        ("Accumulate", "FP32 or FP16"),
        ("Select", "phase-specific"),
    ]
    step_w = (x1 - x0 - 4 * 24) // 5
    for i, (head, body) in enumerate(steps):
        sx = x0 + i * (step_w + 24)
        draw.rounded_rectangle((sx, flow_y, sx + step_w, flow_y + 124), radius=22, fill=SOFT_CYAN, outline="#b6dfe9", width=2)
        draw.text((sx + 20, flow_y + 20), head, font=f(29, True), fill=CYAN)
        draw_wrapped(draw, (sx + 20, flow_y + 64), body, f(25, False), INK, step_w - 40, line_gap=4)
        if i < len(steps) - 1:
            ax = sx + step_w + 6
            ay = flow_y + 62
            draw.line((ax, ay, ax + 14, ay), fill=CYAN, width=6)
            draw.polygon([(ax + 14, ay - 12), (ax + 14, ay + 12), (ax + 32, ay)], fill=CYAN)
    y = flow_y + 166

    knobs = [
        ("Rescale levels", "0, 1, or 2 levels; b=16/32 and tensor coarse scale"),
        ("Scale format", "FP32, FP16, or E8M0 power-of-two shift"),
        ("Accumulator", "FP32 safety vs FP16 lower energy"),
        ("Accuracy floor", "cosine similarity >= 0.98"),
    ]
    for title, body in knobs:
        draw.text((x0, y), title, font=f(31, True), fill=INK)
        draw_wrapped(draw, (x0 + 310, y - 2), body, f(29, False), MUTED, x1 - x0 - 310, line_gap=6)
        y += 82

    y += 18
    x = x0
    for label, fill, fg in [
        ("C1: MXFP4-like, b=32, E8M0, FP32 acc", SOFT_GREEN, GREEN),
        ("C7: NVFP4-like, 2-level FP32", SOFT_RED, RED),
        ("C8/C9: aggressive low-precision modes", SOFT_AMBER, AMBER),
    ]:
        x = draw_pill(draw, (x, y), label, fill, fg)
        if x > x1 - 900:
            x = x0
            y += 80


def draw_evaluation(canvas: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "3. Evaluation", GREEN)
    table_y = y0
    headers = ["Workload", "Shape (M,N,K)", "Role"]
    rows = [
        ("LLM FFN", "decode (1,11008,4096) / prefill (128,11008,4096)", "Llama-style feed-forward GEMM"),
        ("VLM Vision", "decode (1,3072,3072) / prefill (128,3072,3072)", "balanced vision GEMM"),
        ("VLA Action", "decode (1,256,4096) / prefill (128,256,4096)", "latency-sensitive action head"),
    ]
    col_w = [340, 710, x1 - x0 - 340 - 710]
    tx = x0
    for h, w in zip(headers, col_w):
        draw.rectangle((tx, table_y, tx + w, table_y + 72), fill=SOFT_GREEN)
        draw.text((tx + 18, table_y + 18), h, font=f(27, True), fill=INK)
        tx += w
    y = table_y + 72
    for r, row in enumerate(rows):
        tx = x0
        row_h = 112
        for cell, w in zip(row, col_w):
            draw.rectangle((tx, y, tx + w, y + row_h), fill="#ffffff" if r % 2 == 0 else "#f8fbf9", outline=LINE, width=2)
            draw_wrapped(draw, (tx + 18, y + 18), cell, f(25, True if tx == x0 else False), INK if tx == x0 else MUTED, w - 36, line_gap=4)
            tx += w
        y += row_h

    y += 36
    methods = [
        ("Hardware", "AccelForge/Timeloop-style cost modeling extracts total energy, energy per output, latency, area, and bottleneck component."),
        ("Accuracy", "A deterministic quantized-MMA emulator compares layer outputs against FP16 using cosine similarity and SQNR."),
        ("Sweep", "10 configs x 3 workloads x 2 phases = 60 hardware rows + 60 accuracy rows; all completed OK."),
    ]
    for title, body in methods:
        draw.text((x0, y), title, font=f(31, True), fill=GREEN)
        draw_wrapped(draw, (x0 + 230, y - 2), body, f(29, False), MUTED, x1 - x0 - 230, line_gap=6)
        y += 104

    img_box = (x0, y + 8, x1, y1)
    paste_image_fit(canvas, PROPOSAL_FIGURES_DIR / "energy_breakdown.png", img_box, bg=CARD, padding=8)


def draw_pareto(canvas: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "4. Pareto Results", AMBER)
    paste_image_fit(canvas, FIGURES_DIR / "pareto_panels.png", (x0, y0, x1, y1 - 82), bg=CARD, padding=4)
    draw_wrapped(
        draw,
        (x0, y1 - 70),
        "Each panel plots energy per output against cosine similarity. C7 is the NVFP4-like reference; C1 is the MXFP4-like reference.",
        f(27, False),
        MUTED,
        x1 - x0,
        line_gap=4,
    )


def draw_best_table(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "5. Best Configs @ Cosine >= 0.98", BLUE)
    headers = ["Workload", "Decode", "Prefill", "Analysis"]
    rows_by_workload: dict[str, dict[str, dict[str, str]]] = {}
    for row in BEST_ROWS:
        rows_by_workload.setdefault(row["workload_id"], {})[row["phase_id"]] = row
    analysis = {
        "LLM": "phase-adaptive: C8 clears decode accuracy; C1 is the cheaper prefill knee.",
        "VLM": "single simple mode: C1 wins both phases because the balanced GEMM tolerates E8M0 scaling.",
        "VLA": "phase-adaptive: C8 fits decode, while C9's hybrid scale path wins prefill.",
    }
    display_order = ["LLM", "VLM", "VLA"]
    col_w = [230, 500, 500, x1 - x0 - 1230]
    y = y0
    tx = x0
    for h, w in zip(headers, col_w):
        draw.rectangle((tx, y, tx + w, y + 72), fill=SOFT_BLUE)
        draw.text((tx + 16, y + 18), h, font=f(27, True), fill=INK)
        tx += w
    y += 72
    for idx, workload in enumerate(display_order):
        decode = rows_by_workload[workload]["decode"]
        prefill = rows_by_workload[workload]["prefill"]
        cells = [
            workload,
            f"{decode['config_id']}  {fmt_int(decode['energy_pj_per_output'])} pJ/out  cos {float(decode['cosine_similarity']):.3f}",
            f"{prefill['config_id']}  {fmt_int(prefill['energy_pj_per_output'])} pJ/out  cos {float(prefill['cosine_similarity']):.3f}",
            analysis[workload],
        ]
        row_h = 128
        tx = x0
        for ci, (cell, w) in enumerate(zip(cells, col_w)):
            fill = "#ffffff" if idx % 2 == 0 else "#f8fbff"
            draw.rectangle((tx, y, tx + w, y + row_h), fill=fill, outline=LINE, width=2)
            color = INK if ci == 0 else MUTED
            bold = ci == 0 or (ci in (1, 2) and cell.startswith(("C1", "C8", "C9")))
            draw_wrapped(draw, (tx + 16, y + 18), cell, f(25, bold), color, w - 32, line_gap=5)
            tx += w
        y += row_h

    y += 35
    draw.rounded_rectangle((x0, y, x1, y + 128), radius=24, fill=SOFT_AMBER, outline="#f5d09b", width=2)
    draw_wrapped(
        draw,
        (x0 + 28, y + 24),
        "Result: no single configuration wins all six workload-phase cells; phase adaptation matters when decode and prefill stress different hardware costs.",
        f(31, True),
        INK,
        x1 - x0 - 56,
        line_gap=8,
    )


def draw_savings(canvas: Image.Image, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "6. Results Analysis", CYAN)
    metric_y = y0
    workload_default_alpha = {"LLM": "0.3", "VLM": "0.5", "VLA": "0.5"}
    default_rows = {row["workload_id"]: row for row in SAVINGS_ROWS if row["alpha_prefill"] == workload_default_alpha[row["workload_id"]]}
    metric_w = (x1 - x0 - 2 * 22) // 3
    for i, workload in enumerate(["LLM", "VLM", "VLA"]):
        row = default_rows[workload]
        bx = x0 + i * (metric_w + 22)
        draw.rounded_rectangle((bx, metric_y, bx + metric_w, metric_y + 190), radius=24, fill=SOFT_CYAN, outline="#b6dfe9", width=2)
        draw.text((bx + 24, metric_y + 18), workload, font=f(31, True), fill=INK)
        draw.text((bx + 24, metric_y + 62), fmt_pct(row["adaptive_vs_nvfp4_pct"]), font=f(66, True), fill=CYAN)
        draw_wrapped(draw, (bx + 24, metric_y + 130), "less weighted energy vs C7", f(24, True), MUTED, metric_w - 48, line_gap=4)
    y = metric_y + 226

    paste_image_fit(canvas, FIGURES_DIR / "phase_adaptive_strategy_bar.png", (x0, y, x1, y + 575), bg=CARD, padding=0)
    y += 594
    paste_image_fit(canvas, FIGURES_DIR / "phase_adaptive_savings.png", (x0, y, x1, y1 - 138), bg=CARD, padding=0)

    draw_wrapped(
        draw,
        (x0, y1 - 112),
        "Compared with best-fixed, adaptation is selective: VLA gains about 8.2%, LLM about 0.6%, and VLM 0% because both VLM phases already select C1.",
        f(28, False),
        MUTED,
        x1 - x0,
        line_gap=6,
    )


def draw_references(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = card(draw, box, "7. References", AMBER)
    refs = [
        "[1] OCP, Microscaling Formats (MX) Specification.",
        "[2] NVIDIA, Blackwell Architecture and NVFP4 Tensor Core documentation.",
        "[3] Parashar et al., Timeloop: A Systematic Approach to DNN Accelerator Evaluation, ISPASS 2019.",
        "[4] Wu et al., Accelergy: An Architecture-Level Energy Estimation Methodology for Accelerator Designs, ICCAD 2019.",
        "[5] MixPE: Quantization and Hardware Co-design for Efficient LLM Inference, arXiv:2411.16158.",
        "[6] AXE: Accumulator-Aware Post-Training Quantization, arXiv:2409.17092.",
        "[7] HiFloat4, Is Finer Better?, and related FP4/MX format studies, 2026.",
    ]
    y = y0
    for ref in refs:
        y = draw_wrapped(draw, (x0, y), ref, f(24, False), MUTED, x1 - x0, line_gap=4)
        y += 12


def draw_footer(draw: ImageDraw.ImageDraw) -> None:
    y = HEIGHT - 116
    draw.line((MARGIN, y, WIDTH - MARGIN, y), fill=LINE, width=3)
    draw.text(
        (MARGIN, y + 34),
        "Source artifacts: workspace/lab_4/project4_m1/results/*.csv and figures/*.png | Accuracy floor: cosine similarity >= 0.98",
        font=f(26, False),
        fill=MUTED,
    )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)

    draw_header(canvas, draw)

    top = 840
    bottom = HEIGHT - 170
    col_w = (WIDTH - 2 * MARGIN - 2 * GAP) // 3
    col1 = MARGIN
    col2 = MARGIN + col_w + GAP
    col3 = MARGIN + 2 * (col_w + GAP)

    row1_h = 1045
    row2_h = 2465
    row3_h = bottom - top - row1_h - row2_h - 2 * GAP

    draw_motivation(draw, (col1, top, col1 + col_w, top + row1_h))
    draw_principles(draw, (col2, top, col2 + col_w, top + row1_h))
    draw_evaluation(canvas, draw, (col3, top, col3 + col_w, top + row1_h))

    draw_pareto(canvas, draw, (col1, top + row1_h + GAP, col1 + 2 * col_w + GAP, top + row1_h + GAP + row2_h))
    draw_savings(canvas, draw, (col3, top + row1_h + GAP, col3 + col_w, top + row1_h + GAP + row2_h))

    lower_y = top + row1_h + GAP + row2_h + GAP
    draw_best_table(draw, (col1, lower_y, col1 + 2 * col_w + GAP, bottom))
    draw_references(draw, (col3, lower_y, col3 + col_w, bottom))

    draw_footer(draw)

    png_path = OUT_DIR / "phase_adaptive_quantization_poster.png"
    pdf_path = OUT_DIR / "phase_adaptive_quantization_poster.pdf"
    canvas.convert("RGB").save(png_path, quality=95)
    canvas.convert("RGB").save(pdf_path, "PDF", resolution=150.0)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
