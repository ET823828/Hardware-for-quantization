#!/usr/bin/env python3
"""Generate architecture modeling report PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)

OUTPUT = "architecture_modeling_report.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=letter,
    leftMargin=0.9*inch, rightMargin=0.9*inch,
    topMargin=0.8*inch, bottomMargin=0.8*inch,
)

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    'ReportTitle', parent=styles['Title'], fontSize=20, spaceAfter=6,
    textColor=HexColor('#1a1a2e'),
))
styles.add(ParagraphStyle(
    'Subtitle', parent=styles['Normal'], fontSize=11, spaceAfter=16,
    textColor=HexColor('#555555'), alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    'SectionHead', parent=styles['Heading1'], fontSize=14, spaceBefore=18,
    spaceAfter=8, textColor=HexColor('#1a1a2e'),
))
styles.add(ParagraphStyle(
    'SubHead', parent=styles['Heading2'], fontSize=12, spaceBefore=12,
    spaceAfter=6, textColor=HexColor('#2d3436'),
))
styles.add(ParagraphStyle(
    'Body', parent=styles['Normal'], fontSize=10, leading=14,
    spaceAfter=6, alignment=TA_JUSTIFY,
))
styles.add(ParagraphStyle(
    'Formula', parent=styles['Normal'], fontSize=10, leading=14,
    spaceAfter=6, alignment=TA_CENTER, fontName='Courier',
))
styles.add(ParagraphStyle(
    'Caption', parent=styles['Normal'], fontSize=9, leading=12,
    spaceAfter=10, alignment=TA_CENTER, textColor=HexColor('#555555'),
))
styles.add(ParagraphStyle(
    'Reference', parent=styles['Normal'], fontSize=9, leading=12,
    spaceAfter=4, leftIndent=20, firstLineIndent=-20,
))

BLUE = HexColor('#1a1a2e')
LIGHT_BLUE = HexColor('#dfe6e9')
HEADER_BG = HexColor('#2d3436')

def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#b2bec3')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_BLUE]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


story = []

# ============================================================
# Title Page
# ============================================================
story.append(Spacer(1, 1.5*inch))
story.append(Paragraph("NVFP4 Quantization Accelerator", styles['ReportTitle']))
story.append(Paragraph("Architecture Modeling Report", styles['ReportTitle']))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Hardware Design for AI Accelerators &mdash; Final Project", styles['Subtitle']))
story.append(Paragraph("Technology Node: 45 nm CMOS, 0.9 V", styles['Subtitle']))
story.append(Spacer(1, 0.3*inch))
story.append(HRFlowable(width="60%", thickness=1, color=HexColor('#b2bec3')))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("Modeling Framework: AccelForge (Timeloop-based)", styles['Subtitle']))
story.append(Paragraph(
    "Energy Reference: Horowitz, ISSCC 2014<br/>"
    "Area Reference: TSMC 45 nm 6T SRAM, Lab3 die-photo estimates",
    styles['Subtitle'],
))
story.append(PageBreak())

# ============================================================
# 1. Introduction
# ============================================================
story.append(Paragraph("1. Introduction", styles['SectionHead']))
story.append(Paragraph(
    "This report describes the architecture modeling methodology for an accelerator "
    "optimized for the NVIDIA FP4 (NVFP4) quantization pipeline. The NVFP4 format uses "
    "two-level scaling (per-tensor + per-block) to quantize FP16 activations and weights "
    "down to 4-bit floating point (E2M1), perform FP4 x FP4 matrix multiplication with "
    "FP32 accumulation, and then rescale the output back to FP16. The full pipeline "
    "consists of 13 einsum operations modeled explicitly in AccelForge.",
    styles['Body'],
))
story.append(Paragraph(
    "The architecture features a three-level memory hierarchy (DRAM, GLB, RF), "
    "a 2D spatial PE array (8 x 8 = 64 PEs), and three specialized compute units "
    "(QuantMAC, FP4MAC, RescaleMAC) with energy and area values derived from "
    "Horowitz's ISSCC 2014 reference data at the 45 nm node.",
    styles['Body'],
))

# ============================================================
# 2. Workload Specification
# ============================================================
story.append(Paragraph("2. Workload Specification", styles['SectionHead']))

story.append(Paragraph("2.1 Iteration Space", styles['SubHead']))
story.append(Paragraph(
    "The workload operates over a four-dimensional iteration space that results from "
    "splitting the original K dimension into block (kb) and intra-block (ki) components, "
    "where ki = 16 is the NVFP4 block size:",
    styles['Body'],
))
story.append(make_table(
    ['Dimension', 'Range', 'Description'],
    [
        ['m', '0 <= m < 64', 'Activation row (output row)'],
        ['n', '0 <= n < 64', 'Weight column (output column)'],
        ['kb', '0 <= kb < 8', 'Block index along K'],
        ['ki', '0 <= ki < 16', 'Intra-block index (NVFP4 block size)'],
    ],
    col_widths=[1.2*inch, 1.5*inch, 3.0*inch],
))
story.append(Paragraph("Table 1: Iteration space dimensions.", styles['Caption']))

story.append(Paragraph("2.2 Tensor Precision", styles['SubHead']))
story.append(Paragraph(
    "The NVFP4 pipeline uses mixed precision across 15 tensors. The key insight is that "
    "FP4 (4-bit) is used only for the quantized operands (Aq, Wq) of the core matrix "
    "multiply, while FP32 is used for accumulation and rescaling:",
    styles['Body'],
))
story.append(make_table(
    ['Tensor', 'Bits', 'Shape', 'Role'],
    [
        ['A',     '16', '[m, kb, ki]',  'FP16 input activations'],
        ['W',     '16', '[n, kb, ki]',  'FP16 input weights'],
        ['Sga',   '32', '[m]',          'Per-row activation scale (FP32)'],
        ['Sgw',   '32', '[n]',          'Per-column weight scale (FP32)'],
        ['Ascl',  '16', '[m, kb, ki]',  'Tensor-scaled activations'],
        ['Wscl',  '16', '[n, kb, ki]',  'Tensor-scaled weights'],
        ['Sba',   '8',  '[m, kb]',      'Block scale activations (FP8 E4M3)'],
        ['Sbw',   '8',  '[n, kb]',      'Block scale weights (FP8 E4M3)'],
        ['Aq',    '4',  '[m, kb, ki]',  'Quantized activations (FP4 E2M1)'],
        ['Wq',    '4',  '[n, kb, ki]',  'Quantized weights (FP4 E2M1)'],
        ['Yraw',  '32', '[m, n, kb]',   'FP32 partial sums'],
        ['Ytmp',  '32', '[m, n, kb]',   'After block-A rescale'],
        ['Yblk',  '32', '[m, n, kb]',   'After block-W rescale'],
        ['Ytmp2', '32', '[m, n]',       'After tensor-A rescale'],
        ['Y',     '16', '[m, n]',       'Final FP16 output'],
    ],
    col_widths=[0.6*inch, 0.5*inch, 1.3*inch, 3.3*inch],
))
story.append(Paragraph("Table 2: Tensor specifications. Total data footprint: 481.5 KB.", styles['Caption']))

story.append(Paragraph("2.3 Einsum Pipeline (13 Operations)", styles['SubHead']))
story.append(Paragraph(
    "The NVFP4 quantization pipeline is decomposed into 13 explicit einsum operations "
    "organized in four stages: activation quantization (4 ops), weight quantization "
    "(4 ops), core matrix multiply (1 op), and output rescaling (4 ops).",
    styles['Body'],
))

story.append(make_table(
    ['#', 'Einsum', 'Operation', 'Compute', 'Ops'],
    [
        ['1',  'TensorScaleA',  'Sga[m] = reduce(|A[m,kb,ki]|)',              'QuantMAC',   '8,192'],
        ['2',  'TensorQuantA',  'Ascl = A / Sga',                              'QuantMAC',   '8,192'],
        ['3',  'BlockScaleA',   'Sba[m,kb] = reduce(|Ascl|) over ki',          'QuantMAC',   '8,192'],
        ['4',  'BlockQuantA',   'Aq = Ascl / Sba',                             'QuantMAC',   '8,192'],
        ['5',  'TensorScaleW',  'Sgw[n] = reduce(|W[n,kb,ki]|)',              'QuantMAC',   '8,192'],
        ['6',  'TensorQuantW',  'Wscl = W / Sgw',                              'QuantMAC',   '8,192'],
        ['7',  'BlockScaleW',   'Sbw[n,kb] = reduce(|Wscl|) over ki',          'QuantMAC',   '8,192'],
        ['8',  'BlockQuantW',   'Wq = Wscl / Sbw',                             'QuantMAC',   '8,192'],
        ['9',  'MatMulNVFP4',   'Yraw[m,n,kb] += Aq * Wq (reduce ki)',        'FP4MAC',    '524,288'],
        ['10', 'RescaleBlockA', 'Ytmp = Yraw * Sba',                           'RescaleMAC', '32,768'],
        ['11', 'RescaleBlockW', 'Yblk = Ytmp * Sbw',                           'RescaleMAC', '32,768'],
        ['12', 'RescaleTensorA','Ytmp2[m,n] = sum(Yblk * Sga) over kb',       'RescaleMAC', '32,768'],
        ['13', 'RescaleTensorW','Y = Ytmp2 * Sgw',                             'RescaleMAC', '4,096'],
    ],
    col_widths=[0.3*inch, 1.2*inch, 2.6*inch, 0.9*inch, 0.7*inch],
))
story.append(Paragraph("Table 3: Einsum pipeline. Total: 692,224 ops. MatMulNVFP4 = 75.7%.", styles['Caption']))

story.append(PageBreak())

# ============================================================
# 3. Architecture Design
# ============================================================
story.append(Paragraph("3. Architecture Design", styles['SectionHead']))

story.append(Paragraph("3.1 Overview", styles['SubHead']))
story.append(Paragraph(
    "The accelerator uses a three-level memory hierarchy with a 2D spatial array and "
    "three specialized compute units. The architecture is parameterized using Jinja2 "
    "templates, allowing design-space exploration of buffer sizes, PE counts, and bandwidth.",
    styles['Body'],
))
story.append(Paragraph(
    "DRAM (off-chip, 64-bit bus)  -->  GLB (64 KB on-chip SRAM)  "
    "-->  RF (128 B/PE, 8x8 = 64 PEs)  -->  QuantMAC / FP4MAC / RescaleMAC",
    styles['Formula'],
))

story.append(Paragraph("3.2 Memory Hierarchy", styles['SubHead']))
story.append(Paragraph(
    "The three-level memory hierarchy is the most critical design decision. In the "
    "13-einsum pipeline, intermediate tensors (Sga, Ascl, Sba, Aq, Wscl, Sbw, Wq, "
    "Yraw, Ytmp, Yblk, Ytmp2) are produced by one einsum and consumed by the next. "
    "A sufficiently large GLB allows the auto-mapper to fuse consecutive einsums, "
    "keeping intermediates on-chip and avoiding costly DRAM round-trips. Analysis of "
    "the baseline design showed that 88% of total energy was spent on memory access, "
    "making this the dominant optimization target.",
    styles['Body'],
))

story.append(make_table(
    ['Level', 'Name', 'Size', 'Energy/Action', 'Bus Width', 'BW (act/cyc)', 'Area'],
    [
        ['L0', 'DRAM',  'Unlimited',    '10.0 pJ',  '64-bit', '8',  '0 (off-chip)'],
        ['L1', 'GLB',   '64 KB',        '5.0 pJ',   '32-bit', '32', '0.39 mm2'],
        ['L2', 'RF',    '128 B / PE',   '1.0 pJ',   '32-bit', '2',  '1024 um2 / PE'],
    ],
    col_widths=[0.4*inch, 0.55*inch, 0.8*inch, 0.85*inch, 0.65*inch, 0.8*inch, 1.1*inch],
))
story.append(Paragraph("Table 4: Memory hierarchy. RF is replicated 64x (8x8 spatial array).", styles['Caption']))

story.append(Paragraph(
    "<b>Tensor placement policy:</b> DRAM stores non-intermediate tensors (A, W, Y) via "
    "<font face='Courier'>keep: ~Intermediates</font>. GLB stores all intermediates via "
    "<font face='Courier'>keep: ~DRAM</font>. RF is flexible (<font face='Courier'>may_keep: All</font>) "
    "and the auto-mapper decides what to cache per-PE for maximum reuse.",
    styles['Body'],
))

story.append(Paragraph("3.3 Spatial Array (2D, 8 x 8 = 64 PEs)", styles['SubHead']))
story.append(Paragraph(
    "The register file level carries a 2D spatial attribute with fanout 8 in both X and Y "
    "dimensions, creating a grid of 64 processing elements. For the dominant MatMulNVFP4 "
    "einsum (Aq[m,kb,ki] x Wq[n,kb,ki] -> Yraw[m,n,kb]):",
    styles['Body'],
))
story.append(Paragraph(
    "- X dimension maps to m: each column of PEs processes a different activation row.<br/>"
    "- Y dimension maps to n: each row of PEs processes a different weight column.<br/>"
    "- Aq is multicast along Y (shared across n); Wq is multicast along X (shared across m).<br/>"
    "- This implements an <b>output-stationary dataflow</b>: each PE accumulates its own Yraw[m,n].",
    styles['Body'],
))
story.append(Paragraph(
    "For quantization einsums that lack the n dimension (e.g., TensorScaleA operates on "
    "[m, kb, ki] only), the Y-axis PEs are underutilized. However, since quantization "
    "accounts for only 9.5% of total compute, this inefficiency is negligible.",
    styles['Body'],
))

story.append(Paragraph("3.4 Specialized Compute Units", styles['SubHead']))
story.append(Paragraph(
    "Each PE contains three specialized compute units reflecting the heterogeneous "
    "precision requirements of the NVFP4 pipeline. Energy and area values are derived "
    "from Horowitz's 45 nm reference data, with multiplier area scaling as the square "
    "of the mantissa bit-width:",
    styles['Body'],
))

story.append(make_table(
    ['Unit', 'Handles', 'Energy', 'Latency', 'Area/PE', 'Horowitz Basis'],
    [
        ['QuantMAC',   '8 quant einsums\n(9.5% compute)',
         '1.0 pJ', '1 cycle', '90 um2',  '16b FP mult ~ 1.0 pJ'],
        ['FP4MAC',     'MatMulNVFP4\n(75.7% compute)',
         '0.2 pJ', '1 cycle', '60 um2',  '8b Int mult = 0.2 pJ'],
        ['RescaleMAC', '4 rescale einsums\n(14.8% compute)',
         '3.7 pJ', '2 cycles', '720 um2', '32b FP mult = 3.7 pJ'],
    ],
    col_widths=[0.85*inch, 1.2*inch, 0.65*inch, 0.65*inch, 0.7*inch, 1.6*inch],
))
story.append(Paragraph("Table 5: Compute unit specifications per PE.", styles['Caption']))

story.append(Paragraph(
    "<b>Key observation:</b> FP4MAC (0.2 pJ) is 18.5x cheaper than RescaleMAC (3.7 pJ) per "
    "operation. Although MatMulNVFP4 dominates in operation count (75.7%), the FP32 "
    "rescaling steps contribute disproportionately to compute energy due to their high "
    "per-operation cost. This motivates hardware optimization of the rescale datapath "
    "in future designs.",
    styles['Body'],
))

story.append(PageBreak())

# ============================================================
# 4. Energy Reference
# ============================================================
story.append(Paragraph("4. Energy and Area Reference Values", styles['SectionHead']))

story.append(Paragraph("4.1 Compute Energy (Horowitz, ISSCC 2014)", styles['SubHead']))
story.append(Paragraph(
    "All compute energy values are derived from Horowitz's widely-cited energy table for "
    "45 nm CMOS at 0.9 V [1]. The following reference values were used:",
    styles['Body'],
))
story.append(make_table(
    ['Operation', 'Energy (pJ)', 'Used For'],
    [
        ['8-bit Int Add',       '0.03',  '(reference)'],
        ['16-bit Int Add',      '0.05',  '(reference)'],
        ['32-bit Int Add',      '0.1',   '(reference)'],
        ['8-bit Int Multiply',  '0.2',   'FP4MAC (FP4 ~ 8b int complexity)'],
        ['32-bit Int Multiply', '3.1',   '(reference)'],
        ['32-bit FP Add',       '0.9',   '(reference)'],
        ['16-bit FP Multiply',  '~1.0',  'QuantMAC (interpolated)'],
        ['32-bit FP Multiply',  '3.7',   'RescaleMAC'],
    ],
    col_widths=[1.5*inch, 1.0*inch, 3.0*inch],
))
story.append(Paragraph("Table 6: Horowitz 2014 energy reference (45 nm, 0.9 V).", styles['Caption']))

story.append(Paragraph("4.2 Memory Energy", styles['SubHead']))
story.append(make_table(
    ['Memory', 'Horowitz Value', 'Our Model', 'Notes'],
    [
        ['8 KB SRAM (32b read)',   '5 pJ',       '(reference)',   'Closest to RF size'],
        ['32 KB SRAM (32b read)',  '10 pJ',      '(reference)',   'Between GLB sizes'],
        ['64 KB SRAM (32b read)',  '~10-13 pJ',  '5 pJ (GLB)',    'Scratchpad: no tag compare'],
        ['Register file (32b)',    '~0.5-1 pJ',  '1 pJ (RF)',     'Small, near-compute'],
        ['DRAM (64b access)',      '~1300 pJ',   '10 pJ',         'Amortized burst model'],
    ],
    col_widths=[1.5*inch, 1.1*inch, 1.0*inch, 2.1*inch],
))
story.append(Paragraph(
    "Table 7: Memory energy comparison. DRAM uses a simplified model consistent "
    "with the AccelForge lab framework, not the full Horowitz random-access cost.",
    styles['Caption'],
))

story.append(Paragraph("4.3 Area Estimation", styles['SubHead']))
story.append(Paragraph(
    "SRAM area is based on the TSMC 45 nm 6T cell size of 0.296 um<super>2</super>/bit [2], "
    "with a 2.5x overhead factor for peripheral circuits (decoders, sense amplifiers, I/O). "
    "Compute unit areas use lab3's die-photo estimate of 90 um<super>2</super> for a 16-bit MAC "
    "as the baseline, scaled by mantissa-width squared for other precisions [3].",
    styles['Body'],
))
story.append(make_table(
    ['Component', 'Formula', 'Per-Instance', 'Total (64 PEs)'],
    [
        ['GLB (64 KB)',    'bits x 0.296 um2 x 2.5',    '0.39 mm2',     '0.39 mm2'],
        ['RF (128 B/PE)',  'bits x 1.0 um2',             '1,024 um2',    '0.066 mm2'],
        ['QuantMAC',       'FP16 MAC baseline',          '90 um2',       '0.006 mm2'],
        ['FP4MAC',         'FP4 mult + FP32 accum',      '60 um2',       '0.004 mm2'],
        ['RescaleMAC',     '8x FP16 (mantissa2)',        '720 um2',      '0.046 mm2'],
        ['Total', '', '', '0.51 mm2'],
    ],
    col_widths=[1.2*inch, 1.6*inch, 1.1*inch, 1.2*inch],
))
story.append(Paragraph("Table 8: Area breakdown at 45 nm.", styles['Caption']))

story.append(PageBreak())

# ============================================================
# 5. Parameterization
# ============================================================
story.append(Paragraph("5. Design-Space Exploration Parameters", styles['SectionHead']))
story.append(Paragraph(
    "The architecture is implemented as a Jinja2 template (arch_template.yaml), "
    "allowing rapid exploration of the design space by adjusting parameters in the "
    "notebook without modifying the architecture file:",
    styles['Body'],
))
story.append(make_table(
    ['Parameter', 'Default', 'Unit', 'Affects'],
    [
        ['GLB_SIZE',  '524,288',  'bits',        'GLB capacity and area'],
        ['RF_SIZE',   '1,024',    'bits',        'Per-PE RF capacity and area'],
        ['PE_X',      '8',        'count',       'Spatial X fanout (array width)'],
        ['PE_Y',      '8',        'count',       'Spatial Y fanout (array height)'],
        ['DRAM_BW',   '8',        'actions/cyc',  'DRAM bandwidth (latency divisor)'],
        ['GLB_BW',    '32',       'actions/cyc',  'GLB bandwidth (latency divisor)'],
    ],
    col_widths=[1.0*inch, 0.8*inch, 0.9*inch, 3.0*inch],
))
story.append(Paragraph("Table 9: Parameterized architecture variables.", styles['Caption']))

story.append(Paragraph(
    "Area is automatically computed from parameters: GLB area = GLB_SIZE x 7.4e-13 m<super>2</super>, "
    "RF area per PE = RF_SIZE x 1e-12 m<super>2</super>. All parameters are passed to "
    "AccelForge via <font face='Courier'>jinja_parse_data</font> in "
    "<font face='Courier'>Spec.from_yaml()</font>.",
    styles['Body'],
))

# ============================================================
# 6. Compute Analysis
# ============================================================
story.append(Paragraph("6. Compute and Energy Analysis", styles['SectionHead']))

story.append(Paragraph("6.1 Operation Count Breakdown", styles['SubHead']))
story.append(make_table(
    ['Stage', 'Einsums', 'Ops', '% Total', 'Compute Unit', 'Energy/Op'],
    [
        ['Activation Quant',  '1-4',   '32,768',   '4.7%',   'QuantMAC',   '1.0 pJ'],
        ['Weight Quant',      '5-8',   '32,768',   '4.7%',   'QuantMAC',   '1.0 pJ'],
        ['MatMul (FP4xFP4)',  '9',     '524,288',  '75.7%',  'FP4MAC',     '0.2 pJ'],
        ['Rescale',           '10-13', '102,400',  '14.8%',  'RescaleMAC', '3.7 pJ'],
        ['Total',             '1-13',  '692,224',  '100%',   '',           ''],
    ],
    col_widths=[1.2*inch, 0.7*inch, 0.7*inch, 0.6*inch, 0.95*inch, 0.8*inch],
))
story.append(Paragraph("Table 10: Compute distribution by pipeline stage.", styles['Caption']))

story.append(Paragraph("6.2 Compute Energy by Stage", styles['SubHead']))
story.append(Paragraph(
    "Despite MatMulNVFP4 dominating operation count (75.7%), the per-operation energy "
    "differences create a more nuanced energy distribution:",
    styles['Body'],
))
story.append(make_table(
    ['Stage', 'Ops', 'Energy/Op (pJ)', 'Stage Energy (pJ)', '% Compute Energy'],
    [
        ['Quant (A+W)',      '65,536',   '1.0',   '65,536',    '13.6%'],
        ['MatMul (FP4xFP4)', '524,288',  '0.2',   '104,858',   '21.8%'],
        ['Rescale',          '102,400',  '3.7',   '378,880',   '78.8%'],
        ['Total',            '692,224',  '',       '549,274',   ''],
    ],
    col_widths=[1.2*inch, 0.8*inch, 1.1*inch, 1.2*inch, 1.2*inch],
))
story.append(Paragraph(
    "Table 11: Compute energy breakdown. Note: rescale (only 14.8% of ops) dominates "
    "compute energy at 78.8% due to expensive FP32 multiplications (latency: 2 cycles).",
    styles['Caption'],
))

story.append(PageBreak())

# ============================================================
# 7. Data Flow
# ============================================================
story.append(Paragraph("7. Data Flow and Fusion Strategy", styles['SectionHead']))
story.append(Paragraph(
    "The 13-einsum pipeline forms a directed acyclic graph (DAG) of tensor dependencies. "
    "The auto-mapper (AccelForge's FFM mapper) searches for optimal mappings that fuse "
    "consecutive einsums, keeping intermediate tensors in GLB:",
    styles['Body'],
))
story.append(Paragraph(
    "A[DRAM] -> TensorScaleA -> Sga[GLB] -> TensorQuantA -> Ascl[GLB] -> "
    "BlockScaleA -> Sba[GLB] -> BlockQuantA -> Aq[GLB]",
    styles['Formula'],
))
story.append(Paragraph(
    "W[DRAM] -> TensorScaleW -> Sgw[GLB] -> TensorQuantW -> Wscl[GLB] -> "
    "BlockScaleW -> Sbw[GLB] -> BlockQuantW -> Wq[GLB]",
    styles['Formula'],
))
story.append(Paragraph(
    "Aq[GLB] + Wq[GLB] -> MatMulNVFP4 -> Yraw[GLB]",
    styles['Formula'],
))
story.append(Paragraph(
    "Yraw -> RescaleBlockA -> Ytmp -> RescaleBlockW -> Yblk -> "
    "RescaleTensorA -> Ytmp2 -> RescaleTensorW -> Y[DRAM]",
    styles['Formula'],
))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Key insight:</b> Only A, W are read from DRAM and Y is written back. All 12 "
    "intermediate tensors remain in GLB through fusion. This eliminates the dominant "
    "energy cost identified in the baseline analysis (88% of energy was memory access).",
    styles['Body'],
))
story.append(Paragraph(
    "The auto-mapper uses AccelForge's <font face='Courier'>map_workload_to_arch()</font> "
    "with <font face='Courier'>Metrics.LATENCY | Metrics.ENERGY</font> to explore the "
    "mapping space. The best mapping is selected by minimum energy-delay product (EDP). "
    "The generated mapping is exported as YAML for reproducibility.",
    styles['Body'],
))

# ============================================================
# 8. Design Rationale Summary
# ============================================================
story.append(Paragraph("8. Design Rationale Summary", styles['SectionHead']))

story.append(make_table(
    ['Design Choice', 'Rationale'],
    [
        ['3-level memory\n(DRAM/GLB/RF)',
         'GLB enables fusion of 13-einsum pipeline;\nRF provides per-PE accumulator and tile storage'],
        ['64 KB GLB',
         'Holds all intermediates (~481 KB total, tiled)\nwithout DRAM spilling'],
        ['2D spatial (8x8)',
         'Output-stationary dataflow for MatMulNVFP4;\nboth Aq and Wq get spatial reuse'],
        ['3 compute units',
         'Accurate energy modeling for FP4/FP16/FP32;\nreflects real HW heterogeneity (e.g., Blackwell)'],
        ['64-bit DRAM bus',
         'Halves DRAM action count vs 32-bit;\nmatches FP32 accumulator width'],
        ['RescaleMAC latency=2',
         'FP32 multiply has ~2x critical path vs FP16;\nrealistic for non-pipelined 45 nm design'],
    ],
    col_widths=[1.6*inch, 4.1*inch],
))
story.append(Paragraph("Table 12: Summary of architecture design decisions.", styles['Caption']))

# ============================================================
# References
# ============================================================
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("References", styles['SectionHead']))
story.append(Paragraph(
    '[1] M. Horowitz, "1.1 Computing\'s Energy Problem (and What We Can Do About It)," '
    'IEEE International Solid-State Circuits Conference (ISSCC), 2014.',
    styles['Reference'],
))
story.append(Paragraph(
    '[2] TSMC, "45nm Node Planar-SOI Technology with 0.296 um2 6T-SRAM Cell," '
    'TSMC Research, 2008.',
    styles['Reference'],
))
story.append(Paragraph(
    '[3] Google, "BFloat16: The Secret to High Performance on Cloud TPUs," '
    'Google Cloud Blog, 2019. (multiplier area ~ mantissa_bits^2)',
    styles['Reference'],
))
story.append(Paragraph(
    '[4] NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference," '
    'NVIDIA Technical Blog, 2025.',
    styles['Reference'],
))
story.append(Paragraph(
    '[5] A. Parashar et al., "Timeloop: A Systematic Approach to DNN Accelerator Evaluation," '
    'IEEE ISPASS, 2019.',
    styles['Reference'],
))

# Build
doc.build(story)
print(f"Report generated: {OUTPUT}")
