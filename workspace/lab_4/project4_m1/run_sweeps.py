from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:
    np = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiment_defs import (
    ACCUMULATOR_BITS,
    ARCH_DEFAULTS,
    DEFAULT_MANIFEST_PATH,
    FIGURES_DIR,
    GENERATED_DIR,
    PHASES,
    QUANT_CONFIGS,
    RESULTS_DIR,
    SCALE_FORMATS,
    SPECIAL_CONFIGS,
    WORKLOADS,
    default_manifest,
    ensure_dir,
    get_quant_config,
    manifest_accuracy_input,
    manifest_or_default,
    manifest_runs,
    write_json_file,
)


PROPOSAL_HARDWARE_CSV = RESULTS_DIR / "proposal_hardware_summary.csv"
MILESTONE3_HARDWARE_CSV = RESULTS_DIR / "milestone3_hardware_summary.csv"
LEGACY_HARDWARE_CSV = RESULTS_DIR / "legacy_validation_summary.csv"
ACCURACY_CSV = RESULTS_DIR / "accuracy_summary.csv"


def json_as_yaml_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def format_float(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{value:.12g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def render_arch_yaml(run_spec: dict[str, Any]) -> str:
    config = get_quant_config(run_spec["config_id"])
    num_quantmac = int(run_spec["num_quantmac"])
    num_rescalemac = int(run_spec["num_rescalemac"])

    quant_area = ARCH_DEFAULTS["QUANTMAC_AREA"] * num_quantmac
    quant_energy = ARCH_DEFAULTS["QUANTMAC_ENERGY"]
    quant_latency = ARCH_DEFAULTS["QUANTMAC_LATENCY"]
    fp4_area = ARCH_DEFAULTS["FP4MAC_AREA"]
    fp4_energy = ARCH_DEFAULTS["FP4MAC_ENERGY"]
    fp4_latency = ARCH_DEFAULTS["FP4MAC_LATENCY"]
    fp16_area = ARCH_DEFAULTS["FP16MAC_AREA"]
    fp16_energy = ARCH_DEFAULTS["FP16MAC_ENERGY"]
    fp16_latency = ARCH_DEFAULTS["FP16MAC_LATENCY"]

    if isinstance(config, dict):
        fine_format = "fp32"
        coarse_format = "fp32"
    else:
        fine_format = config.fine_rescale_format or "fp32"
        coarse_format = config.coarse_rescale_format or fine_format

    fine_scale = SCALE_FORMATS[fine_format]
    coarse_scale = SCALE_FORMATS[coarse_format]
    fine_area = fine_scale["area_m2"] * num_rescalemac
    coarse_area = coarse_scale["area_m2"] * num_rescalemac

    enabled_quant = (
        "\"('Wq' in All and 'Sw' in All and 'Wdq' in All) "
        "or ('Sba' in All and ('A' in All or 'Ascl' in All)) "
        "or ('Sbw' in All and ('W' in All or 'Wscl' in All)) "
        "or ('Sga' in All and 'A' in All) "
        "or ('Sgw' in All and 'W' in All)\""
    )
    enabled_fp4 = "\"'Aq' in All and 'Wq' in All\""
    enabled_rescale_fine = "\"('Yraw' in All and 'Sba' in All) or ('Ytmp' in All and 'Sbw' in All)\""
    enabled_rescale_coarse = "\"('Yblk' in All and 'Sga' in All) or ('Ytmp2' in All and 'Sgw' in All)\""
    enabled_fp16 = "\"('A' in All and 'Y' in All and 'W' in All) or ('A' in All and 'Y' in All and 'Wdq' in All)\""

    return f"""arch:
  nodes:
  - !Memory
    name: DRAM
    size: 99999999999
    leak_power: 0
    area: 0
    total_latency: "ceil(max((read_actions + metadata_read_actions) / {ARCH_DEFAULTS['DRAM_BW']}, (write_actions + metadata_write_actions) / {ARCH_DEFAULTS['DRAM_BW']}))"
    tensors: {{keep: ~Intermediates, may_keep: All}}
    actions:
    - {{name: read, energy: 10.0, bits_per_action: 64, latency: 0}}
    - {{name: write, energy: 10.0, bits_per_action: 64, latency: 0}}
    - {{name: metadata_read, energy: 2.0, bits_per_action: 16, latency: 0}}
    - {{name: metadata_write, energy: 2.0, bits_per_action: 16, latency: 0}}

  - !Memory
    name: GLB
    size: {ARCH_DEFAULTS['GLB_SIZE']}
    leak_power: 0
    area: {format_float(ARCH_DEFAULTS['GLB_SIZE'] * 7.4e-13)}
    total_latency: "ceil(max(total_read_actions / {ARCH_DEFAULTS['GLB_BW']}, total_write_actions / {ARCH_DEFAULTS['GLB_BW']}))"
    tensors: {{keep: ~DRAM, may_keep: All}}
    actions:
    - {{name: read, energy: 5.0, bits_per_action: 32, latency: 0}}
    - {{name: write, energy: 5.0, bits_per_action: 32, latency: 0}}
    - {{name: metadata_read, energy: 2.0, bits_per_action: 16, latency: 0}}
    - {{name: metadata_write, energy: 2.0, bits_per_action: 16, latency: 0}}

  - !Memory
    name: RF
    size: {ARCH_DEFAULTS['RF_SIZE']}
    leak_power: 0
    area: {format_float(ARCH_DEFAULTS['RF_SIZE'] * 1e-12)}
    total_latency: "ceil(max(total_read_actions / 2, total_write_actions / 2))"
    tensors: {{may_keep: All}}
    spatial:
    - {{name: X, fanout: {ARCH_DEFAULTS['PE_X']}}}
    - {{name: Y, fanout: {ARCH_DEFAULTS['PE_Y']}}}
    actions:
    - {{name: read, energy: 1.0, bits_per_action: 32, latency: 0}}
    - {{name: write, energy: 1.0, bits_per_action: 32, latency: 0}}

  - !Compute
    name: QuantMAC
    enabled: {enabled_quant}
    leak_power: 0
    area: {format_float(quant_area)}
    total_latency: "ceil(compute_latency / {num_quantmac})"
    actions:
    - {{name: compute, energy: {format_float(quant_energy)}, latency: {quant_latency}}}

  - !Compute
    name: FP4MAC
    enabled: {enabled_fp4}
    leak_power: 0
    area: {format_float(fp4_area)}
    actions:
    - {{name: compute, energy: {format_float(fp4_energy)}, latency: {fp4_latency}}}

  - !Compute
    name: RescaleFineMAC
    enabled: {enabled_rescale_fine}
    leak_power: 0
    area: {format_float(fine_area)}
    total_latency: "ceil(compute_latency / {num_rescalemac})"
    actions:
    - {{name: compute, energy: {format_float(fine_scale['energy_pj'])}, latency: {int(fine_scale['latency_cycles'])}}}

  - !Compute
    name: RescaleCoarseMAC
    enabled: {enabled_rescale_coarse}
    leak_power: 0
    area: {format_float(coarse_area)}
    total_latency: "ceil(compute_latency / {num_rescalemac})"
    actions:
    - {{name: compute, energy: {format_float(coarse_scale['energy_pj'])}, latency: {int(coarse_scale['latency_cycles'])}}}

  - !Compute
    name: FP16MAC
    enabled: {enabled_fp16}
    leak_power: 0
    area: {format_float(fp16_area)}
    actions:
    - {{name: compute, energy: {format_float(fp16_energy)}, latency: {fp16_latency}}}
"""


def build_baseline_workload(shape: dict[str, int]) -> dict[str, Any]:
    return {
        "workload": {
            "iteration_space_shape": {
                "m": f"0 <= m < {shape['m']}",
                "n": f"0 <= n < {shape['n']}",
                "k": f"0 <= k < {shape['k']}",
            },
            "bits_per_value": {"A": 16, "W": 16, "Y": 16},
            "einsums": [
                {
                    "name": "MatMul",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "k"], "density": 1.0},
                        {"name": "W", "projection": ["n", "k"], "density": 1.0},
                        {"name": "Y", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        }
    }


def build_zero_level_workload(shape: dict[str, int], acc_bits: int) -> dict[str, Any]:
    return {
        "workload": {
            "iteration_space_shape": {
                "m": f"0 <= m < {shape['m']}",
                "n": f"0 <= n < {shape['n']}",
                "k": f"0 <= k < {shape['k']}",
            },
            "bits_per_value": {"Aq": 4, "Wq": 4, "Y": acc_bits},
            "einsums": [
                {
                    "name": "MatMulRaw4",
                    "tensor_accesses": [
                        {"name": "Aq", "projection": ["m", "k"], "density": 1.0},
                        {"name": "Wq", "projection": ["n", "k"], "density": 1.0},
                        {"name": "Y", "projection": ["m", "n"], "output": True},
                    ],
                }
            ],
        }
    }


def build_weight_only_workload(shape: dict[str, int], block_size: int, scale_bits: int, output_bits: int) -> dict[str, Any]:
    kb = shape["k"] // block_size
    return {
        "workload": {
            "iteration_space_shape": {
                "m": f"0 <= m < {shape['m']}",
                "n": f"0 <= n < {shape['n']}",
                "kb": f"0 <= kb < {kb}",
                "ki": f"0 <= ki < {block_size}",
            },
            "bits_per_value": {
                "A": 16,
                "Wq": 4,
                "Sw": scale_bits,
                "Wdq": 16,
                "Y": output_bits,
            },
            "einsums": [
                {
                    "name": "DequantW",
                    "tensor_accesses": [
                        {"name": "Wq", "projection": ["n", "kb", "ki"], "density": 1.0},
                        {"name": "Sw", "projection": ["n", "kb"], "density": 1.0},
                        {"name": "Wdq", "projection": ["n", "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "MatMulQ",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Wdq", "projection": ["n", "kb", "ki"], "density": 1.0},
                        {"name": "Y", "projection": ["m", "n"], "output": True},
                    ],
                },
            ],
        }
    }


def split_n_projection(shape: dict[str, int]) -> tuple[dict[str, str], list[str]]:
    if shape["n"] % 256 == 0:
        return (
            {
                "nbo": f"0 <= nbo < {shape['n'] // 256}",
                "nbi": "0 <= nbi < 16",
                "ni": "0 <= ni < 16",
            },
            ["nbo", "nbi", "ni"],
        )
    if shape["n"] % 16 == 0:
        return (
            {
                "nb": f"0 <= nb < {shape['n'] // 16}",
                "ni": "0 <= ni < 16",
            },
            ["nb", "ni"],
        )
    raise ValueError(f"Workload requires n to be divisible by 16, got n={shape['n']}")


def build_one_level_workload(shape: dict[str, int], block_size: int, scale_bits: int, acc_bits: int) -> dict[str, Any]:
    kb = shape["k"] // block_size
    n_iter_shape, n_projection = split_n_projection(shape)
    return {
        "workload": {
            "iteration_space_shape": {
                "m": f"0 <= m < {shape['m']}",
                **n_iter_shape,
                "kb": f"0 <= kb < {kb}",
                "ki": f"0 <= ki < {block_size}",
            },
            "bits_per_value": {
                "A": 16,
                "W": 16,
                "Sba": scale_bits,
                "Sbw": scale_bits,
                "Aq": 4,
                "Wq": 4,
                "Yraw": acc_bits,
                "Ytmp": acc_bits,
                "Y": 16,
            },
            "einsums": [
                {
                    "name": "BlockScaleA",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "output": True},
                    ],
                },
                {
                    "name": "BlockQuantA",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "density": 1.0},
                        {"name": "Aq", "projection": ["m", "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "BlockScaleW",
                    "tensor_accesses": [
                        {"name": "W", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "BlockQuantW",
                    "tensor_accesses": [
                        {"name": "W", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "density": 1.0},
                        {"name": "Wq", "projection": [*n_projection, "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "MatMulBlock",
                    "tensor_accesses": [
                        {"name": "Aq", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Wq", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Yraw", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockA",
                    "tensor_accesses": [
                        {"name": "Yraw", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "density": 1.0},
                        {"name": "Ytmp", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockW",
                    "tensor_accesses": [
                        {"name": "Ytmp", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "density": 1.0},
                        {"name": "Y", "projection": ["m", *n_projection], "output": True},
                    ],
                },
            ],
        }
    }


def build_two_level_workload(
    shape: dict[str, int],
    block_size: int,
    fine_scale_bits: int,
    coarse_scale_bits: int,
    acc_bits: int,
    coarse_granularity: str,
    *,
    split_n: bool = False,
    rank1_tensor_scales: bool = False,
) -> dict[str, Any]:
    kb = shape["k"] // block_size
    if coarse_granularity not in {"tensor", "row"}:
        raise ValueError(f"Unsupported coarse granularity: {coarse_granularity}")
    iteration_space_shape = {
        "m": f"0 <= m < {shape['m']}",
        "kb": f"0 <= kb < {kb}",
        "ki": f"0 <= ki < {block_size}",
    }
    if split_n:
        n_iter_shape, n_projection = split_n_projection(shape)
        iteration_space_shape.update(n_iter_shape)
    else:
        iteration_space_shape["n"] = f"0 <= n < {shape['n']}"
        n_projection = ["n"]

    if coarse_granularity == "tensor":
        if rank1_tensor_scales:
            iteration_space_shape["ga"] = "0 <= ga < 1"
            iteration_space_shape["gw"] = "0 <= gw < 1"
            activation_projection = ["ga"]
            weight_projection = ["gw"]
        else:
            activation_projection = []
            weight_projection = []
    else:
        activation_projection = ["m"]
        weight_projection = list(n_projection)
    return {
        "workload": {
            "iteration_space_shape": iteration_space_shape,
            "bits_per_value": {
                "A": 16,
                "W": 16,
                "Sga": coarse_scale_bits,
                "Sgw": coarse_scale_bits,
                "Ascl": 16,
                "Wscl": 16,
                "Sba": fine_scale_bits,
                "Sbw": fine_scale_bits,
                "Aq": 4,
                "Wq": 4,
                "Yraw": acc_bits,
                "Ytmp": acc_bits,
                "Yblk": acc_bits,
                "Ytmp2": acc_bits,
                "Y": 16,
            },
            "einsums": [
                {
                    "name": "TensorScaleA",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sga", "projection": activation_projection, "output": True},
                    ],
                },
                {
                    "name": "TensorQuantA",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sga", "projection": activation_projection, "density": 1.0},
                        {"name": "Ascl", "projection": ["m", "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "BlockScaleA",
                    "tensor_accesses": [
                        {"name": "Ascl", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "output": True},
                    ],
                },
                {
                    "name": "BlockQuantA",
                    "tensor_accesses": [
                        {"name": "Ascl", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "density": 1.0},
                        {"name": "Aq", "projection": ["m", "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "TensorScaleW",
                    "tensor_accesses": [
                        {"name": "W", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sgw", "projection": weight_projection, "output": True},
                    ],
                },
                {
                    "name": "TensorQuantW",
                    "tensor_accesses": [
                        {"name": "W", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sgw", "projection": weight_projection, "density": 1.0},
                        {"name": "Wscl", "projection": [*n_projection, "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "BlockScaleW",
                    "tensor_accesses": [
                        {"name": "Wscl", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "BlockQuantW",
                    "tensor_accesses": [
                        {"name": "Wscl", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "density": 1.0},
                        {"name": "Wq", "projection": [*n_projection, "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "MatMulNVFP4",
                    "tensor_accesses": [
                        {"name": "Aq", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Wq", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Yraw", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockA",
                    "tensor_accesses": [
                        {"name": "Yraw", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "density": 1.0},
                        {"name": "Ytmp", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockW",
                    "tensor_accesses": [
                        {"name": "Ytmp", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "density": 1.0},
                        {"name": "Yblk", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleTensorA",
                    "tensor_accesses": [
                        {"name": "Yblk", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sga", "projection": activation_projection, "density": 1.0},
                        {"name": "Ytmp2", "projection": ["m", *n_projection], "output": True},
                    ],
                },
                {
                    "name": "RescaleTensorW",
                    "tensor_accesses": [
                        {"name": "Ytmp2", "projection": ["m", *n_projection], "density": 1.0},
                        {"name": "Sgw", "projection": weight_projection, "density": 1.0},
                        {"name": "Y", "projection": ["m", *n_projection], "output": True},
                    ],
                },
            ],
        }
    }


def build_compact_two_level_workload(
    shape: dict[str, int],
    block_size: int,
    fine_scale_bits: int,
    coarse_scale_bits: int,
    acc_bits: int,
    coarse_granularity: str,
    *,
    split_n: bool = False,
    rank1_tensor_scales: bool = False,
) -> dict[str, Any]:
    """Mapper-safe two-level graph for large prefill cases.

    The full two-level graph materializes tensor-scale and block-scale stages
    separately. For the LLM prefill tensor-coarse cases this creates a huge
    AccelForge join space and Docker kills joblib workers. This compact graph
    keeps the same block matmul and two rescale levels, while treating coarse
    scales as supplied metadata instead of mapping their extraction pass.
    """

    kb = shape["k"] // block_size
    if coarse_granularity not in {"tensor", "row"}:
        raise ValueError(f"Unsupported coarse granularity: {coarse_granularity}")
    iteration_space_shape = {
        "m": f"0 <= m < {shape['m']}",
        "kb": f"0 <= kb < {kb}",
        "ki": f"0 <= ki < {block_size}",
    }
    if split_n:
        n_iter_shape, n_projection = split_n_projection(shape)
        iteration_space_shape.update(n_iter_shape)
    else:
        iteration_space_shape["n"] = f"0 <= n < {shape['n']}"
        n_projection = ["n"]

    if coarse_granularity == "tensor":
        if rank1_tensor_scales:
            iteration_space_shape["ga"] = "0 <= ga < 1"
            iteration_space_shape["gw"] = "0 <= gw < 1"
            activation_projection = ["ga"]
            weight_projection = ["gw"]
        else:
            activation_projection = []
            weight_projection = []
    else:
        activation_projection = ["m"]
        weight_projection = list(n_projection)

    return {
        "workload": {
            "iteration_space_shape": iteration_space_shape,
            "bits_per_value": {
                "A": 16,
                "W": 16,
                "Sga": coarse_scale_bits,
                "Sgw": coarse_scale_bits,
                "Sba": fine_scale_bits,
                "Sbw": fine_scale_bits,
                "Aq": 4,
                "Wq": 4,
                "Yraw": acc_bits,
                "Ytmp": acc_bits,
                "Yblk": acc_bits,
                "Ytmp2": acc_bits,
                "Y": 16,
            },
            "einsums": [
                {
                    "name": "BlockScaleA",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "output": True},
                    ],
                },
                {
                    "name": "BlockQuantA",
                    "tensor_accesses": [
                        {"name": "A", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "density": 1.0},
                        {"name": "Aq", "projection": ["m", "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "BlockScaleW",
                    "tensor_accesses": [
                        {"name": "W", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "BlockQuantW",
                    "tensor_accesses": [
                        {"name": "W", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "density": 1.0},
                        {"name": "Wq", "projection": [*n_projection, "kb", "ki"], "output": True},
                    ],
                },
                {
                    "name": "MatMulNVFP4",
                    "tensor_accesses": [
                        {"name": "Aq", "projection": ["m", "kb", "ki"], "density": 1.0},
                        {"name": "Wq", "projection": [*n_projection, "kb", "ki"], "density": 1.0},
                        {"name": "Yraw", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockA",
                    "tensor_accesses": [
                        {"name": "Yraw", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sba", "projection": ["m", "kb"], "density": 1.0},
                        {"name": "Ytmp", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleBlockW",
                    "tensor_accesses": [
                        {"name": "Ytmp", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sbw", "projection": [*n_projection, "kb"], "density": 1.0},
                        {"name": "Yblk", "projection": ["m", *n_projection, "kb"], "output": True},
                    ],
                },
                {
                    "name": "RescaleTensorA",
                    "tensor_accesses": [
                        {"name": "Yblk", "projection": ["m", *n_projection, "kb"], "density": 1.0},
                        {"name": "Sga", "projection": activation_projection, "density": 1.0},
                        {"name": "Ytmp2", "projection": ["m", *n_projection], "output": True},
                    ],
                },
                {
                    "name": "RescaleTensorW",
                    "tensor_accesses": [
                        {"name": "Ytmp2", "projection": ["m", *n_projection], "density": 1.0},
                        {"name": "Sgw", "projection": weight_projection, "density": 1.0},
                        {"name": "Y", "projection": ["m", *n_projection], "output": True},
                    ],
                },
            ],
        }
    }


def uses_compact_two_level_workload(run_spec: dict[str, Any]) -> bool:
    if str(os.environ.get("PROJECT4_FULL_TWO_LEVEL_PREFILL", "")).lower() in {"1", "true", "yes"}:
        return False
    config = get_quant_config(run_spec["config_id"])
    if isinstance(config, dict):
        return False
    return (
        run_spec.get("suite") == "proposal"
        and run_spec.get("workload_id") == "LLM"
        and run_spec.get("phase_id") == "prefill"
        and config.topology == "two_level"
        and (config.coarse_granularity or "tensor") == "tensor"
    )


def build_workload(run_spec: dict[str, Any]) -> dict[str, Any]:
    config = get_quant_config(run_spec["config_id"])
    shape = run_spec["shape"]
    if run_spec["config_id"] == "BASELINE_FP16":
        return build_baseline_workload(shape)
    if run_spec["config_id"] == "LEGACY_W4A16":
        return build_weight_only_workload(shape, block_size=16, scale_bits=16, output_bits=16)
    if run_spec["config_id"] == "LEGACY_NVFP4_FULL":
        return build_two_level_workload(
            shape,
            block_size=16,
            fine_scale_bits=8,
            coarse_scale_bits=32,
            acc_bits=32,
            coarse_granularity="row",
        )
    if isinstance(config, dict):
        raise ValueError(f"Unsupported special config: {run_spec['config_id']}")
    acc_bits = ACCUMULATOR_BITS[config.accumulator_format]
    if config.topology == "zero_level":
        return build_zero_level_workload(shape, acc_bits=acc_bits)
    if config.topology == "one_level":
        return build_one_level_workload(
            shape,
            block_size=config.block_size or 16,
            scale_bits=config.fine_scale_bits or SCALE_FORMATS[config.fine_rescale_format]["storage_bits"],
            acc_bits=acc_bits,
        )
    if config.topology == "two_level":
        if uses_compact_two_level_workload(run_spec):
            return build_compact_two_level_workload(
                shape,
                block_size=config.block_size or 16,
                fine_scale_bits=config.fine_scale_bits or SCALE_FORMATS[config.fine_rescale_format]["storage_bits"],
                coarse_scale_bits=config.coarse_scale_bits or SCALE_FORMATS[config.coarse_rescale_format]["storage_bits"],
                acc_bits=acc_bits,
                coarse_granularity=config.coarse_granularity or "tensor",
                split_n=True,
                rank1_tensor_scales=True,
            )
        return build_two_level_workload(
            shape,
            block_size=config.block_size or 16,
            fine_scale_bits=config.fine_scale_bits or SCALE_FORMATS[config.fine_rescale_format]["storage_bits"],
            coarse_scale_bits=config.coarse_scale_bits or SCALE_FORMATS[config.coarse_rescale_format]["storage_bits"],
            acc_bits=acc_bits,
            coarse_granularity=config.coarse_granularity or "tensor",
            split_n=True,
            rank1_tensor_scales=True,
        )
    raise ValueError(f"Unknown topology: {config.topology}")


def run_output_dir(run_spec: dict[str, Any]) -> Path:
    return ensure_dir(GENERATED_DIR / run_spec["suite"] / run_spec["run_id"])


def write_run_inputs(run_spec: dict[str, Any]) -> dict[str, Path]:
    out_dir = run_output_dir(run_spec)
    workload_path = out_dir / "workload.yaml"
    arch_path = out_dir / "arch.yaml"
    manifest_path = out_dir / "run_spec.json"

    write_json_file(manifest_path, run_spec)
    workload_path.write_text(json_as_yaml_text(build_workload(run_spec)))
    arch_path.write_text(render_arch_yaml(run_spec))
    return {
        "out_dir": out_dir,
        "workload_path": workload_path,
        "arch_path": arch_path,
        "manifest_path": manifest_path,
        "mapping_path": out_dir / "mapping.yaml",
        "breakdown_path": out_dir / "breakdown.json",
    }


def resolve_accuracy_snapshot_path(path_text: str, *, manifest_path: Path | None = None) -> Path:
    raw_path = Path(path_text).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
        candidates.append(HERE / "accuracy_inputs" / raw_path.name)
    else:
        if manifest_path is not None:
            candidates.append(manifest_path.parent / raw_path)
        candidates.append(HERE / raw_path)
        candidates.append(HERE / "accuracy_inputs" / raw_path.name)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return raw_path


def import_accelforge() -> tuple[Any, Any] | tuple[None, str]:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        af = importlib.import_module("accelforge")
        metrics_mod = importlib.import_module("accelforge.mapper")
        return af, metrics_mod.Metrics
    except Exception as exc:
        return None, str(exc)


def resolve_total_metric_column(columns: list[str], metric: str) -> str:
    metric = metric.lower()
    exact_candidates = {
        "energy": {"energy", "energy_total", "total_energy"},
        "latency": {"latency", "latency_total", "total_latency"},
    }[metric]
    exact = [col for col in columns if col.lower() in exact_candidates]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Ambiguous {metric} total columns: {exact}")

    fallback = [col for col in columns if metric in col.lower() and "<sep>" not in col.lower()]
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) > 1:
        raise ValueError(f"Ambiguous {metric} columns without explicit total: {fallback}")
    raise ValueError(f"Could not resolve total {metric} column from dataframe columns.")


def derive_totals_from_row(row: Any, einsum_names: list[str]) -> tuple[float, float]:
    energy_total = 0.0
    latency_per_einsum: dict[str, float] = {}
    columns = list(getattr(row, "index", row.keys()))
    for col in columns:
        parts = col.split("<SEP>")
        if len(parts) < 3 or parts[0] not in einsum_names:
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            continue
        metric = parts[1]
        if metric == "energy":
            energy_total += value
        elif metric == "latency":
            einsum = parts[0]
            latency_per_einsum[einsum] = max(latency_per_einsum.get(einsum, 0.0), value)
    latency_total = sum(latency_per_einsum.values())
    return energy_total, latency_total


def select_best_mapping_index(df: Any, einsum_names: list[str]) -> tuple[int, float, float, str, str]:
    energy_col = None
    latency_col = None
    energy_values = None
    latency_values = None
    try:
        energy_col = resolve_total_metric_column(list(df.columns), "energy")
        energy_values = [float(value) for value in df[energy_col].values]
    except ValueError:
        energy_col = "derived_from_breakdown"
    try:
        latency_col = resolve_total_metric_column(list(df.columns), "latency")
        latency_values = [float(value) for value in df[latency_col].values]
    except ValueError:
        latency_col = "derived_from_breakdown"

    if energy_values is None or latency_values is None:
        derived = [derive_totals_from_row(df.iloc[idx], einsum_names) for idx in range(len(df))]
        if energy_values is None:
            energy_values = [item[0] for item in derived]
        if latency_values is None:
            latency_values = [item[1] for item in derived]

    edp_values = [energy * latency for energy, latency in zip(energy_values, latency_values)]
    best_idx = int(min(range(len(edp_values)), key=lambda idx: edp_values[idx]))
    return best_idx, energy_values[best_idx], latency_values[best_idx], energy_col, latency_col


def extract_hardware_breakdown(df: Any, best_idx: int, einsum_names: list[str]) -> dict[str, Any]:
    row = df.iloc[best_idx]
    energy_per_einsum: dict[str, float] = {}
    latency_per_einsum: dict[str, float] = {}
    energy_compute_per_einsum: dict[str, float] = {}
    energy_memory_per_einsum: dict[str, float] = {}
    latency_compute_per_einsum: dict[str, float] = {}
    latency_memory_per_einsum: dict[str, float] = {}
    energy_per_component: dict[str, float] = {}
    latency_per_component: dict[str, float] = {}
    compute_components: set[str] = set()
    leak_entries: list[tuple[str, str, float]] = []

    for col in df.columns:
        parts = col.split("<SEP>")
        if len(parts) < 3 or parts[0] not in einsum_names or parts[1] != "energy":
            continue
        einsum = parts[0]
        component = parts[2]
        action = parts[-1]
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            continue
        energy_per_einsum[einsum] = energy_per_einsum.get(einsum, 0.0) + value
        energy_per_component[component] = energy_per_component.get(component, 0.0) + value
        if action == "compute":
            compute_components.add(component)
            energy_compute_per_einsum[einsum] = energy_compute_per_einsum.get(einsum, 0.0) + value
        elif action in ("read", "write", "metadata_read", "metadata_write"):
            energy_memory_per_einsum[einsum] = energy_memory_per_einsum.get(einsum, 0.0) + value
        elif action == "leak":
            leak_entries.append((einsum, component, value))

    for einsum, component, value in leak_entries:
        if component in compute_components:
            energy_compute_per_einsum[einsum] = energy_compute_per_einsum.get(einsum, 0.0) + value
        else:
            energy_memory_per_einsum[einsum] = energy_memory_per_einsum.get(einsum, 0.0) + value

    for col in df.columns:
        parts = col.split("<SEP>")
        if len(parts) < 3 or parts[0] not in einsum_names or parts[1] != "latency":
            continue
        einsum = parts[0]
        component = parts[2]
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            continue
        latency_per_einsum[einsum] = max(latency_per_einsum.get(einsum, 0.0), value)
        latency_per_component[component] = max(latency_per_component.get(component, 0.0), value)
        if component in compute_components:
            latency_compute_per_einsum[einsum] = max(latency_compute_per_einsum.get(einsum, 0.0), value)
        else:
            latency_memory_per_einsum[einsum] = max(latency_memory_per_einsum.get(einsum, 0.0), value)

    bottleneck_component = ""
    if latency_per_component:
        bottleneck_component = max(latency_per_component.items(), key=lambda item: item[1])[0]

    return {
        "einsum_names": einsum_names,
        "energy_per_einsum": energy_per_einsum,
        "latency_per_einsum": latency_per_einsum,
        "energy_compute_per_einsum": energy_compute_per_einsum,
        "energy_memory_per_einsum": energy_memory_per_einsum,
        "latency_compute_per_einsum": latency_compute_per_einsum,
        "latency_memory_per_einsum": latency_memory_per_einsum,
        "energy_per_component": energy_per_component,
        "latency_per_component": latency_per_component,
        "bottleneck_component": bottleneck_component,
    }


def assert_energy_total_matches_breakdown(
    *,
    energy_total: float,
    energy_per_einsum: dict[str, float],
    tolerance_abs: float = 1e-3,
    tolerance_rel: float = 1e-6,
) -> None:
    energy_sum = sum(energy_per_einsum.values())
    tolerance = max(tolerance_abs, abs(energy_total) * tolerance_rel)
    if abs(energy_total - energy_sum) > tolerance:
        raise ValueError(
            f"Energy total mismatch: total={energy_total}, per-einsum-sum={energy_sum}, tolerance={tolerance}"
        )


def run_hardware_case(run_spec: dict[str, Any]) -> dict[str, Any]:
    paths = write_run_inputs(run_spec)
    hardware_model = "compact_two_level_prefill" if uses_compact_two_level_workload(run_spec) else "full_accelforge_graph"
    row = {
        "suite": run_spec["suite"],
        "run_id": run_spec["run_id"],
        "status": "generated_only",
        "workload_id": run_spec["workload_id"],
        "phase_id": run_spec["phase_id"],
        "config_id": run_spec["config_id"],
        "arch_id": run_spec["arch_id"],
        "m": run_spec["shape"]["m"],
        "n": run_spec["shape"]["n"],
        "k": run_spec["shape"]["k"],
        "num_quantmac": run_spec["num_quantmac"],
        "num_rescalemac": run_spec["num_rescalemac"],
        "energy_pj": "",
        "energy_total_pj": "",
        "energy_pj_per_output": "",
        "latency_cycles": "",
        "area_m2": "",
        "bottleneck_component": "",
        "workload_file": str(paths["workload_path"]),
        "arch_file": str(paths["arch_path"]),
        "mapping_file": str(paths["mapping_path"]),
        "breakdown_file": str(paths["breakdown_path"]),
        "hardware_model": hardware_model,
        "error": "",
    }

    af_result = import_accelforge()
    if af_result[0] is None:
        row["error"] = f"accelforge unavailable: {af_result[1]}"
        return row

    af, Metrics = af_result
    try:
        spec = af.Spec.from_yaml(str(paths["arch_path"]), str(paths["workload_path"]))
        spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
        all_mappings = spec.map_workload_to_arch()
        df = all_mappings.data
        if len(df) == 0:
            row["status"] = "invalid"
            row["error"] = "No valid mappings were returned."
            return row

        best_idx, energy_total, latency_total, energy_col, latency_col = select_best_mapping_index(
            df,
            list(all_mappings.einsum_names),
        )
        result = all_mappings[best_idx]
        mapping_warning = ""
        try:
            paths["mapping_path"].write_text(result.mapping().to_yaml())
        except Exception as exc:
            mapping_warning = f"mapping serialization failed: {exc}"

        breakdown = extract_hardware_breakdown(df, best_idx, list(all_mappings.einsum_names))
        assert_energy_total_matches_breakdown(
            energy_total=energy_total,
            energy_per_einsum=breakdown["energy_per_einsum"],
        )
        breakdown["energy_total"] = energy_total
        breakdown["latency_total"] = latency_total
        breakdown["energy_total_column"] = energy_col
        breakdown["latency_total_column"] = latency_col
        breakdown["hardware_model"] = hardware_model
        output_elements = int(run_spec["shape"]["m"]) * int(run_spec["shape"]["n"])
        breakdown["output_elements"] = output_elements
        breakdown["energy_pj_per_output"] = energy_total / output_elements if output_elements else float("nan")

        area_total = None
        area_warning = ""
        for method_name in (
            "calculate_component_area_energy_latency_leak",
            "calculate_component_area_energy_latency_leak_power",
            "calculate_area_energy_latency_leak",
        ):
            method = getattr(spec, method_name, None)
            if callable(method):
                try:
                    method()
                    break
                except Exception as exc:
                    area_warning = f"{method_name} failed: {exc}"

        try:
            area_total = getattr(spec.arch, "total_area", None)
        except Exception as exc:
            if not area_warning:
                area_warning = f"total_area unavailable: {exc}"
            try:
                # Fallback: sum any per-node area values that are already materialized.
                nodes = getattr(spec.arch, "nodes", [])
                partial_area = 0.0
                found_any = False
                for node in nodes:
                    area_value = getattr(node, "area", None)
                    if isinstance(area_value, (int, float)):
                        partial_area += float(area_value)
                        found_any = True
                if found_any:
                    area_total = partial_area
            except Exception:
                pass

        breakdown["area_total"] = area_total
        write_json_file(paths["breakdown_path"], breakdown)

        row["status"] = "ok"
        row["energy_pj"] = energy_total
        row["energy_total_pj"] = energy_total
        row["energy_pj_per_output"] = breakdown["energy_pj_per_output"]
        row["latency_cycles"] = latency_total
        row["area_m2"] = breakdown["area_total"]
        row["bottleneck_component"] = breakdown["bottleneck_component"]
        warnings = [warning for warning in (mapping_warning, area_warning) if warning]
        if warnings:
            row["error"] = " | ".join(warnings)
        return row
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{exc}\n{traceback.format_exc()}"
        return row


def fp_levels_e2m1() -> list[float]:
    return [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def nearest_level(value: float, levels: list[float]) -> float:
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    best = min(levels, key=lambda level: abs(level - magnitude))
    return sign * best


def round_custom_float(value: float, mantissa_bits: int, exp_min: int, exp_max: int) -> float:
    if value == 0.0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    frac, exp = math.frexp(magnitude)
    exp = exp - 1
    exp = max(exp_min, min(exp_max, exp))
    normalized = magnitude / (2.0 ** exp)
    step = 2.0 ** (-mantissa_bits)
    rounded = round((normalized - 1.0) / step) * step + 1.0
    rounded = max(1.0, min(2.0 - step, rounded))
    return sign * rounded * (2.0 ** exp)


def quantize_scale_value(value: float, scale_format: str) -> float:
    if value == 0.0:
        return 0.0
    if scale_format == "fp32":
        return float(value)
    if scale_format == "fp16":
        return round_custom_float(value, mantissa_bits=10, exp_min=-14, exp_max=15)
    if scale_format == "e8m0":
        exponent = round(math.log(abs(value), 2.0))
        return float(2.0 ** exponent)
    raise ValueError(f"Unsupported scale format: {scale_format}")


def quantize_accumulator(value: float, accumulator_format: str) -> float:
    if accumulator_format == "fp32":
        return float(value)
    if accumulator_format == "fp16":
        return round_custom_float(value, mantissa_bits=10, exp_min=-14, exp_max=15)
    raise ValueError(f"Unsupported accumulator format: {accumulator_format}")


def blockify(values: list[float], block_size: int) -> list[list[float]]:
    return [values[idx : idx + block_size] for idx in range(0, len(values), block_size)]


def quantize_fp4_block(block: list[float], scale_format: str) -> tuple[list[float], float]:
    max_level = fp_levels_e2m1()[-1]
    max_abs = max((abs(value) for value in block), default=0.0)
    if max_abs == 0.0:
        return [0.0 for _ in block], 1.0
    scale = quantize_scale_value(max_abs / max_level, scale_format)
    if scale == 0.0:
        scale = 1.0
    quantized = [nearest_level(value / scale, fp_levels_e2m1()) for value in block]
    return quantized, scale


def matrix_from_payload(payload: Any, key: str) -> list[list[float]]:
    if np is not None and hasattr(payload, "shape"):
        array = np.asarray(payload, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array for {key}, got shape {array.shape}")
        return [[float(value) for value in row] for row in array.tolist()]
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected non-empty 2D list for {key}")
    rows: list[list[float]] = []
    for row in payload:
        if not isinstance(row, list) or not row:
            raise ValueError(f"Expected non-empty row list for {key}")
        rows.append([float(value) for value in row])
    return rows


def load_accuracy_snapshot(entry: dict[str, Any], *, manifest_path: Path | None = None) -> tuple[list[list[float]], list[list[float]], Path]:
    path = resolve_accuracy_snapshot_path(entry["path"], manifest_path=manifest_path)
    if not path.exists():
        raise FileNotFoundError(path)

    activation_key = entry.get("activation_key", "a")
    weight_key = entry.get("weight_key", "w")
    file_format = entry.get("format") or path.suffix.lstrip(".").lower()

    if file_format == "npz":
        if np is None:
            raise RuntimeError("numpy is required to load .npz accuracy snapshots")
        with np.load(path, allow_pickle=False) as payload:
            a_rows = matrix_from_payload(payload[activation_key], activation_key)
            w_rows = matrix_from_payload(payload[weight_key], weight_key)
        return a_rows, w_rows, path

    if file_format == "json":
        payload = json.loads(path.read_text())
        a_rows = matrix_from_payload(payload[activation_key], activation_key)
        w_rows = matrix_from_payload(payload[weight_key], weight_key)
        return a_rows, w_rows, path

    raise ValueError(f"Unsupported accuracy snapshot format: {file_format}")


def prepare_snapshot_views(
    *,
    a_rows: list[list[float]],
    w_rows: list[list[float]],
    phase_m: int,
    workload_n: int,
    workload_k: int,
    sample_m_max: int,
    sample_n_max: int,
) -> tuple[list[list[float]], list[list[float]], int, int, int]:
    if not a_rows or not w_rows:
        raise ValueError("Accuracy snapshot must contain non-empty activation and weight matrices.")
    if any(len(row) != workload_k for row in a_rows):
        raise ValueError(f"Activation snapshot K dimension must be {workload_k}.")
    if any(len(row) != workload_k for row in w_rows):
        raise ValueError(f"Weight snapshot K dimension must be {workload_k}.")

    m_eval = min(phase_m, sample_m_max, len(a_rows))
    n_eval = min(workload_n, sample_n_max, len(w_rows))
    if m_eval <= 0 or n_eval <= 0:
        raise ValueError("Accuracy snapshot does not contain enough rows for evaluation.")
    return a_rows[:m_eval], w_rows[:n_eval], m_eval, n_eval, workload_k


def sample_matrix_rows(rng: random.Random, rows: int, cols: int, distribution: str) -> list[list[float]]:
    data: list[list[float]] = []
    for _ in range(rows):
        row: list[float] = []
        for _ in range(cols):
            if distribution == "gaussian":
                value = rng.gauss(0.0, 0.9)
            elif distribution == "gaussian_narrow":
                value = rng.gauss(0.0, 0.5)
            elif distribution == "heavy_tail":
                base = rng.gauss(0.0, 0.7)
                tail = rng.gauss(0.0, 0.2)
                denom = max(0.25, abs(rng.gauss(1.0, 0.35)))
                value = (base + tail) / denom
            else:
                value = rng.gauss(0.0, 1.0)
            row.append(value)
        data.append(row)
    return data


def dot_product(lhs: list[float], rhs: list[float], accumulator_format: str) -> float:
    total = 0.0
    for left, right in zip(lhs, rhs):
        total = quantize_accumulator(total + left * right, accumulator_format)
    return total


def matmul_reference(a_rows: list[list[float]], w_rows: list[list[float]]) -> list[list[float]]:
    return [[sum(left * right for left, right in zip(a_row, w_row)) for w_row in w_rows] for a_row in a_rows]


def matmul_zero_level(a_rows: list[list[float]], w_rows: list[list[float]], accumulator_format: str) -> list[list[float]]:
    aq_rows = [[nearest_level(value, fp_levels_e2m1()) for value in row] for row in a_rows]
    wq_rows = [[nearest_level(value, fp_levels_e2m1()) for value in row] for row in w_rows]
    return [[dot_product(aq_row, wq_row, accumulator_format) for wq_row in wq_rows] for aq_row in aq_rows]


def matmul_one_level(
    a_rows: list[list[float]],
    w_rows: list[list[float]],
    block_size: int,
    scale_format: str,
    accumulator_format: str,
) -> list[list[float]]:
    a_blocks = [blockify(row, block_size) for row in a_rows]
    w_blocks = [blockify(row, block_size) for row in w_rows]
    a_quant = []
    w_quant = []
    for row_blocks in a_blocks:
        q_blocks = []
        scales = []
        for block in row_blocks:
            quantized, scale = quantize_fp4_block(block, scale_format)
            q_blocks.append(quantized)
            scales.append(scale)
        a_quant.append((q_blocks, scales))
    for row_blocks in w_blocks:
        q_blocks = []
        scales = []
        for block in row_blocks:
            quantized, scale = quantize_fp4_block(block, scale_format)
            q_blocks.append(quantized)
            scales.append(scale)
        w_quant.append((q_blocks, scales))

    output: list[list[float]] = []
    for a_q_blocks, a_scales in a_quant:
        out_row: list[float] = []
        for w_q_blocks, w_scales in w_quant:
            total = 0.0
            for block_index, (a_block, w_block) in enumerate(zip(a_q_blocks, w_q_blocks)):
                raw = dot_product(a_block, w_block, accumulator_format)
                total = quantize_accumulator(
                    total + raw * a_scales[block_index] * w_scales[block_index],
                    accumulator_format,
                )
            out_row.append(total)
        output.append(out_row)
    return output


def quantize_two_level_rows(
    rows: list[list[float]],
    *,
    block_size: int,
    fine_scale_format: str,
    coarse_scale_format: str,
    coarse_granularity: str,
) -> list[tuple[list[list[float]], float, list[float]]]:
    if coarse_granularity not in {"tensor", "row"}:
        raise ValueError(f"Unsupported coarse granularity: {coarse_granularity}")

    max_level = fp_levels_e2m1()[-1]
    tensor_scale = 1.0
    if coarse_granularity == "tensor":
        tensor_max = max((abs(value) for row in rows for value in row), default=0.0)
        if tensor_max > 0.0:
            tensor_scale = quantize_scale_value(tensor_max / max_level, coarse_scale_format)
            if tensor_scale == 0.0:
                tensor_scale = 1.0

    quantized_rows: list[tuple[list[list[float]], float, list[float]]] = []
    for row in rows:
        row_scale = tensor_scale
        if coarse_granularity == "row":
            row_max = max((abs(value) for value in row), default=0.0)
            if row_max > 0.0:
                row_scale = quantize_scale_value(row_max / max_level, coarse_scale_format)
                if row_scale == 0.0:
                    row_scale = 1.0
        scaled_row = [value / row_scale for value in row]
        blocks = blockify(scaled_row, block_size)
        q_blocks = []
        fine_scales = []
        for block in blocks:
            quantized, fine_scale = quantize_fp4_block(block, fine_scale_format)
            q_blocks.append(quantized)
            fine_scales.append(fine_scale)
        quantized_rows.append((q_blocks, row_scale, fine_scales))
    return quantized_rows


def matmul_two_level(
    a_rows: list[list[float]],
    w_rows: list[list[float]],
    block_size: int,
    fine_scale_format: str,
    coarse_scale_format: str,
    accumulator_format: str,
    coarse_granularity: str,
) -> list[list[float]]:
    if coarse_granularity not in {"tensor", "row"}:
        raise ValueError(f"Unsupported coarse granularity: {coarse_granularity}")

    a_tensor_scaled = quantize_two_level_rows(
        a_rows,
        block_size=block_size,
        fine_scale_format=fine_scale_format,
        coarse_scale_format=coarse_scale_format,
        coarse_granularity=coarse_granularity,
    )
    w_tensor_scaled = quantize_two_level_rows(
        w_rows,
        block_size=block_size,
        fine_scale_format=fine_scale_format,
        coarse_scale_format=coarse_scale_format,
        coarse_granularity=coarse_granularity,
    )

    output: list[list[float]] = []
    for a_q_blocks, a_coarse_scale, a_fine_scales in a_tensor_scaled:
        out_row: list[float] = []
        for w_q_blocks, w_coarse_scale, w_fine_scales in w_tensor_scaled:
            total = 0.0
            for block_index, (a_block, w_block) in enumerate(zip(a_q_blocks, w_q_blocks)):
                raw = dot_product(a_block, w_block, accumulator_format)
                contribution = raw
                contribution *= a_fine_scales[block_index]
                contribution *= w_fine_scales[block_index]
                contribution *= a_coarse_scale
                contribution *= w_coarse_scale
                total = quantize_accumulator(total + contribution, accumulator_format)
            out_row.append(total)
        output.append(out_row)
    return output


def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def evaluate_accuracy_case(
    run_spec: dict[str, Any],
    sample_m_max: int,
    sample_n_max: int,
    manifest: dict[str, Any],
    input_mode: str,
    manifest_path: Path | None,
) -> dict[str, Any]:
    config = get_quant_config(run_spec["config_id"])
    if run_spec["config_id"] in SPECIAL_CONFIGS:
        raise ValueError("Accuracy evaluation is only defined for proposal configs C0-C9.")
    assert run_spec["config_id"] in QUANT_CONFIGS
    workload = WORKLOADS[run_spec["workload_id"]]
    phase = PHASES[run_spec["phase_id"]]

    if input_mode == "proposal":
        entry = manifest_accuracy_input(manifest, run_spec["workload_id"])
        if entry is None:
            return {
                "suite": run_spec["suite"],
                "run_id": run_spec["run_id"],
                "status": "missing_inputs",
                "workload_id": run_spec["workload_id"],
                "phase_id": run_spec["phase_id"],
                "config_id": run_spec["config_id"],
                "m_eval": "",
                "n_eval": "",
                "k_eval": "",
                "cosine_similarity": "",
                "sqnr_db": "",
                "mse": "",
                "reference_norm": "",
                "input_source": "",
                "error": f"No accuracy input manifest entry for workload={run_spec['workload_id']}",
            }
        try:
            a_rows_all, w_rows_all, snapshot_path = load_accuracy_snapshot(entry, manifest_path=manifest_path)
        except FileNotFoundError as exc:
            return {
                "suite": run_spec["suite"],
                "run_id": run_spec["run_id"],
                "status": "missing_inputs",
                "workload_id": run_spec["workload_id"],
                "phase_id": run_spec["phase_id"],
                "config_id": run_spec["config_id"],
                "m_eval": "",
                "n_eval": "",
                "k_eval": "",
                "cosine_similarity": "",
                "sqnr_db": "",
                "mse": "",
                "reference_norm": "",
                "input_source": str(exc),
                "error": f"Missing accuracy snapshot: {exc}",
            }
        a_rows, w_rows, m_eval, n_eval, k_eval = prepare_snapshot_views(
            a_rows=a_rows_all,
            w_rows=w_rows_all,
            phase_m=phase.m,
            workload_n=workload.n,
            workload_k=workload.k,
            sample_m_max=sample_m_max,
            sample_n_max=sample_n_max,
        )
        input_source = str(snapshot_path)
    elif input_mode == "debug":
        m_eval = min(phase.m, sample_m_max)
        n_eval = min(workload.n, sample_n_max)
        k_eval = workload.k
        seed = sum(ord(ch) for ch in f"{run_spec['workload_id']}::{run_spec['phase_id']}::{run_spec['config_id']}")
        rng = random.Random(seed)
        a_rows = sample_matrix_rows(rng, m_eval, k_eval, workload.distribution)
        w_rows = sample_matrix_rows(rng, n_eval, k_eval, workload.distribution)
        input_source = f"debug::{workload.distribution}"
    else:
        raise ValueError(f"Unsupported accuracy input mode: {input_mode}")

    reference = matmul_reference(a_rows, w_rows)
    if config.topology == "zero_level":
        quantized = matmul_zero_level(a_rows, w_rows, accumulator_format=config.accumulator_format)
    elif config.topology == "one_level":
        quantized = matmul_one_level(
            a_rows,
            w_rows,
            block_size=config.block_size or 16,
            scale_format=config.fine_rescale_format or "fp32",
            accumulator_format=config.accumulator_format,
        )
    elif config.topology == "two_level":
        quantized = matmul_two_level(
            a_rows,
            w_rows,
            block_size=config.block_size or 16,
            fine_scale_format=config.fine_rescale_format or "fp32",
            coarse_scale_format=config.coarse_rescale_format or "fp32",
            accumulator_format=config.accumulator_format,
            coarse_granularity=config.coarse_granularity or "tensor",
        )
    else:
        raise ValueError(f"Unsupported topology for accuracy: {config.topology}")

    flat_ref = flatten_matrix(reference)
    flat_quant = flatten_matrix(quantized)
    dot = sum(left * right for left, right in zip(flat_ref, flat_quant))
    ref_norm = math.sqrt(sum(value * value for value in flat_ref))
    quant_norm = math.sqrt(sum(value * value for value in flat_quant))
    cosine = 0.0 if ref_norm == 0.0 or quant_norm == 0.0 else dot / (ref_norm * quant_norm)
    mse = sum((left - right) ** 2 for left, right in zip(flat_ref, flat_quant)) / max(1, len(flat_ref))
    signal_power = sum(value * value for value in flat_ref) / max(1, len(flat_ref))
    sqnr = float("inf") if mse == 0.0 else 10.0 * math.log10(signal_power / mse)
    return {
        "suite": run_spec["suite"],
        "run_id": run_spec["run_id"],
        "status": "ok",
        "workload_id": run_spec["workload_id"],
        "phase_id": run_spec["phase_id"],
        "config_id": run_spec["config_id"],
        "m_eval": m_eval,
        "n_eval": n_eval,
        "k_eval": k_eval,
        "cosine_similarity": cosine,
        "sqnr_db": sqnr,
        "mse": mse,
        "reference_norm": ref_norm,
        "input_source": input_source,
        "error": "",
    }


def append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(csv_path.parent)
    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
    fieldnames: list[str] = []
    for row in existing_rows + rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        return
    combined = {row["run_id"]: row for row in existing_rows}
    for row in rows:
        combined[row["run_id"]] = {key: str(value) for key, value in row.items()}
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_id in sorted(combined):
            writer.writerow(combined[run_id])


def load_existing_rows_by_run_id(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def filter_completed_runs(runs: list[dict[str, Any]], csv_path: Path, *, rerun_ok: bool) -> list[dict[str, Any]]:
    if rerun_ok:
        return runs
    existing = load_existing_rows_by_run_id(csv_path)
    return [run_spec for run_spec in runs if existing.get(run_spec["run_id"], {}).get("status") != "ok"]


def filter_runs_by_selectors(runs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    workload_ids = set(getattr(args, "workload", None) or [])
    phase_ids = set(getattr(args, "phase", None) or [])
    config_ids = set(getattr(args, "config", None) or [])
    arch_ids = set(getattr(args, "arch", None) or [])
    filtered = []
    for run_spec in runs:
        if workload_ids and run_spec["workload_id"] not in workload_ids:
            continue
        if phase_ids and run_spec["phase_id"] not in phase_ids:
            continue
        if config_ids and run_spec["config_id"] not in config_ids:
            continue
        if arch_ids and run_spec["arch_id"] not in arch_ids:
            continue
        filtered.append(run_spec)
    return filtered


def add_run_selectors(parser: argparse.ArgumentParser, *, include_arch: bool = False) -> None:
    parser.add_argument("--workload", action="append", help="Only run this workload_id. Can be passed multiple times.")
    parser.add_argument("--phase", action="append", help="Only run this phase_id. Can be passed multiple times.")
    parser.add_argument("--config", action="append", help="Only run this config_id. Can be passed multiple times.")
    if include_arch:
        parser.add_argument("--arch", action="append", help="Only run this arch_id. Can be passed multiple times.")


def command_status(args: argparse.Namespace) -> None:
    manifest = manifest_or_default(Path(args.manifest))
    suite = args.suite
    runs = filter_runs_by_selectors(manifest_runs(manifest, suite), args)
    csv_path = target_hardware_csv(suite)
    existing = load_existing_rows_by_run_id(csv_path)
    counts: dict[str, int] = {}
    missing: list[str] = []
    non_ok: list[tuple[str, str]] = []
    for run_spec in runs:
        row = existing.get(run_spec["run_id"])
        status = row.get("status", "missing") if row else "missing"
        counts[status] = counts.get(status, 0) + 1
        if status == "missing":
            missing.append(run_spec["run_id"])
        elif status != "ok":
            non_ok.append((run_spec["run_id"], status))
    print(f"Status for suite={suite} in {csv_path}")
    print(f"selected={len(runs)} counts={counts}")
    if missing:
        print("Missing runs:")
        for run_id in missing[: args.max_rows]:
            print(f"  {run_id}")
        if len(missing) > args.max_rows:
            print(f"  ... {len(missing) - args.max_rows} more")
    if non_ok:
        print("Non-ok runs:")
        for run_id, status in non_ok[: args.max_rows]:
            print(f"  {run_id}: {status}")
        if len(non_ok) > args.max_rows:
            print(f"  ... {len(non_ok) - args.max_rows} more")


def target_hardware_csv(suite: str) -> Path:
    if suite == "proposal":
        return PROPOSAL_HARDWARE_CSV
    if suite == "milestone3":
        return MILESTONE3_HARDWARE_CSV
    if suite == "legacy_validation":
        return LEGACY_HARDWARE_CSV
    raise KeyError(f"Unknown suite: {suite}")


def run_hardware_case_with_timing(run_spec: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    row = run_hardware_case(run_spec)
    row["duration_s"] = round(time.time() - started_at, 3)
    return row


def evaluate_accuracy_case_with_timing(
    run_spec: dict[str, Any],
    sample_m_max: int,
    sample_n_max: int,
    manifest: dict[str, Any],
    input_mode: str,
    manifest_path: Path | None,
) -> dict[str, Any]:
    started_at = time.time()
    row = evaluate_accuracy_case(
        run_spec,
        sample_m_max=sample_m_max,
        sample_n_max=sample_n_max,
        manifest=manifest,
        input_mode=input_mode,
        manifest_path=manifest_path,
    )
    row["duration_s"] = round(time.time() - started_at, 3)
    return row


def print_case_start(kind: str, index: int, total: int, run_spec: dict[str, Any]) -> None:
    print(
        f"[{kind} {index}/{total}] start {run_spec['run_id']} "
        f"(workload={run_spec['workload_id']}, phase={run_spec['phase_id']}, "
        f"config={run_spec['config_id']}, arch={run_spec['arch_id']})"
    )


def print_case_done(kind: str, completed: int, total: int, row: dict[str, Any]) -> None:
    duration = row.get("duration_s", "")
    duration_text = f"{duration}s" if duration not in ("", None) else "n/a"
    status = row.get("status", "unknown")
    extra = ""
    if status == "ok":
        extra = (
            f", energy={row.get('energy_pj', '')}, latency={row.get('latency_cycles', '')}, "
            f"bottleneck={row.get('bottleneck_component', '')}"
        )
    elif status == "generated_only":
        extra = ", generated inputs only"
    elif status == "error":
        extra = ", error recorded"
    print(f"[{kind} {completed}/{total}] done {row['run_id']} status={status}, duration={duration_text}{extra}")


def process_runs_with_progress(
    *,
    kind: str,
    runs: list[dict[str, Any]],
    csv_path: Path,
    worker_fn: Any,
    jobs: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(runs)
    if total == 0:
        return rows

    jobs = max(1, int(jobs))
    if jobs == 1:
        for index, run_spec in enumerate(runs, start=1):
            print_case_start(kind, index, total, run_spec)
            row = worker_fn(run_spec)
            append_rows(csv_path, [row])
            rows.append(row)
            print_case_done(kind, index, total, row)
        return rows

    for index, run_spec in enumerate(runs, start=1):
        print_case_start(kind, index, total, run_spec)

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_index = {
            executor.submit(worker_fn, run_spec): index
            for index, run_spec in enumerate(runs, start=1)
        }
        completed = 0
        for future in as_completed(future_to_index):
            row = future.result()
            append_rows(csv_path, [row])
            rows.append(row)
            completed += 1
            print_case_done(kind, completed, total, row)
    return rows


def command_write_manifest(args: argparse.Namespace) -> None:
    manifest = default_manifest()
    write_json_file(Path(args.output), manifest)
    print(f"Wrote manifest with {len(manifest['proposal_runs'])} proposal runs and {len(manifest['milestone3_runs'])} milestone3 runs to {args.output}")


def command_run_hardware(args: argparse.Namespace) -> None:
    manifest = manifest_or_default(Path(args.manifest))
    suite = args.suite
    runs = filter_runs_by_selectors(manifest_runs(manifest, suite), args)
    if args.limit:
        runs = runs[: args.limit]
    csv_path = target_hardware_csv(suite)
    requested_count = len(runs)
    runs = filter_completed_runs(runs, csv_path, rerun_ok=args.rerun_ok)
    skipped_count = requested_count - len(runs)
    if skipped_count:
        print(f"Skipping {skipped_count} existing ok hardware rows. Use --rerun-ok to recompute them.")
    rows = process_runs_with_progress(
        kind=f"hardware:{suite}",
        runs=runs,
        csv_path=csv_path,
        worker_fn=run_hardware_case_with_timing,
        jobs=args.jobs,
    )
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"Processed {len(rows)} hardware runs for suite={suite}. OK={ok_count}. Summary: {csv_path}")


def command_run_accuracy(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = manifest_or_default(manifest_path)
    runs = filter_runs_by_selectors(manifest_runs(manifest, args.suite), args)
    unique_runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run_spec in runs:
        key = (run_spec["workload_id"], run_spec["phase_id"], run_spec["config_id"])
        unique_runs.setdefault(key, run_spec)
    ordered_runs = [unique_runs[key] for key in sorted(unique_runs)]
    if args.limit:
        ordered_runs = ordered_runs[: args.limit]
    def worker(run_spec: dict[str, Any]) -> dict[str, Any]:
        try:
            return evaluate_accuracy_case_with_timing(
                run_spec,
                sample_m_max=args.sample_m_max,
                sample_n_max=args.sample_n_max,
                manifest=manifest,
                input_mode=args.input_mode,
                manifest_path=manifest_path,
            )
        except Exception as exc:
            return {
                "suite": run_spec["suite"],
                "run_id": run_spec["run_id"],
                "status": "error",
                "workload_id": run_spec["workload_id"],
                "phase_id": run_spec["phase_id"],
                "config_id": run_spec["config_id"],
                "m_eval": "",
                "n_eval": "",
                "k_eval": "",
                "cosine_similarity": "",
                "sqnr_db": "",
                "mse": "",
                "reference_norm": "",
                "input_source": "",
                "duration_s": "",
                "error": f"{exc}\n{traceback.format_exc()}",
            }

    rows = process_runs_with_progress(
        kind="accuracy",
        runs=ordered_runs,
        csv_path=ACCURACY_CSV,
        worker_fn=worker,
        jobs=1,
    )
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"Processed {len(rows)} accuracy runs for suite={args.suite}. OK={ok_count}. Summary: {ACCURACY_CSV}")


def command_verify_legacy(args: argparse.Namespace) -> None:
    manifest = manifest_or_default(Path(args.manifest))
    runs = manifest_runs(manifest, "legacy_validation")
    rows = process_runs_with_progress(
        kind="hardware:legacy_validation",
        runs=runs,
        csv_path=LEGACY_HARDWARE_CSV,
        worker_fn=run_hardware_case_with_timing,
        jobs=1,
    )

    reference_files = {
        "BASELINE_FP16": HERE / "mapping_baseline.breakdown.json",
        "LEGACY_W4A16": HERE / "mapping_nvfp4_weight_only.breakdown.json",
        "LEGACY_NVFP4_FULL": HERE / "mapping_nvfp4_full_auto.breakdown.json",
    }
    for row in rows:
        print(f"[{row['config_id']}] status={row['status']}")
        if row["status"] != "ok":
            print(f"  skipped comparison: {row['error']}")
            continue
        reference_path = reference_files[row["config_id"]]
        generated_path = Path(row["breakdown_file"])
        if not reference_path.exists() or not generated_path.exists():
            print("  missing comparison data")
            continue
        reference = json.loads(reference_path.read_text())
        generated = json.loads(generated_path.read_text())
        energy_delta = generated["energy_total"] - reference["energy_total"]
        latency_delta = generated["latency_total"] - reference["latency_total"]
        print(f"  energy delta={energy_delta:.6g} pJ")
        print(f"  latency delta={latency_delta:.6g} cycles")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run project 4 quantization experiments without relying on the notebook.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_manifest = subparsers.add_parser("write-manifest", help="Write the checked-in default experiment manifest.")
    write_manifest.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    write_manifest.set_defaults(func=command_write_manifest)

    status = subparsers.add_parser("status", help="Show missing/non-ok rows for a hardware summary without running new cases.")
    status.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    status.add_argument("--suite", choices=["proposal", "milestone3", "legacy_validation"], default="proposal")
    add_run_selectors(status, include_arch=True)
    status.add_argument("--max-rows", type=int, default=80)
    status.set_defaults(func=command_status)

    run_hardware = subparsers.add_parser("run-hardware", help="Generate workload/arch files and run the hardware sweep if AccelForge is available.")
    run_hardware.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    run_hardware.add_argument("--suite", choices=["proposal", "milestone3", "legacy_validation"], default="proposal")
    add_run_selectors(run_hardware, include_arch=True)
    run_hardware.add_argument("--limit", type=int)
    run_hardware.add_argument("--jobs", type=int, default=1, help="Number of parallel workers to use for independent cases.")
    run_hardware.add_argument("--rerun-ok", action="store_true", help="Recompute runs that already have ok rows in the target CSV.")
    run_hardware.set_defaults(func=command_run_hardware)

    run_accuracy = subparsers.add_parser("run-accuracy", help="Run the pure-Python accuracy emulator for proposal configs.")
    run_accuracy.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    run_accuracy.add_argument("--suite", choices=["proposal"], default="proposal", help="Accuracy is currently defined for proposal configs.")
    add_run_selectors(run_accuracy)
    run_accuracy.add_argument("--limit", type=int)
    run_accuracy.add_argument("--sample-m-max", type=int, default=4)
    run_accuracy.add_argument("--sample-n-max", type=int, default=16)
    run_accuracy.add_argument("--input-mode", choices=["proposal", "debug"], default="proposal")
    run_accuracy.set_defaults(func=command_run_accuracy)

    verify_legacy = subparsers.add_parser("verify-legacy", help="Re-run the legacy 4096x4096 baselines and compare against saved notebook outputs.")
    verify_legacy.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    verify_legacy.set_defaults(func=command_verify_legacy)

    return parser


def main() -> None:
    ensure_dir(RESULTS_DIR)
    ensure_dir(FIGURES_DIR)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
