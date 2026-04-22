from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GENERATED_DIR = ROOT / "generated"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DEFAULT_MANIFEST_PATH = ROOT / "experiment_manifest.json"

DEFAULT_ACCURACY_FLOOR = 0.98
DEFAULT_SATURATION_TOLERANCE = 0.05


@dataclass(frozen=True)
class WorkloadDef:
    workload_id: str
    label: str
    n: int
    k: int
    distribution: str
    description: str


@dataclass(frozen=True)
class PhaseDef:
    phase_id: str
    label: str
    m: int


@dataclass(frozen=True)
class QuantConfig:
    config_id: str
    topology: str
    description: str
    block_size: int | None
    fine_rescale_format: str | None
    coarse_rescale_format: str | None
    fine_scale_bits: int | None
    coarse_scale_bits: int | None
    accumulator_format: str
    reference_scheme: str | None = None


@dataclass(frozen=True)
class ArchVariant:
    arch_id: str
    num_quantmac: int
    num_rescalemac: int
    label: str


WORKLOADS: dict[str, WorkloadDef] = {
    "LLM": WorkloadDef(
        workload_id="LLM",
        label="LLM FFN",
        n=11008,
        k=4096,
        distribution="gaussian",
        description="Llama-style FFN GEMM",
    ),
    "VLM": WorkloadDef(
        workload_id="VLM",
        label="VLM Vision",
        n=3072,
        k=3072,
        distribution="gaussian_narrow",
        description="Balanced vision-style GEMM",
    ),
    "VLA": WorkloadDef(
        workload_id="VLA",
        label="VLA Action Head",
        n=256,
        k=4096,
        distribution="heavy_tail",
        description="Latency-sensitive action-head GEMM",
    ),
}


PHASES: dict[str, PhaseDef] = {
    "decode": PhaseDef(phase_id="decode", label="Decode", m=1),
    "prefill": PhaseDef(phase_id="prefill", label="Prefill", m=128),
}


SCALE_FORMATS: dict[str, dict[str, float | int | str]] = {
    "fp32": {
        "label": "FP32",
        "storage_bits": 32,
        "energy_pj": 3.7,
        "latency_cycles": 2,
        "area_m2": 7.2e-10,
    },
    "fp16": {
        "label": "FP16",
        "storage_bits": 16,
        "energy_pj": 1.0,
        "latency_cycles": 1,
        "area_m2": 1.8e-10,
    },
    "e8m0": {
        "label": "E8M0",
        "storage_bits": 8,
        "energy_pj": 0.03,
        "latency_cycles": 1,
        "area_m2": 1.0e-11,
    },
}


ACCUMULATOR_BITS = {
    "fp32": 32,
    "fp16": 16,
}


QUANT_CONFIGS: dict[str, QuantConfig] = {
    "C0": QuantConfig(
        config_id="C0",
        topology="zero_level",
        description="Raw FP4 lower bound without scale factors",
        block_size=None,
        fine_rescale_format=None,
        coarse_rescale_format=None,
        fine_scale_bits=None,
        coarse_scale_bits=None,
        accumulator_format="fp32",
        reference_scheme="ideal-w4a4",
    ),
    "C1": QuantConfig(
        config_id="C1",
        topology="one_level",
        description="MXFP4-like one-level b=32 with E8M0 rescale",
        block_size=32,
        fine_rescale_format="e8m0",
        coarse_rescale_format=None,
        fine_scale_bits=8,
        coarse_scale_bits=None,
        accumulator_format="fp32",
        reference_scheme="mxfp4-like",
    ),
    "C2": QuantConfig(
        config_id="C2",
        topology="one_level",
        description="Cheap fine-grained one-level b=16 with E8M0 rescale",
        block_size=16,
        fine_rescale_format="e8m0",
        coarse_rescale_format=None,
        fine_scale_bits=8,
        coarse_scale_bits=None,
        accumulator_format="fp32",
    ),
    "C3": QuantConfig(
        config_id="C3",
        topology="one_level",
        description="One-level b=16 with FP16 rescale",
        block_size=16,
        fine_rescale_format="fp16",
        coarse_rescale_format=None,
        fine_scale_bits=16,
        coarse_scale_bits=None,
        accumulator_format="fp32",
    ),
    "C4": QuantConfig(
        config_id="C4",
        topology="one_level",
        description="One-level b=16 with FP32 rescale",
        block_size=16,
        fine_rescale_format="fp32",
        coarse_rescale_format=None,
        fine_scale_bits=32,
        coarse_scale_bits=None,
        accumulator_format="fp32",
    ),
    "C5": QuantConfig(
        config_id="C5",
        topology="two_level",
        description="Hybrid two-level with E8M0 fine and FP16 coarse rescale",
        block_size=16,
        fine_rescale_format="e8m0",
        coarse_rescale_format="fp16",
        fine_scale_bits=8,
        coarse_scale_bits=16,
        accumulator_format="fp32",
    ),
    "C6": QuantConfig(
        config_id="C6",
        topology="two_level",
        description="Two-level with FP16 fine and FP16 coarse rescale",
        block_size=16,
        fine_rescale_format="fp16",
        coarse_rescale_format="fp16",
        fine_scale_bits=16,
        coarse_scale_bits=16,
        accumulator_format="fp32",
    ),
    "C7": QuantConfig(
        config_id="C7",
        topology="two_level",
        description="NVFP4-like two-level b=16 coarse+tower pipeline with FP32 rescale",
        block_size=16,
        fine_rescale_format="fp32",
        coarse_rescale_format="fp32",
        fine_scale_bits=8,
        coarse_scale_bits=32,
        accumulator_format="fp32",
        reference_scheme="nvfp4-like",
    ),
    "C8": QuantConfig(
        config_id="C8",
        topology="one_level",
        description="Aggressive one-level b=16 with FP16 rescale and FP16 accumulation",
        block_size=16,
        fine_rescale_format="fp16",
        coarse_rescale_format=None,
        fine_scale_bits=16,
        coarse_scale_bits=None,
        accumulator_format="fp16",
    ),
    "C9": QuantConfig(
        config_id="C9",
        topology="two_level",
        description="Aggressive two-level with E8M0 fine, FP16 coarse, FP16 accumulation",
        block_size=16,
        fine_rescale_format="e8m0",
        coarse_rescale_format="fp16",
        fine_scale_bits=8,
        coarse_scale_bits=16,
        accumulator_format="fp16",
    ),
}


SPECIAL_CONFIGS: dict[str, dict[str, Any]] = {
    "BASELINE_FP16": {
        "config_id": "BASELINE_FP16",
        "topology": "baseline_fp16",
        "description": "Full precision baseline MatMul",
    },
    "LEGACY_W4A16": {
        "config_id": "LEGACY_W4A16",
        "topology": "legacy_weight_only",
        "description": "Notebook-compatible NVFP4 weight-only baseline",
        "block_size": 16,
        "scale_bits": 16,
        "dequant_format": "fp16",
    },
    "LEGACY_NVFP4_FULL": {
        "config_id": "LEGACY_NVFP4_FULL",
        "topology": "legacy_nvfp4_full",
        "description": "Notebook-compatible full NVFP4 W4A4 model",
    },
}


ARCH_DEFAULTS: dict[str, float | int] = {
    "GLB_SIZE": 524288,
    "RF_SIZE": 1024,
    "PE_X": 8,
    "PE_Y": 8,
    "DRAM_BW": 8,
    "GLB_BW": 32,
    "QUANTMAC_ENERGY": 1.0,
    "QUANTMAC_LATENCY": 1,
    "QUANTMAC_AREA": 9e-11,
    "FP4MAC_ENERGY": 0.2,
    "FP4MAC_LATENCY": 1,
    "FP4MAC_AREA": 6e-11,
    "FP16MAC_ENERGY": 1.0,
    "FP16MAC_LATENCY": 1,
    "FP16MAC_AREA": 9e-11,
}


PROPOSAL_CONFIG_IDS = list(QUANT_CONFIGS.keys())
MILESTONE3_CONFIG_IDS = ["C1", "C5", "C7"]
UNIT_COUNT_OPTIONS = [1, 2, 4, 8]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_file(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def phase_shape(workload_id: str, phase_id: str) -> dict[str, int]:
    workload = WORKLOADS[workload_id]
    phase = PHASES[phase_id]
    return {"m": phase.m, "n": workload.n, "k": workload.k}


def default_arch_variant() -> ArchVariant:
    return ArchVariant(
        arch_id="baseline",
        num_quantmac=1,
        num_rescalemac=1,
        label="Baseline architecture",
    )


def milestone3_variants() -> list[ArchVariant]:
    variants: list[ArchVariant] = []
    for num_quantmac, num_rescalemac in product(UNIT_COUNT_OPTIONS, repeat=2):
        arch_id = f"q{num_quantmac}_r{num_rescalemac}"
        variants.append(
            ArchVariant(
                arch_id=arch_id,
                num_quantmac=num_quantmac,
                num_rescalemac=num_rescalemac,
                label=f"QuantMAC={num_quantmac}, RescaleMAC={num_rescalemac}",
            )
        )
    return variants


def run_id(
    suite: str,
    workload_id: str,
    phase_id: str,
    config_id: str,
    arch_id: str,
) -> str:
    return f"{suite.lower()}__{workload_id.lower()}__{phase_id.lower()}__{config_id.lower()}__{arch_id.lower()}"


def default_manifest() -> dict[str, Any]:
    proposal_runs: list[dict[str, Any]] = []
    for workload_id, phase_id, config_id in product(WORKLOADS, PHASES, PROPOSAL_CONFIG_IDS):
        arch = default_arch_variant()
        proposal_runs.append(
            {
                "suite": "proposal",
                "run_id": run_id("proposal", workload_id, phase_id, config_id, arch.arch_id),
                "workload_id": workload_id,
                "phase_id": phase_id,
                "config_id": config_id,
                "arch_id": arch.arch_id,
                "num_quantmac": arch.num_quantmac,
                "num_rescalemac": arch.num_rescalemac,
                "shape": phase_shape(workload_id, phase_id),
            }
        )

    milestone3_runs: list[dict[str, Any]] = []
    for workload_id, phase_id, config_id, arch in product(
        WORKLOADS,
        PHASES,
        MILESTONE3_CONFIG_IDS,
        milestone3_variants(),
    ):
        milestone3_runs.append(
            {
                "suite": "milestone3",
                "run_id": run_id("milestone3", workload_id, phase_id, config_id, arch.arch_id),
                "workload_id": workload_id,
                "phase_id": phase_id,
                "config_id": config_id,
                "arch_id": arch.arch_id,
                "num_quantmac": arch.num_quantmac,
                "num_rescalemac": arch.num_rescalemac,
                "shape": phase_shape(workload_id, phase_id),
            }
        )

    legacy_runs = [
        {
            "suite": "legacy_validation",
            "run_id": "legacy_validation__baseline_fp16",
            "workload_id": "LEGACY_4096",
            "phase_id": "legacy",
            "config_id": "BASELINE_FP16",
            "arch_id": "baseline",
            "num_quantmac": 1,
            "num_rescalemac": 1,
            "shape": {"m": 4096, "n": 4096, "k": 4096},
        },
        {
            "suite": "legacy_validation",
            "run_id": "legacy_validation__legacy_w4a16",
            "workload_id": "LEGACY_4096",
            "phase_id": "legacy",
            "config_id": "LEGACY_W4A16",
            "arch_id": "baseline",
            "num_quantmac": 1,
            "num_rescalemac": 1,
            "shape": {"m": 4096, "n": 4096, "k": 4096},
        },
        {
            "suite": "legacy_validation",
            "run_id": "legacy_validation__legacy_nvfp4_full",
            "workload_id": "LEGACY_4096",
            "phase_id": "legacy",
            "config_id": "LEGACY_NVFP4_FULL",
            "arch_id": "baseline",
            "num_quantmac": 1,
            "num_rescalemac": 1,
            "shape": {"m": 4096, "n": 4096, "k": 4096},
        },
    ]

    return {
        "schema_version": 1,
        "defaults": {
            "accuracy_floor": DEFAULT_ACCURACY_FLOOR,
            "saturation_tolerance": DEFAULT_SATURATION_TOLERANCE,
        },
        "proposal_runs": proposal_runs,
        "milestone3_runs": milestone3_runs,
        "legacy_validation_runs": legacy_runs,
    }


def manifest_runs(manifest: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    key = {
        "proposal": "proposal_runs",
        "milestone3": "milestone3_runs",
        "legacy_validation": "legacy_validation_runs",
    }[suite]
    return list(manifest.get(key, []))


def get_quant_config(config_id: str) -> QuantConfig | dict[str, Any]:
    if config_id in QUANT_CONFIGS:
        return QUANT_CONFIGS[config_id]
    if config_id in SPECIAL_CONFIGS:
        return SPECIAL_CONFIGS[config_id]
    raise KeyError(f"Unknown config_id: {config_id}")


def manifest_or_default(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = DEFAULT_MANIFEST_PATH
    if path.exists():
        return json.loads(path.read_text())
    manifest = default_manifest()
    write_json_file(path, manifest)
    return manifest
