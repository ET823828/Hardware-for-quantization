from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import analyze_results
import experiment_defs
import run_sweeps


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class QuantExperimentTests(unittest.TestCase):
    def test_build_two_level_workload_uses_tensor_vs_row_projections(self) -> None:
        tensor_workload = run_sweeps.build_two_level_workload(
            shape={"m": 4, "n": 8, "k": 16},
            block_size=4,
            fine_scale_bits=8,
            coarse_scale_bits=16,
            acc_bits=32,
            coarse_granularity="tensor",
        )
        row_workload = run_sweeps.build_two_level_workload(
            shape={"m": 4, "n": 8, "k": 16},
            block_size=4,
            fine_scale_bits=8,
            coarse_scale_bits=16,
            acc_bits=32,
            coarse_granularity="row",
        )

        tensor_scale_a = tensor_workload["workload"]["einsums"][0]["tensor_accesses"][1]["projection"]
        row_scale_a = row_workload["workload"]["einsums"][0]["tensor_accesses"][1]["projection"]
        tensor_scale_w = tensor_workload["workload"]["einsums"][4]["tensor_accesses"][1]["projection"]
        row_scale_w = row_workload["workload"]["einsums"][4]["tensor_accesses"][1]["projection"]

        self.assertEqual(tensor_scale_a, [])
        self.assertEqual(tensor_scale_w, [])
        self.assertEqual(row_scale_a, ["m"])
        self.assertEqual(row_scale_w, ["n"])

    def test_two_level_quantization_assigns_different_coarse_scales_for_tensor_vs_row(self) -> None:
        rows = [
            [6.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
        ]

        tensor_quant = run_sweeps.quantize_two_level_rows(
            rows,
            block_size=2,
            fine_scale_format="fp16",
            coarse_scale_format="e8m0",
            coarse_granularity="tensor",
        )
        row_quant = run_sweeps.quantize_two_level_rows(
            rows,
            block_size=2,
            fine_scale_format="fp16",
            coarse_scale_format="e8m0",
            coarse_granularity="row",
        )

        tensor_scales = [item[1] for item in tensor_quant]
        row_scales = [item[1] for item in row_quant]
        self.assertEqual(tensor_scales[0], tensor_scales[1])
        self.assertNotEqual(row_scales[0], row_scales[1])

    def test_resolve_total_metric_column_requires_unambiguous_total(self) -> None:
        self.assertEqual(run_sweeps.resolve_total_metric_column(["energy", "foo<SEP>energy"], "energy"), "energy")
        self.assertEqual(run_sweeps.resolve_total_metric_column(["latency_total", "x<SEP>latency"], "latency"), "latency_total")

        with self.assertRaises(ValueError):
            run_sweeps.resolve_total_metric_column(["energy_total", "total_energy"], "energy")

        with self.assertRaises(ValueError):
            run_sweeps.resolve_total_metric_column(["einsum<SEP>energy"], "energy")

    def test_effective_bits_and_compression(self) -> None:
        c1_row = {"config_id": "C1", "n": "3072", "k": "3072"}
        c7_row = {"config_id": "C7", "n": "11008", "k": "4096"}

        c1_bits = analyze_results.effective_bits_per_weight(c1_row)
        c7_bits = analyze_results.effective_bits_per_weight(c7_row)

        self.assertAlmostEqual(c1_bits, 4.25)
        self.assertAlmostEqual(c7_bits, 4.5 + (32.0 / (11008.0 * 4096.0)))
        self.assertGreater(16.0 / c1_bits, 1.0)

    def test_run_accuracy_uses_manifest_snapshot_and_marks_missing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            snapshot_path = tmp / "llm.json"
            a_rows = [[float(idx + col) for col in range(4096)] for idx in range(4)]
            w_rows = [[float(idx - col) for col in range(4096)] for idx in range(16)]
            snapshot_path.write_text(json.dumps({"a": a_rows, "w": w_rows}))

            manifest = {
                "schema_version": 2,
                "defaults": {
                    "accuracy_floor": 0.98,
                    "saturation_tolerance": 0.05,
                    "alpha_by_workload": experiment_defs.DEFAULT_ALPHA_BY_WORKLOAD,
                },
                "accuracy_inputs": {
                    "LLM": {
                        "path": str(snapshot_path),
                        "format": "json",
                        "activation_key": "a",
                        "weight_key": "w",
                    }
                },
                "proposal_runs": [
                    {
                        "suite": "proposal",
                        "run_id": "proposal__llm__decode__c7__baseline",
                        "workload_id": "LLM",
                        "phase_id": "decode",
                        "config_id": "C7",
                        "arch_id": "baseline",
                        "num_quantmac": 1,
                        "num_rescalemac": 1,
                        "shape": {"m": 1, "n": 11008, "k": 4096},
                    }
                ],
                "milestone3_runs": [],
                "legacy_validation_runs": [],
            }
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            original_accuracy_csv = run_sweeps.ACCURACY_CSV
            try:
                run_sweeps.ACCURACY_CSV = tmp / "accuracy_summary.csv"
                run_sweeps.command_run_accuracy(
                    Namespace(
                        manifest=str(manifest_path),
                        limit=None,
                        sample_m_max=4,
                        sample_n_max=16,
                        input_mode="proposal",
                    )
                )
                with run_sweeps.ACCURACY_CSV.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "ok")
                self.assertEqual(rows[0]["input_source"], str(snapshot_path))

                manifest["accuracy_inputs"]["LLM"]["path"] = str(tmp / "missing.json")
                manifest_path.write_text(json.dumps(manifest))
                run_sweeps.command_run_accuracy(
                    Namespace(
                        manifest=str(manifest_path),
                        limit=None,
                        sample_m_max=4,
                        sample_n_max=16,
                        input_mode="proposal",
                    )
                )
                with run_sweeps.ACCURACY_CSV.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["status"], "missing_inputs")
            finally:
                run_sweeps.ACCURACY_CSV = original_accuracy_csv

    def test_analyzer_reports_incomplete_proposal_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            hardware_csv = tmp / "proposal_hardware.csv"
            accuracy_csv = tmp / "accuracy.csv"
            combined_csv = tmp / "combined.csv"
            best_csv = tmp / "best.csv"
            pareto_csv = tmp / "pareto.csv"
            adaptive_csv = tmp / "adaptive.csv"
            status_json = tmp / "analysis_status.json"

            write_csv(
                hardware_csv,
                [
                    {
                        "suite": "proposal",
                        "run_id": "proposal__llm__decode__c7__baseline",
                        "status": "generated_only",
                        "workload_id": "LLM",
                        "phase_id": "decode",
                        "config_id": "C7",
                        "arch_id": "baseline",
                        "m": 1,
                        "n": 11008,
                        "k": 4096,
                    }
                ],
            )
            write_csv(
                accuracy_csv,
                [
                    {
                        "suite": "proposal",
                        "run_id": "proposal__llm__decode__c7__baseline",
                        "status": "ok",
                        "workload_id": "LLM",
                        "phase_id": "decode",
                        "config_id": "C7",
                        "cosine_similarity": 0.99,
                    }
                ],
            )

            original_values = (
                analyze_results.PROPOSAL_HARDWARE_CSV,
                analyze_results.ACCURACY_CSV,
                analyze_results.COMBINED_CSV,
                analyze_results.BEST_CONFIGS_CSV,
                analyze_results.PARETO_CSV,
                analyze_results.ADAPTIVE_CSV,
                analyze_results.ANALYSIS_STATUS_JSON,
            )
            try:
                analyze_results.PROPOSAL_HARDWARE_CSV = hardware_csv
                analyze_results.ACCURACY_CSV = accuracy_csv
                analyze_results.COMBINED_CSV = combined_csv
                analyze_results.BEST_CONFIGS_CSV = best_csv
                analyze_results.PARETO_CSV = pareto_csv
                analyze_results.ADAPTIVE_CSV = adaptive_csv
                analyze_results.ANALYSIS_STATUS_JSON = status_json

                rows, best_rows, adaptive_rows = analyze_results.analyze_proposal(0.98)
                self.assertEqual(rows, [])
                self.assertEqual(best_rows, [])
                self.assertEqual(adaptive_rows, [])
                status = json.loads(status_json.read_text())
                self.assertFalse(status["proposal_ready"])
                self.assertFalse(combined_csv.exists())
            finally:
                (
                    analyze_results.PROPOSAL_HARDWARE_CSV,
                    analyze_results.ACCURACY_CSV,
                    analyze_results.COMBINED_CSV,
                    analyze_results.BEST_CONFIGS_CSV,
                    analyze_results.PARETO_CSV,
                    analyze_results.ADAPTIVE_CSV,
                    analyze_results.ANALYSIS_STATUS_JSON,
                ) = original_values


if __name__ == "__main__":
    unittest.main()
