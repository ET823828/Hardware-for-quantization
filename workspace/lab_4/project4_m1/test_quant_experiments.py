from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import analyze_results
import dump_accuracy_snapshot
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
    def test_qwen2_vl_conversation_resolves_image_path_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "example.jpg"
            image_path.write_bytes(b"fake")
            conversation = dump_accuracy_snapshot.build_qwen2_vl_conversation(
                str(image_path),
                "Describe the scene.",
            )

        self.assertEqual(len(conversation), 1)
        content = conversation[0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertTrue(content[0]["path"].endswith("example.jpg"))
        self.assertEqual(content[1]["text"], "Describe the scene.")

    def test_openvla_prompt_builder_wraps_instruction(self) -> None:
        prompt = dump_accuracy_snapshot.build_openvla_prompt("pick up the red block.")
        self.assertEqual(prompt, "In: What action should the robot take to pick up the red block?\nOut:")

    def test_build_two_level_workload_uses_rank1_tensor_scales_and_split_n(self) -> None:
        tensor_workload = run_sweeps.build_two_level_workload(
            shape={"m": 4, "n": 16, "k": 16},
            block_size=4,
            fine_scale_bits=8,
            coarse_scale_bits=16,
            acc_bits=32,
            coarse_granularity="tensor",
            split_n=True,
            rank1_tensor_scales=True,
        )
        row_workload = run_sweeps.build_two_level_workload(
            shape={"m": 4, "n": 16, "k": 16},
            block_size=4,
            fine_scale_bits=8,
            coarse_scale_bits=16,
            acc_bits=32,
            coarse_granularity="row",
            split_n=True,
        )

        tensor_scale_a = tensor_workload["workload"]["einsums"][0]["tensor_accesses"][1]["projection"]
        row_scale_a = row_workload["workload"]["einsums"][0]["tensor_accesses"][1]["projection"]
        tensor_scale_w = tensor_workload["workload"]["einsums"][4]["tensor_accesses"][1]["projection"]
        row_scale_w = row_workload["workload"]["einsums"][4]["tensor_accesses"][1]["projection"]
        tensor_iter = tensor_workload["workload"]["iteration_space_shape"]
        row_iter = row_workload["workload"]["iteration_space_shape"]

        self.assertEqual(tensor_scale_a, ["ga"])
        self.assertEqual(tensor_scale_w, ["gw"])
        self.assertEqual(row_scale_a, ["m"])
        self.assertEqual(row_scale_w, ["nb", "ni"])
        self.assertIn("ga", tensor_iter)
        self.assertIn("gw", tensor_iter)
        self.assertEqual(row_iter["nb"], "0 <= nb < 1")
        self.assertEqual(row_iter["ni"], "0 <= ni < 16")

    def test_build_one_level_workload_splits_large_n_dimension_into_small_ranks(self) -> None:
        workload = run_sweeps.build_one_level_workload(
            shape={"m": 128, "n": 11008, "k": 4096},
            block_size=32,
            scale_bits=8,
            acc_bits=32,
        )

        iter_shape = workload["workload"]["iteration_space_shape"]
        matmul = next(e for e in workload["workload"]["einsums"] if e["name"] == "MatMulBlock")

        self.assertEqual(iter_shape["nbo"], "0 <= nbo < 43")
        self.assertEqual(iter_shape["nbi"], "0 <= nbi < 16")
        self.assertEqual(iter_shape["ni"], "0 <= ni < 16")
        self.assertEqual(matmul["tensor_accesses"][1]["projection"], ["nbo", "nbi", "ni", "kb", "ki"])
        self.assertEqual(matmul["tensor_accesses"][2]["projection"], ["m", "nbo", "nbi", "ni", "kb"])

    def test_split_n_projection_splits_256_into_two_small_ranks(self) -> None:
        iter_shape, projection = run_sweeps.split_n_projection({"m": 128, "n": 256, "k": 4096})

        self.assertEqual(iter_shape, {"nb": "0 <= nb < 16", "ni": "0 <= ni < 16"})
        self.assertEqual(projection, ["nb", "ni"])

    def test_llm_prefill_two_level_uses_compact_mapper_workload(self) -> None:
        run_spec = next(
            row
            for row in experiment_defs.default_manifest()["proposal_runs"]
            if row["workload_id"] == "LLM" and row["phase_id"] == "prefill" and row["config_id"] == "C5"
        )

        workload = run_sweeps.build_workload(run_spec)
        names = [einsum["name"] for einsum in workload["workload"]["einsums"]]
        iter_shape = workload["workload"]["iteration_space_shape"]
        tensor_rescale_w = workload["workload"]["einsums"][-1]

        self.assertTrue(run_sweeps.uses_compact_two_level_workload(run_spec))
        self.assertEqual(len(names), 7)
        self.assertNotIn("TensorScaleA", names)
        self.assertNotIn("BlockScaleW", names)
        self.assertNotIn("TensorQuantW", names)
        self.assertIn("RescaleTensorA", names)
        self.assertIn("RescaleTensorW", names)
        self.assertEqual(iter_shape["nbo"], "0 <= nbo < 43")
        self.assertEqual(tensor_rescale_w["tensor_accesses"][1]["projection"], ["gw"])

    def test_vla_prefill_two_level_compact_workload_splits_n_256(self) -> None:
        run_spec = next(
            row
            for row in experiment_defs.default_manifest()["proposal_runs"]
            if row["workload_id"] == "VLA" and row["phase_id"] == "prefill" and row["config_id"] == "C5"
        )

        workload = run_sweeps.build_workload(run_spec)
        iter_shape = workload["workload"]["iteration_space_shape"]
        projections = [
            access["projection"]
            for einsum in workload["workload"]["einsums"]
            for access in einsum["tensor_accesses"]
        ]

        self.assertTrue(run_sweeps.uses_compact_two_level_workload(run_spec))
        self.assertNotIn("n", iter_shape)
        self.assertEqual(iter_shape["nb"], "0 <= nb < 16")
        self.assertEqual(iter_shape["ni"], "0 <= ni < 16")
        self.assertTrue(any(projection == ["m", "nb", "ni", "kb"] for projection in projections))
        self.assertFalse(any("n" in projection for projection in projections))

    def test_filter_completed_runs_skips_ok_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "hardware.csv"
            write_csv(
                csv_path,
                [
                    {"run_id": "done", "status": "ok"},
                    {"run_id": "retry", "status": "error"},
                ],
            )
            runs = [{"run_id": "done"}, {"run_id": "retry"}, {"run_id": "missing"}]

            filtered = run_sweeps.filter_completed_runs(runs, csv_path, rerun_ok=False)
            self.assertEqual([run["run_id"] for run in filtered], ["retry", "missing"])
            self.assertEqual(run_sweeps.filter_completed_runs(runs, csv_path, rerun_ok=True), runs)

    def test_filter_runs_by_selectors(self) -> None:
        runs = [
            {"workload_id": "LLM", "phase_id": "decode", "config_id": "C1", "arch_id": "baseline"},
            {"workload_id": "LLM", "phase_id": "prefill", "config_id": "C7", "arch_id": "baseline"},
            {"workload_id": "VLM", "phase_id": "prefill", "config_id": "C7", "arch_id": "baseline"},
        ]
        args = Namespace(workload=["LLM"], phase=["prefill"], config=["C7"], arch=None)

        filtered = run_sweeps.filter_runs_by_selectors(runs, args)

        self.assertEqual(filtered, [runs[1]])

    def test_prefill_m_sweep_manifest_has_unique_hardware_runs(self) -> None:
        manifest = experiment_defs.default_manifest()
        runs = manifest["prefill_m_sweep_runs"]
        run_ids = {run["run_id"] for run in runs}
        shapes_by_m = {run["shape"]["m"] for run in runs}
        configs = {run["config_id"] for run in runs}

        self.assertEqual(len(runs), 20)
        self.assertEqual(len(run_ids), 20)
        self.assertEqual(shapes_by_m, {128, 256, 512, 1024, 2048})
        self.assertEqual(configs, {"BASELINE_FP16", "C0", "C1", "C7"})
        self.assertTrue(all(run["suite"] == "prefill_m_sweep" for run in runs))
        self.assertTrue(all(run["workload_id"] == "LLM" for run in runs))
        self.assertTrue(all(run["shape"]["n"] == 11008 and run["shape"]["k"] == 4096 for run in runs))
        self.assertEqual(
            run_sweeps.target_hardware_csv("prefill_m_sweep").name,
            "prefill_m_sweep_hardware_summary.csv",
        )

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

    def test_select_best_mapping_index_falls_back_to_derived_totals(self) -> None:
        class FakeFrame:
            def __init__(self, rows):
                self._rows = rows
                self.columns = list(rows[0].keys())
                self.iloc = self

            def __len__(self):
                return len(self._rows)

            def __getitem__(self, idx):
                return self._rows[idx]

        rows = [
            {
                "MatMul<SEP>energy<SEP>Compute<SEP>compute": 10.0,
                "MatMul<SEP>latency<SEP>Compute": 5.0,
            },
            {
                "MatMul<SEP>energy<SEP>Compute<SEP>compute": 20.0,
                "MatMul<SEP>latency<SEP>Compute": 2.0,
            },
        ]
        df = FakeFrame(rows)
        best_idx, energy_total, latency_total, energy_col, latency_col = run_sweeps.select_best_mapping_index(df, ["MatMul"])
        self.assertEqual(best_idx, 1)
        self.assertEqual(energy_total, 20.0)
        self.assertEqual(latency_total, 2.0)
        self.assertEqual(energy_col, "derived_from_breakdown")
        self.assertEqual(latency_col, "derived_from_breakdown")

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
                        suite="proposal",
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
                        suite="proposal",
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

    def test_dump_accuracy_snapshot_from_files_packs_npz(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed in this test environment")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            a_path = tmp / "a.npy"
            w_path = tmp / "w.npy"
            out_path = tmp / "snapshot.npz"

            np.save(a_path, np.arange(12, dtype=np.float32).reshape(3, 4))
            np.save(w_path, np.arange(20, dtype=np.float32).reshape(5, 4))

            dump_accuracy_snapshot.command_from_files(
                Namespace(
                    a_path=str(a_path),
                    w_path=str(w_path),
                    output=str(out_path),
                    a_key=None,
                    w_key=None,
                )
            )

            with np.load(out_path, allow_pickle=False) as payload:
                self.assertEqual(payload["a"].shape, (3, 4))
                self.assertEqual(payload["w"].shape, (5, 4))

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

    def test_proposal_completion_status_requires_manifest_row_counts_and_real_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            hardware_csv = tmp / "proposal_hardware.csv"
            accuracy_csv = tmp / "accuracy.csv"

            write_csv(
                hardware_csv,
                [
                    {
                        "suite": "proposal",
                        "run_id": "proposal__llm__decode__c7__baseline",
                        "status": "ok",
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
                        "input_source": str(tmp / "missing_snapshot.npz"),
                    }
                ],
            )

            original_values = (
                analyze_results.PROPOSAL_HARDWARE_CSV,
                analyze_results.ACCURACY_CSV,
            )
            try:
                analyze_results.PROPOSAL_HARDWARE_CSV = hardware_csv
                analyze_results.ACCURACY_CSV = accuracy_csv
                status = analyze_results.proposal_completion_status()
                self.assertFalse(status["proposal_ready"])
                self.assertEqual(status["expected_hardware_rows"], 60)
                self.assertEqual(status["expected_accuracy_rows"], 60)
                self.assertEqual(status["observed_hardware_rows"], 1)
                self.assertEqual(status["observed_accuracy_rows"], 1)
                self.assertTrue(any("row count mismatch" in issue for issue in status["issues"]))
            finally:
                (
                    analyze_results.PROPOSAL_HARDWARE_CSV,
                    analyze_results.ACCURACY_CSV,
                ) = original_values

    def test_analyze_prefill_m_sweep_writes_ratios_and_rescale_pct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            hardware_csv = tmp / "prefill_m_sweep_hardware_summary.csv"
            sensitivity_csv = tmp / "prefill_m_sensitivity.csv"
            c7_breakdown = tmp / "c7_breakdown.json"
            c7_breakdown.write_text(
                json.dumps(
                    {
                        "energy_total": 250.0,
                        "energy_per_einsum": {
                            "MatMulNVFP4": 60.0,
                            "RescaleBlockA": 90.0,
                            "RescaleBlockW": 60.0,
                            "RescaleTensorA": 20.0,
                            "RescaleTensorW": 10.0,
                            "BlockQuantA": 10.0,
                        },
                    }
                )
            )
            base_row = {
                "suite": "prefill_m_sweep",
                "status": "ok",
                "workload_id": "LLM",
                "phase_id": "prefill_m128",
                "arch_id": "baseline",
                "m": 128,
                "n": 11008,
                "k": 4096,
                "latency_cycles": 100.0,
                "breakdown_file": "",
            }
            write_csv(
                hardware_csv,
                [
                    {
                        **base_row,
                        "run_id": "prefill_m_sweep__llm__prefill_m128__baseline_fp16__baseline",
                        "config_id": "BASELINE_FP16",
                        "energy_total_pj": 200.0,
                    },
                    {
                        **base_row,
                        "run_id": "prefill_m_sweep__llm__prefill_m128__c0__baseline",
                        "config_id": "C0",
                        "energy_total_pj": 100.0,
                    },
                    {
                        **base_row,
                        "run_id": "prefill_m_sweep__llm__prefill_m128__c1__baseline",
                        "config_id": "C1",
                        "energy_total_pj": 150.0,
                    },
                    {
                        **base_row,
                        "run_id": "prefill_m_sweep__llm__prefill_m128__c7__baseline",
                        "config_id": "C7",
                        "energy_total_pj": 250.0,
                        "latency_cycles": 50.0,
                        "breakdown_file": str(c7_breakdown),
                    },
                ],
            )

            original_values = (
                analyze_results.PREFILL_M_SWEEP_HARDWARE_CSV,
                analyze_results.PREFILL_M_SENSITIVITY_CSV,
            )
            try:
                analyze_results.PREFILL_M_SWEEP_HARDWARE_CSV = hardware_csv
                analyze_results.PREFILL_M_SENSITIVITY_CSV = sensitivity_csv

                rows = analyze_results.analyze_prefill_m_sweep()

                self.assertEqual(len(rows), 1)
                self.assertAlmostEqual(float(rows[0]["c7_vs_fp16_energy"]), 1.25)
                self.assertAlmostEqual(float(rows[0]["c7_vs_ideal_energy"]), 2.5)
                self.assertAlmostEqual(float(rows[0]["c7_latency_vs_fp16"]), 0.5)
                self.assertAlmostEqual(float(rows[0]["c1_vs_c7_energy"]), 0.6)
                self.assertAlmostEqual(float(rows[0]["c7_rescale_pct"]), 72.0)
                self.assertTrue(sensitivity_csv.exists())
            finally:
                (
                    analyze_results.PREFILL_M_SWEEP_HARDWARE_CSV,
                    analyze_results.PREFILL_M_SENSITIVITY_CSV,
                ) = original_values


if __name__ == "__main__":
    unittest.main()
