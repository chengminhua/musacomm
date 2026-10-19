"""Synthetic CPU-only evidence tests; no measured numbers are fabricated for publication."""
from __future__ import annotations

import copy
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from musacomm.catalog import catalog
from musacomm import exporter
from musacomm.runtime.helpers import summarize_blocks


def fixture(directory: Path, primitive: str = "all_reduce") -> Path:
    results = directory / "journals"
    results.mkdir()
    spec = catalog.new_spec("collective", primitive, "A8", "bf16", environment_id="synthetic-fixture")
    if primitive == "all_to_all_v":
        spec["route"] = next(catalog.variable_routes(8))
        spec["size"] = next(catalog.variable_sizes(spec["route"], 8, "bf16"))
    else:
        spec["reduce_op"] = "sum"
        spec["size"] = next(catalog.fixed_sizes(primitive, 8, "bf16"))
    case = catalog.build_case(spec)
    blocks = [{"iterations": 200,
               "ranks": [{"rank": rank, "event_seconds": 0.25 + rank / 1000,
                          "host_seconds": 0.3 + rank / 1000} for rank in range(8)]} for _ in range(5)]
    correctness = [{"valid": True, "wrong_elements": 0} for _ in range(8)]
    for rank in range(8):
        env = {"record": "run_environment", "rank": rank, "local_rank": rank,
               "physical_gpu": rank, "host_id": "fixture-host", "environment_id": "synthetic-fixture",
               "manifest": {"sha256": "a" * 64, "cases": 1, "world_sizes": [8], "case_ids": [case["case_id"]]},
               "runner_sha256": "b" * 64, "support_code_hashes": {"helpers": "c" * 64},
               "settings": {"blocks": 5, "block_seconds": 0.2, "warmup": 20},
               "communication_environment": {}, "native_library": {"version_code": 0},
               "torch_version": "synthetic", "torch_musa_version": "synthetic"}
        row = {key: copy.deepcopy(case[key]) for key in (
            "case_id", "case_hash", "config_id", "primitive", "dtype", "reduce_op", "root", "buffer_mode", "size")}
        row.update(record="case_result", rank=rank, status="passed", requested_case_status="passed",
                   requested_backend_satisfied=True, formal_timing_qualified=True,
                   correctness_pre=correctness, timing_output_validation=correctness, correctness_post=correctness,
                   timing_blocks=blocks, summary=summarize_blocks(case, blocks),
                   implementation="synthetic_test_fixture", execution_signature="d" * 64,
                   platform_status="not_reached", official_status="reference_unverified")
        summary = {"record": "run_summary", "rank": rank, "manifest_cases": 1, "recorded_cases": 1,
                   "all_requested_cases_passed": True, "status_counts": {"passed": 1},
                   "requested_case_status_counts": {"passed": 1}}
        (results / f"rank-{rank:04d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (env, row, summary)))
    return results


def change(results: Path, rank: int, edit) -> None:
    path = results / f"rank-{rank:04d}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    edit(records)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


class ExportTests(unittest.TestCase):
    def assert_rejected(self, results, output, edit):
        change(results, 0, edit)
        with self.assertRaises(ValueError):
            exporter.export_results(results, output)
        self.assertFalse(output.exists())

    def test_import_has_no_argument_parsing_or_gpu_side_effect(self):
        saved = list(sys.argv)
        try:
            sys.argv[:] = ["unrelated-program", "--unrelated-argument"]
            importlib.reload(exporter)
            self.assertNotIn("torch", vars(exporter))
        finally:
            sys.argv[:] = saved

    def test_complete_export_recomputes_summary_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            results = fixture(directory)
            output = directory / "results.csv"
            report = exporter.export_results(results, output)
            self.assertEqual(report["ranks"], 8)
            self.assertEqual(report["cases"], 1)
            self.assertEqual(len(report["sha256"]), 64)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertGreater(float(rows[0]["algbw_GBps"]), 0)
            self.assertAlmostEqual(float(rows[0]["normalized_busbw_GBps"]),
                                   float(rows[0]["algbw_GBps"]) * 1.75)
            previous = output.read_bytes()
            with self.assertRaises(FileExistsError):
                exporter.export_results(results, output)
            self.assertEqual(output.read_bytes(), previous)

    def test_alltoallv_export_has_no_invented_algbw_or_api_count(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory / "results.csv"
            exporter.export_results(fixture(directory, "all_to_all_v"), output)
            with output.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["algbw_GBps"], "")
            self.assertEqual(row["normalized_busbw_GBps"], "")
            self.assertEqual(row["api_count"], "")
            self.assertGreater(float(row["global_nonself_effective_GBps"]), 0)

    def test_missing_eighth_journal_is_not_a_seven_rank_success(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            results = fixture(directory)
            (results / "rank-0007.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "exactly eight"):
                exporter.export_results(results, directory / "results.csv")

    def test_gates_reject_incomplete_mismatched_and_unqualified_records(self):
        mutations = {
            "wrong-world": lambda r: r[0]["manifest"].update(world_sizes=[7]),
            "missing-membership": lambda r: r[0]["manifest"].pop("case_ids"),
            "wrong-membership": lambda r: r[0]["manifest"].update(case_ids=["different"]),
            "missing-sha": lambda r: r[0]["manifest"].pop("sha256"),
            "missing-row": lambda r: r.pop(1),
            "bad-summary-count": lambda r: r[2].update(recorded_cases=0),
            "pending-backend": lambda r: r[1].update(requested_backend_satisfied=False),
            "wrong-output": lambda r: r[1]["correctness_post"][3].update(wrong_elements=1),
            "missing-correctness-rank": lambda r: r[1]["correctness_pre"].pop(),
            "missing-timing-rank": lambda r: r[1]["timing_blocks"][0]["ranks"].pop(),
            "bad-window": lambda r: [s.update(event_seconds=0.1) for s in r[1]["timing_blocks"][0]["ranks"]],
            "bad-sample": lambda r: r[1]["timing_blocks"][0]["ranks"][0].update(host_seconds=float("nan")),
            "bad-aggregate": lambda r: r[1]["summary"].update(median_max_rank_device_seconds_per_op=1.0),
            "another-host": lambda r: r[0].update(host_id="another-host"),
            "duplicated-gpu": lambda r: r[0].update(physical_gpu=1),
            "different-code": lambda r: r[0].update(runner_sha256="f" * 64),
            "not-formal": lambda r: r[1].update(formal_timing_qualified=False),
            "no-summary": lambda r: r.pop(),
            "cleanup-failed": lambda r: r.append({"record": "cleanup_error", "rank": 0}),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.assert_rejected(fixture(directory), directory / "results.csv", mutation)


if __name__ == "__main__":
    unittest.main()
