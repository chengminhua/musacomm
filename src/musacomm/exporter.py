"""Export complete, eight-rank-validated result sets without importing MUSA.

Passing these journal gates does not establish exclusive hardware ownership.
The launcher records independent resource/contamination checks beside results.
All bandwidth values are logical effective rates, not physical-wire counters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .runtime.helpers import summarize_blocks


EXPECTED_RANKS = 8
COLUMNS = [
    "case_id", "config_id", "environment_id", "hosts", "ranks", "primitive", "implementation",
    "dtype", "reduce_op", "root", "buffer_mode", "actual_bytes", "api_count",
    "median_max_rank_device_us", "median_max_rank_host_us", "algbw_GBps",
    "normalized_busbw_GBps", "global_nonself_effective_GBps", "rank_mean_nonself_effective_GBps",
    "global_cross_node_effective_GBps", "per_direction_effective_GBps", "bidirectional_sum_effective_GBps",
    "block_relative_range", "platform_status", "official_status", "execution_signature",
]
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_positive(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _read_journal(path: Path) -> tuple[dict, dict, list[dict]]:
    try:
        log = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read complete JSON journal {path.name}: {exc}") from exc
    _require(all(isinstance(row, dict) for row in log), f"non-object record in {path.name}")
    environments = [(index, row) for index, row in enumerate(log) if row.get("record") == "run_environment"]
    summaries = [(index, row) for index, row in enumerate(log) if row.get("record") == "run_summary"]
    _require(len(environments) == len(summaries) == 1, f"one environment and summary required in {path.name}")
    (env_index, env), (summary_index, summary) = environments[0], summaries[0]
    _require(env_index < summary_index, "journal environment/summary order is invalid")
    _require(not any(row.get("record") in {"case_abort", "cleanup_error"} for row in log),
             "aborted case or cleanup error prevents qualified export")
    indexed_rows = [(index, row) for index, row in enumerate(log) if row.get("record") == "case_result"]
    rows = [row for _, row in indexed_rows]
    _require(all(env_index < index < summary_index for index, _ in indexed_rows),
             "case result outside environment/summary boundary")
    _require(all(row.get("rank") == env.get("rank") for row in log), "journal record rank mismatch")
    return env, summary, rows


def _validate_result(row: dict, rank: int) -> None:
    _require(row.get("rank") == rank, "case result rank mismatch")
    _require(row.get("status") == row.get("requested_case_status") == "passed"
             and row.get("formal_timing_qualified") is True,
             "result is not qualified for performance export")
    _require(row.get("requested_backend_satisfied") is True, "requested backend was not satisfied")
    for field in ("correctness_pre", "timing_output_validation", "correctness_post"):
        values = row.get(field)
        _require(isinstance(values, list) and len(values) == EXPECTED_RANKS,
                 f"missing full all-rank {field}")
        _require(all(isinstance(v, dict) and v.get("valid") is True
                     and type(v.get("wrong_elements")) is int and v["wrong_elements"] == 0
                     for v in values), f"failed all-rank {field}")
    blocks = row.get("timing_blocks")
    _require(isinstance(blocks, list) and len(blocks) >= 5, "at least five timing blocks required")
    for block in blocks:
        _require(type(block.get("iterations")) is int and block["iterations"] > 0,
                 "invalid iteration count")
        samples = block.get("ranks")
        _require(isinstance(samples, list) and len(samples) == EXPECTED_RANKS,
                 "timing block is missing ranks")
        _require(all(isinstance(s, dict) and type(s.get("rank")) is int for s in samples)
                 and sorted(s["rank"] for s in samples) == list(range(EXPECTED_RANKS)),
                 "timing block must contain each rank exactly once")
        _require(all(_finite_positive(s.get(k)) for s in samples for k in ("event_seconds", "host_seconds")),
                 "timing samples must be finite and positive")
        _require(max(s["event_seconds"] for s in samples) >= 0.2, "timing window gate failed")
    try:
        case = {"primitive": row["primitive"], "size": row["size"],
                "topology": {"ranks": EXPECTED_RANKS}, "direction": row.get("direction")}
        expected = summarize_blocks(case, blocks)
        _require(_canonical(row.get("summary")) == _canonical(expected),
                 "stored summary does not match raw timing blocks")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed result/summary") from exc


def _validated_rows(results: Path) -> tuple[list[dict], list[dict]]:
    files = sorted(results.glob("rank-*.jsonl"))
    expected_names = [f"rank-{rank:04d}.jsonl" for rank in range(EXPECTED_RANKS)]
    _require([path.name for path in files] == expected_names,
             "exactly eight rank journals (rank-0000 through rank-0007) are required")
    all_rows, environments = [], []
    for rank, path in enumerate(files):
        env, summary, rows = _read_journal(path)
        _require(env.get("rank") == rank and env.get("local_rank") == rank,
                 "rank journal/environment mapping mismatch")
        manifest = env.get("manifest", {})
        _require(manifest.get("world_sizes") == [EXPECTED_RANKS], "manifest must describe exactly eight ranks")
        _require(isinstance(manifest.get("sha256"), str) and bool(_HASH.fullmatch(manifest["sha256"])),
                 "missing/invalid manifest SHA-256")
        count, case_ids = manifest.get("cases"), manifest.get("case_ids")
        _require(type(count) is int and count > 0 and isinstance(case_ids, list)
                 and len(case_ids) == count and all(isinstance(v, str) for v in case_ids)
                 and len(set(case_ids)) == count, "manifest case membership is missing or invalid")
        _require(summary.get("all_requested_cases_passed") is True
                 and summary.get("manifest_cases") == count
                 and summary.get("recorded_cases") == count and len(rows) == count,
                 "incomplete or non-passed journal")
        _require([row.get("case_id") for row in rows] == case_ids,
                 "case sequence differs from the manifest")
        _require(summary.get("status_counts") == {"passed": count}
                 and summary.get("requested_case_status_counts") == {"passed": count},
                 "summary status counts disagree with qualified rows")
        for row in rows:
            _validate_result(row, rank)
        all_rows.append(rows)
        environments.append(env)
    host_ids = {env.get("host_id") for env in environments}
    _require(len(host_ids) == 1 and all(isinstance(host, str) and host for host in host_ids),
             "v0.1 export requires one bound host")
    physical = [env.get("physical_gpu") for env in environments]
    _require(all(type(gpu) is int and gpu >= 0 for gpu in physical) and len(set(physical)) == EXPECTED_RANKS,
             "physical GPU binding must be unique across ranks")
    for field in ("manifest", "environment_id", "runner_sha256", "support_code_hashes",
                  "settings", "communication_environment", "native_library", "torch_version", "torch_musa_version"):
        _require(all(field in env for env in environments), f"missing environment evidence: {field}")
        _require(len({_canonical(env[field]) for env in environments}) == 1,
                 f"all-rank environment mismatch: {field}")
    _require(environments[0]["environment_id"] not in {None, "", "UNVERIFIED"}, "environment binding is unverified")
    aggregate_fields = ("case_id", "case_hash", "config_id", "primitive", "dtype", "reduce_op", "root",
                        "buffer_mode", "size", "implementation", "execution_signature", "summary", "timing_blocks",
                        "correctness_pre", "timing_output_validation", "correctness_post")
    for group in zip(*all_rows):
        for field in aggregate_fields:
            _require(all(field in row for row in group)
                     and len({_canonical(row[field]) for row in group}) == 1,
                     f"all-rank aggregate mismatch: {field}")
    return environments, all_rows[0]


def export_results(results: Path, output: Path) -> dict[str, Any]:
    """Validate raw journals and publish a new CSV, never overwriting a file.

    Raises ValueError on incomplete/malformed evidence and FileExistsError when
    the output already exists. No partial CSV is published on validation error.
    """
    results, output = Path(results), Path(output)
    try:
        environments, rows = _validated_rows(results)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed result journal") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".musacomm-export-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            for result in rows:
                summary, bandwidth = result["summary"], result["summary"]["device_bandwidth"]
                row = {key: result.get(key) for key in COLUMNS}
                row.update(environment_id=environments[0]["environment_id"], hosts=environments[0]["host_id"],
                           ranks=EXPECTED_RANKS, actual_bytes=result["size"]["actual"], api_count=result["size"].get("api_count"),
                           median_max_rank_device_us=summary["median_max_rank_device_seconds_per_op"] * 1e6,
                           median_max_rank_host_us=summary["median_max_rank_host_seconds_per_op"] * 1e6,
                           algbw_GBps=bandwidth.get("algbw_GBps"), normalized_busbw_GBps=bandwidth.get("busbw_GBps"),
                           block_relative_range=summary["device_relative_range"])
                for key in COLUMNS:
                    if key.endswith("effective_GBps"):
                        row[key] = bandwidth.get(key)
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"export": str(output), "ranks": EXPECTED_RANKS, "cases": len(rows),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "warning": "logical algbw and normalized busbw are not measured physical-wire counters; journal validation is not resource isolation"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = export_results(args.results, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"export refused: {exc}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
