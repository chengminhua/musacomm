"""Pure-stdlib contracts and correctness oracles; never touches GPUs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterator


REDUCTIONS = {"all_reduce", "reduce_scatter", "reduce"}
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "uint8_bytes": 1}


def catalog_modules(directory: str | Path | None = None):
    """Return the installed, version-matched schemas without changing sys.path.

    An explicit path is accepted only for compatibility with callers that pass
    the bundled catalog's own directory. Arbitrary external schema modules can
    silently change the measurement contract and are not loaded.
    """
    from musacomm.catalog import catalog, schema

    if directory is not None and Path(directory).resolve() != Path(catalog.__file__).resolve().parent:
        raise ValueError("only the bundled musacomm catalog is supported")
    return catalog, schema


def iter_cases(path: str | Path, schema) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            try:
                schema.assert_valid_case(case)
            except Exception as exc:
                raise ValueError(f"manifest line {line_number}: {exc}") from exc
            yield case


def audit_manifest(path: str | Path, schema) -> dict[str, Any]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    seen, worlds, case_ids = set(), set(), []
    for case in iter_cases(path, schema):
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
        case_ids.append(case["case_id"])
        worlds.add(case["topology"]["ranks"])
    if not seen:
        raise ValueError("empty case manifest")
    return {"sha256": digest.hexdigest(), "cases": len(seen), "world_sizes": sorted(worlds),
            "case_ids": case_ids}


@dataclass(frozen=True)
class RankPlan:
    rank: int
    world: int
    primitive: str
    element_bytes: int
    input_elements: int
    output_elements: int
    api_count: int | None
    alias: str | None
    alias_offset: int
    allocated_elements: int
    sendcounts: tuple[int, ...] = ()
    recvcounts: tuple[int, ...] = ()
    senddispls: tuple[int, ...] = ()
    recvdispls: tuple[int, ...] = ()
    global_sdispls: tuple[int, ...] = ()

    @property
    def storage_bytes(self) -> int:
        return self.allocated_elements * self.element_bytes


def make_rank_plan(case: dict[str, Any], rank: int, schema) -> RankPlan:
    schema.assert_valid_case(case)
    if case["status"] != "planned" or case["family"] not in {"collective", "p2p"}:
        raise ValueError("not an executable communication case")
    n, p, size = case["topology"]["ranks"], case["primitive"], case["size"]
    if not 0 <= rank < n or size["actual"] is None:
        raise ValueError("rank out of range or unresolved payload contract")
    d = DTYPE_BYTES[case["dtype"]]
    ins, outs = size["input_bytes_by_rank"][rank], size["output_bytes_by_rank"][rank]
    if ins % d or outs % d:
        raise ValueError("logical bytes are not a whole number of elements")
    ni, no, count = ins // d, outs // d, size.get("api_count")
    alias, offset = None, 0
    if case["buffer_mode"] == "in_place":
        if p in {"all_reduce", "broadcast", "reduce"}:
            alias = "same"
        elif p == "all_gather":
            alias, offset = "input_in_output", rank * count
        elif p == "reduce_scatter":
            alias, offset = "output_in_input", rank * count
        elif p == "gather" and rank == case["root"]:
            alias, offset = "input_in_output", rank * count
        elif p == "scatter" and rank == case["root"]:
            alias, offset = "output_in_input", rank * count
    # A non-null one-element dummy is retained for logically empty buffers.
    allocated = max(1, ni, no) if alias else max(1, ni) + max(1, no)
    kwargs = {}
    if p == "all_to_all_v":
        arrays = schema.alltoallv_counts(case)
        kwargs = {key: tuple(arrays[key][rank])
                  for key in ("sendcounts", "recvcounts", "senddispls", "recvdispls")}
        kwargs["global_sdispls"] = tuple(x for row in arrays["senddispls"] for x in row)
    return RankPlan(rank, n, p, d, ni, no, count, alias, offset, allocated, **kwargs)


def memory_decision(plan: RankPlan, *, free_bytes: int, total_bytes: int,
                    fraction: float, workspace_reserve: int,
                    extra_backend_bytes: int = 0) -> dict[str, Any]:
    if not (0 < fraction <= 0.60) or not 0 <= free_bytes <= total_bytes or total_bytes <= 0:
        raise ValueError("invalid memory evidence or fraction above plan's 60% ceiling")
    if workspace_reserve < 0 or extra_backend_bytes < 0:
        raise ValueError("reserves must be non-negative")
    baseline = total_bytes - free_bytes
    required = plan.storage_bytes + workspace_reserve + extra_backend_bytes
    projected = baseline + required
    ok = projected <= math.floor(total_bytes * fraction) and required <= free_bytes
    return {"allowed": ok, "storage_bytes": plan.storage_bytes,
            "workspace_reserve_bytes": workspace_reserve,
            "extra_backend_reserve_bytes": extra_backend_bytes,
            "free_bytes": free_bytes, "total_bytes": total_bytes,
            "baseline_used_bytes": baseline, "projected_used_bytes": projected,
            "budget_bytes": math.floor(total_bytes * fraction)}


def pattern_value(source: int, destination: int, index: int, seed: int = 23) -> int:
    # Exact in BF16/FP16/FP32/uint8; rank, peer and within-block position vary.
    return (source * 37 + destination * 13 + index * 17 + (index // 251) * 29 + seed) % 251


def reduction_input_value(rank: int, index: int, op: str, n: int = 8,
                          root: int | None = None, receive_count: int | None = None) -> float:
    if op in {"prod", "min", "max"}:
        if receive_count is not None:
            destination, within = divmod(index, receive_count)
            winner = (destination + 1 + within % (n - 1)) % n
        elif root is not None:
            winner = (root + 1 + index % (n - 1)) % n
        else:
            winner = index % n
        if op == "prod":
            # Exactly one rank contributes magnitude 2. For Reduce/RS it is
            # never the rank whose in-place output is checked, so a no-op
            # leaves magnitude 1 and cannot pass even a one-element case.
            sign = -1 if (rank + index) % 3 == 0 else 1
            return sign * (2 if rank == winner else 1)
        delta = (rank - winner) % n
        return 1 + index % 7 + (delta if op == "min" else n - 1 - delta)
    return 1 + index % 7 + rank % 3


def reduction_reference_value(index: int, n: int, op: str, root: int | None = None,
                              receive_count: int | None = None) -> float:
    values = [reduction_input_value(r, index, op, n, root, receive_count) for r in range(n)]
    if op == "sum":
        return sum(values)
    if op == "avg":
        return sum(values) / n
    if op == "prod":
        return math.prod(values)
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    raise ValueError(f"unknown reduction {op}")


def performance_constant(case: dict[str, Any]) -> int | None:
    if case["primitive"] in REDUCTIONS and case["buffer_mode"] == "in_place":
        return 0 if case["reduce_op"] == "sum" else 1
    return None


def input_segments(case: dict[str, Any], plan: RankPlan) -> list[tuple[int, int, int, int]]:
    """(flat offset, length, source rank, destination tag), index restarts per segment."""
    p, r, n, c = plan.primitive, plan.rank, plan.world, plan.api_count
    if not plan.input_elements or p in REDUCTIONS:
        return []
    if p in {"all_to_all", "scatter"}:
        return [(peer * c, c, r, peer) for peer in range(n)]
    if p == "all_to_all_v":
        return [(offset, count, r, peer) for peer, (offset, count)
                in enumerate(zip(plan.senddispls, plan.sendcounts)) if count]
    if p == "send_recv":
        src, dst = case["participants"]
        return [(0, plan.input_elements, r, dst if r == src else src)]
    return [(0, plan.input_elements, r, 0)]


def output_segments(case: dict[str, Any], plan: RankPlan) -> list[tuple[int, int, int, int]]:
    p, r, n, c = plan.primitive, plan.rank, plan.world, plan.api_count
    if not plan.output_elements or p in REDUCTIONS:
        return []
    if p in {"all_gather", "gather"}:
        return [(source * c, c, source, 0) for source in range(n)]
    if p == "all_to_all":
        return [(source * c, c, source, r) for source in range(n)]
    if p == "all_to_all_v":
        return [(offset, count, source, r) for source, (offset, count)
                in enumerate(zip(plan.recvdispls, plan.recvcounts)) if count]
    if p == "scatter":
        return [(0, plan.output_elements, case["root"], r)]
    if p == "broadcast":
        return [(0, plan.output_elements, case["root"], 0)]
    if p == "send_recv":
        src, dst = case["participants"]
        return [(0, plan.output_elements, dst if r == src else src, r)]
    raise ValueError(f"no oracle segments for {p}")


def expected_output_values(case: dict[str, Any], plan: RankPlan) -> list[float]:
    """Small CPU fixtures only; GPU runner validates in bounded chunks instead."""
    if plan.output_elements > 100_000:
        raise ValueError("CPU fixture oracle is intentionally bounded")
    if plan.primitive in REDUCTIONS:
        offset = plan.rank * plan.api_count if plan.primitive == "reduce_scatter" else 0
        return [reduction_reference_value(offset + i, plan.world, case["reduce_op"],
                                         case["root"] if plan.primitive == "reduce" else None,
                                         plan.api_count if plan.primitive == "reduce_scatter" else None)
                for i in range(plan.output_elements)]
    result = [None] * plan.output_elements
    for start, count, src, dst in output_segments(case, plan):
        result[start:start + count] = [pattern_value(src, dst, i) for i in range(count)]
    if any(v is None for v in result):
        raise ValueError("oracle left an output position uncovered")
    return result


def validate_visible_mapping(physical_ids: list[int], visible_devices: str | None,
                             device_count: int, local_rank: int) -> dict[str, Any]:
    """Reject a declared permutation unless the actual runtime mask agrees.

    Physical device enumeration/container device mounts remain the exclusive
    host orchestrator's evidence. No UUID or PCI identity is invented here.
    """
    if (len(set(physical_ids)) != len(physical_ids) or any(i < 0 for i in physical_ids)
            or not 0 <= local_rank < len(physical_ids) or device_count != len(physical_ids)):
        raise ValueError("one visible GPU and unique bound physical ID per local rank required")
    if visible_devices is None:
        if physical_ids != list(range(device_count)):
            raise ValueError("nonidentity physical map requires a matching MUSA_VISIBLE_DEVICES mask")
        evidence = "identity_runtime_enumeration_orchestrator_verified_container_devices"
    else:
        try:
            observed = [int(part.strip()) for part in visible_devices.split(",")]
        except ValueError as exc:
            raise ValueError("UUID/non-numeric runtime masks need an explicit identity adapter") from exc
        if observed != physical_ids:
            raise ValueError("MUSA_VISIBLE_DEVICES disagrees with claimed physical GPU mapping")
        evidence = "explicit_numeric_runtime_visibility_mask"
    return {"logical_device": local_rank, "physical_gpu": physical_ids[local_rank],
            "visible_device_count": device_count, "MUSA_VISIBLE_DEVICES": visible_devices,
            "binding_evidence": evidence}


def choose_iterations(pilot_event_seconds: float, pilot_iterations: int, size: int,
                      target_seconds: float = 0.2, small_threshold: int = 1024**2,
                      maximum: int = 1_000_000) -> int:
    if not math.isfinite(pilot_event_seconds) or pilot_event_seconds <= 0 or pilot_iterations < 1:
        raise ValueError("pilot must have a positive finite device completion duration")
    minimum = 200 if size <= small_threshold else 20
    required = max(minimum, math.ceil(target_seconds * 1.1 * pilot_iterations / pilot_event_seconds))
    if required > maximum:
        raise ValueError(f"pilot needs {required} iterations, above explicit maximum {maximum}")
    return required


def bandwidth_metrics(case: dict[str, Any], seconds_per_operation: float) -> dict[str, Any]:
    if not math.isfinite(seconds_per_operation) or seconds_per_operation <= 0:
        raise ValueError("completion duration must be finite and positive")
    size, p, n = case["size"], case["primitive"], case["topology"]["ranks"]
    if not size["actual"]:
        return {"bandwidth_applicable": False, "algbw_GBps": None, "busbw_GBps": None}
    rate = lambda b: b / seconds_per_operation / 1e9
    if p == "all_to_all_v":
        total = size["global_nonself_payload_bytes"]
        return {"bandwidth_applicable": total > 0, "algbw_GBps": None, "busbw_GBps": None,
                "global_nonself_effective_GBps": rate(total),
                "rank_mean_nonself_effective_GBps": rate(total) / n,
                "global_cross_node_effective_GBps": rate(size["cross_node_payload_bytes"])}
    alg = rate(size["actual"])
    factor = size["busbw_factor"]
    result = {"bandwidth_applicable": True, "algbw_GBps": alg,
              "busbw_GBps": None if factor is None else alg * factor,
              "normalization_payload_bytes": size["actual"],
              "normalization_semantics": size["semantics"]}
    if p == "send_recv":
        result["per_direction_effective_GBps"] = alg
        result["bidirectional_sum_effective_GBps"] = 2 * alg if case["direction"] == "bidirectional" else None
    return result


def summarize_blocks(case: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        raise ValueError("no accepted timing blocks")
    event, host = [], []
    for block in blocks:
        rows, iterations = block["ranks"], block["iterations"]
        if iterations < 1 or not rows:
            raise ValueError("invalid timing block")
        if sorted(row["rank"] for row in rows) != list(range(case["topology"]["ranks"])):
            raise ValueError("timing block must contain each rank exactly once")
        for row in rows:
            if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("event_seconds", "host_seconds")):
                raise ValueError("non-positive/non-finite timing sample")
        event.append(max(row["event_seconds"] for row in rows) / iterations)
        host.append(max(row["host_seconds"] for row in rows) / iterations)
    median = statistics.median(event)
    return {"device_max_rank_seconds_per_op_by_block": event,
            "host_max_rank_seconds_per_op_by_block": host,
            "median_max_rank_device_seconds_per_op": median,
            "median_max_rank_host_seconds_per_op": statistics.median(host),
            "device_relative_range": max(event) / min(event) - 1.0,
            "single_iteration_percentiles_available": False,
            "device_bandwidth": bandwidth_metrics(case, median),
            "host_completion_bandwidth": bandwidth_metrics(case, statistics.median(host))}
