#!/usr/bin/env python3
"""Deterministic CPU-only plans for MUSA communication benchmarks.

The public v0.1 launcher supports a single node with exactly eight GPUs. The
AB16 and MoE plans retained here are schema fixtures, not advertised executable
hardware/backend support.

This module does not connect to hosts, discover hardware, allocate GPUs, or
access the resource-manager database. A planned case is never evidence that an
installed backend supports it. Hardware/capability binding precedes execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "musacomm.case.v1"
PLAN_VERSION = "musacomm.catalog.v1"
DTYPES = {"bf16": 2, "fp16": 2, "fp32": 4}
REDUCE_OPS = ("sum", "min", "max", "prod", "avg")
REDUCTION = {"all_reduce", "reduce_scatter", "reduce"}
ROOTED = {"broadcast", "reduce", "gather", "scatter"}
DIVIDED = {"all_gather", "reduce_scatter", "all_to_all", "gather", "scatter"}
INPLACE_CANDIDATES = {
    "all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce", "gather", "scatter"
}
PRIMITIVES = (
    "all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce",
    "all_to_all", "all_to_all_v", "gather", "scatter",
)
CORE_PRIMITIVES = PRIMITIVES + ("send_recv",)
SPEC_FIELDS = (
    "family", "primitive", "backend", "environment_id", "topology", "physical_mapping",
    "dtype", "reduce_op", "root", "buffer_mode", "layout_mode", "metric",
    "measurement_variant", "moe", "route", "participants", "direction", "size",
)
MOE_COMMON_ROUTES = ("balanced", "uniform", "expert_hot_rank_balanced", "rank_hot")
MOE_REMOTE_ROUTES = ("node_local", "node_remote", "node_mixed")
BASE_SIZES = tuple(2**power for power in range(3, 31))
BASE_TOKENS = tuple(2**power for power in range(16))

SIZE_SEMANTICS = {
    "all_reduce": "per_rank_input_and_output_bytes",
    "all_gather": "per_rank_total_receive_bytes",
    "reduce_scatter": "per_rank_total_input_bytes",
    "broadcast": "root_single_message_bytes",
    "reduce": "per_rank_input_or_root_result_bytes",
    "all_to_all": "per_rank_total_send_including_self_bytes",
    "gather": "root_total_receive_bytes",
    "scatter": "root_total_send_bytes",
    "send_recv": "bytes_per_direction",
    "all_to_all_v": "Q_max_rank_total_send_or_receive_bytes",
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def topology(topology_id: str) -> dict[str, Any]:
    if topology_id not in {"A8", "AB16"}:
        raise ValueError("core topology must be A8 or AB16")
    return {"id": topology_id, "nodes": 1 if topology_id == "A8" else 2,
            "ranks": 8 if topology_id == "A8" else 16, "ranks_per_node": 8}


def normalize_mapping(topo: dict[str, Any], supplied: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    n = topo["ranks"]
    if supplied is None:
        return [{"rank": rank, "node": "A" if rank < 8 else "B", "physical_gpu": None, "host": None}
                for rank in range(n)]
    if len(supplied) != n:
        raise ValueError(f"mapping for {topo['id']} needs {n} rank entries")
    result = sorted(copy.deepcopy(supplied), key=lambda item: item["rank"])
    if [item["rank"] for item in result] != list(range(n)):
        raise ValueError("mapping ranks must be unique and contiguous")
    seen = set()
    for item in result:
        expected_node = "A" if item["rank"] < 8 else "B"
        if item.get("node") != expected_node:
            raise ValueError("core rank/node partition must remain A:0..7, B:8..15")
        gpu = item.get("physical_gpu")
        host = item.get("host")
        if gpu is not None and (not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0):
            raise ValueError("physical_gpu must be a non-negative integer or null")
        if host is not None and (not isinstance(host, str) or not host):
            raise ValueError("host must be a non-empty string or null")
        item["physical_gpu"], item["host"] = gpu, host
        if gpu is not None:
            key = (item["node"], gpu)
            if key in seen:
                raise ValueError("two ranks map to the same node/physical GPU")
            seen.add(key)
    return result


def new_spec(family: str, primitive: str, topology_id: str, dtype: str,
             *, mappings: dict[str, list[dict[str, Any]]] | None = None,
             environment_id: str = "UNVERIFIED") -> dict[str, Any]:
    topo = topology(topology_id)
    return {
        "family": family, "primitive": primitive,
        "backend": "mt_deepep_normal" if family == "moe" else "mccl_native",
        "environment_id": environment_id, "topology": topo,
        "physical_mapping": normalize_mapping(topo, (mappings or {}).get(topology_id)),
        "dtype": dtype, "reduce_op": None, "root": None,
        "buffer_mode": "out_of_place", "layout_mode": None,
        "metric": "collective", "measurement_variant": "standalone_continuous",
        "moe": None, "route": None, "participants": None, "direction": None,
    }


def busbw_factor(primitive: str, n: int) -> float | None:
    if primitive == "all_reduce":
        return 2 * (n - 1) / n
    if primitive in {"all_gather", "reduce_scatter", "all_to_all", "gather", "scatter"}:
        return (n - 1) / n
    if primitive in {"broadcast", "reduce", "send_recv"}:
        return 1.0
    return None


def fixed_sizes(primitive: str, n: int, dtype: str) -> Iterator[dict[str, Any]]:
    """Deduplicate actual legal S while retaining every requested size alias.

    1/2/3 elements mean the API's per-block count for divided primitives.
    Rounding is explicitly upward, never silently downward to a zero count.
    """
    d = 1 if dtype == "uint8_bytes" else DTYPES[dtype]
    quantum = d * (n if primitive in DIVIDED else 1)
    by_actual: dict[int, list[dict[str, Any]]] = {}
    for count in (1, 2, 3):
        by_actual.setdefault(count * quantum, []).append({"kind": "api_count", "requested": count})
    for requested in BASE_SIZES:
        actual = ((requested + quantum - 1) // quantum) * quantum
        by_actual.setdefault(actual, []).append({"kind": "S_bytes", "requested": requested})
    for actual, requests in sorted(by_actual.items()):
        count = actual // quantum
        input_bytes: list[int]
        output_bytes: list[int]
        if primitive == "all_gather":
            input_bytes, output_bytes = [actual // n] * n, [actual] * n
        elif primitive == "reduce_scatter":
            input_bytes, output_bytes = [actual] * n, [actual // n] * n
        else:
            # Rooted and P2P ownership is finalized by fixed_size_for_spec.
            input_bytes, output_bytes = [actual] * n, [actual] * n
        yield {
            "kind": "S_bytes", "requested": min(item["requested"] for item in requests
                   if item["kind"] == "S_bytes") if any(item["kind"] == "S_bytes" for item in requests)
                   else actual,
            "actual": actual, "unit": "bytes", "requests": requests,
            "rounding": "ceil_to_legal_quantum", "legal_quantum_bytes": quantum,
            "api_count": count, "element_bytes": d,
            "semantics": SIZE_SEMANTICS[primitive],
            "input_bytes_by_rank": input_bytes, "output_bytes_by_rank": output_bytes,
            "per_peer_bytes": actual // n if primitive in DIVIDED else None,
            "root_total_bytes": actual if primitive in ROOTED else None,
            "busbw_factor": busbw_factor(primitive, n), "capacity_status": "unverified",
        }


def fixed_size_for_spec(size: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    size = copy.deepcopy(size)
    primitive, n, root, actual = spec["primitive"], spec["topology"]["ranks"], spec["root"], size["actual"]
    if primitive == "broadcast":
        size["input_bytes_by_rank"] = [actual if rank == root else 0 for rank in range(n)]
    elif primitive == "reduce":
        size["output_bytes_by_rank"] = [actual if rank == root else 0 for rank in range(n)]
    elif primitive == "gather":
        size["input_bytes_by_rank"] = [actual // n] * n
        size["output_bytes_by_rank"] = [actual if rank == root else 0 for rank in range(n)]
    elif primitive == "scatter":
        size["input_bytes_by_rank"] = [actual if rank == root else 0 for rank in range(n)]
        size["output_bytes_by_rank"] = [actual // n] * n
    elif primitive == "send_recv":
        src, dst = spec["participants"]
        senders = {src, dst} if spec["direction"] == "bidirectional" else {src}
        receivers = {src, dst} if spec["direction"] == "bidirectional" else {dst}
        size["input_bytes_by_rank"] = [actual if rank in senders else 0 for rank in range(n)]
        size["output_bytes_by_rank"] = [actual if rank in receivers else 0 for rank in range(n)]
        # Never label a bidirectional summed rate as one-direction busbw.
        if spec["direction"] == "bidirectional":
            size["busbw_factor"] = None
    return size


def _selected(n: int, count: int, template: str) -> list[int]:
    return list(range(count)) if template == "contiguous" else [(index * n) // count for index in range(count)]


def variable_routes(n: int, seed: int = 20261014) -> Iterator[dict[str, Any]]:
    """Finite, deterministic AllToAllv route-family manifest for the catalog schema."""
    def emit(route_id: str, weights: list[list[int]], **details: Any) -> dict[str, Any]:
        return {"id": route_id, "seed": seed, "weight_matrix": weights, **details}

    yield emit("equal", [[1] * n for _ in range(n)])
    rng = random.Random(seed + n)
    yield emit("random_unequal", [[rng.randint(1, 8) for _ in range(n)] for _ in range(n)])
    for degree in sorted({1, 2, min(4, n - 1)}):
        for template in ("ring", "seeded_permutation"):
            weights = [[0] * n for _ in range(n)]
            for src in range(n):
                dests = [(src + step) % n for step in range(1, n)]
                if template == "seeded_permutation":
                    random.Random(seed + n * 100 + src).shuffle(dests)
                for dst in dests[:degree]:
                    weights[src][dst] = 1
            yield emit(f"sparse_{degree}_{template}", weights, family="sparse", degree=degree, template=template)
    for side in ("zero_send", "zero_receive"):
        for fraction in (0.25, 0.5):
            for template in ("contiguous", "dispersed"):
                inactive = _selected(n, int(n * fraction), template)
                weights = [[int((src not in inactive) if side == "zero_send" else (dst not in inactive))
                            for dst in range(n)] for src in range(n)]
                yield emit(f"{side}_{int(fraction * 100)}_{template}", weights,
                           family=side, fraction=fraction, template=template, inactive_ranks=inactive)
    for family in ("incast", "outcast"):
        for count in sorted({1, n // 4}):
            for template in ("contiguous", "dispersed"):
                selected = _selected(n, count, template)
                weights = [[int((dst in selected) if family == "incast" else (src in selected))
                            for dst in range(n)] for src in range(n)]
                yield emit(f"{family}_{count}_{template}", weights, family=family,
                           count=count, template=template, selected_ranks=selected)
    # A peer-level AllToAllv cannot expose an intra-peer message/expert split.
    # Keep this explicit requested family pending materialization, not a fake
    # duplicate uniform benchmark claiming to exercise expert-hot routing.
    yield {"id": "message_hot_rank_balanced", "seed": seed, "weight_matrix": None,
           "materialization_status": "needs_message_or_expert_pack_contract",
           "hot_selection_target": 0.8, "hot_destination_fraction": 0.25}
    hot = set(range(n // 4))
    yield emit("rank_hot", [[12 if dst in hot else 1 for dst in range(n)] for _ in range(n)],
               hot_selection_target=0.8, hot_destination_fraction=0.25)
    if n == 16:
        for percent in (0, 25, 50, 75, 100):
            remote, local = percent // 25, 4 - percent // 25
            weights = [[remote if src // 8 != dst // 8 else local for dst in range(n)] for src in range(n)]
            yield emit(f"node_fraction_{percent}", weights, cross_node_payload_target=percent / 100)
    yield emit("all_zero", [[0] * n for _ in range(n)], zero_payload=True)


def variable_sizes(route: dict[str, Any], n: int, dtype: str) -> Iterator[dict[str, Any]]:
    d, weights = DTYPES[dtype], route["weight_matrix"]
    if weights is None:
        for requested in sorted(set((d, 2 * d, 3 * d) + BASE_SIZES)):
            yield {"kind": "Q_bytes", "requested": requested, "actual": None, "unit": "bytes",
                   "semantics": SIZE_SEMANTICS["all_to_all_v"], "element_bytes": d,
                   "weight_unit_count": None, "capacity_status": "unverified",
                   "materialization_status": "unresolved", "busbw_factor": None}
        return
    send_units = [sum(row) for row in weights]
    receive_units = [sum(weights[src][dst] for src in range(n)) for dst in range(n)]
    q_quantum = max(send_units + receive_units) * d
    if q_quantum == 0:
        yield {"kind": "Q_bytes", "requested": 0, "actual": 0, "unit": "bytes",
               "semantics": SIZE_SEMANTICS["all_to_all_v"], "element_bytes": d,
               "weight_unit_count": 0, "input_bytes_by_rank": [0] * n,
               "output_bytes_by_rank": [0] * n, "global_nonself_payload_bytes": 0,
               "cross_node_payload_bytes": 0, "capacity_status": "unverified", "busbw_factor": None,
               "bandwidth_applicable": False}
        return
    by_actual: dict[int, list[dict[str, Any]]] = {}
    for unit_count in (1, 2, 3):
        by_actual.setdefault(unit_count * q_quantum, []).append({"kind": "weight_unit_count", "requested": unit_count})
    for requested in BASE_SIZES:
        actual = ((requested + q_quantum - 1) // q_quantum) * q_quantum
        by_actual.setdefault(actual, []).append({"kind": "Q_bytes", "requested": requested})
    for actual, requests in sorted(by_actual.items()):
        unit_count = actual // q_quantum
        edge_byte_unit = unit_count * d
        yield {
            "kind": "Q_bytes", "requested": min(item["requested"] for item in requests
                   if item["kind"] == "Q_bytes") if any(item["kind"] == "Q_bytes" for item in requests)
                   else actual,
            "actual": actual, "unit": "bytes", "requests": requests,
            "semantics": SIZE_SEMANTICS["all_to_all_v"], "element_bytes": d,
            "legal_quantum_bytes": q_quantum, "rounding": "ceil_to_fixed_weight_quantum",
            "weight_unit_count": unit_count,
            "input_bytes_by_rank": [units * edge_byte_unit for units in send_units],
            "output_bytes_by_rank": [units * edge_byte_unit for units in receive_units],
            "global_nonself_payload_bytes": sum(weights[i][j] for i in range(n) for j in range(n) if i != j) * edge_byte_unit,
            "cross_node_payload_bytes": sum(weights[i][j] for i in range(n) for j in range(n)
                                            if i // 8 != j // 8) * edge_byte_unit,
            "capacity_status": "unverified", "busbw_factor": None,
            "bandwidth_applicable": True,
        }


def validate_spec(spec: dict[str, Any]) -> list[str]:
    """Check semantics independently from installed-backend capability.

    This accepts the core catalog vocabulary only. A valid result never means
    that a dtype/op, aliasing mode, backend, or host is actually supported.
    Shape/schema errors are returned rather than silently becoming executable.
    """
    errors = []
    missing = [key for key in SPEC_FIELDS if key not in spec]
    if missing:
        return [f"missing spec fields: {','.join(missing)}"]
    topo = spec["topology"]
    if not isinstance(topo, dict) or topo.get("id") not in {"A8", "AB16"}:
        return ["topology must be a core A8/AB16 object"]
    if topo != topology(topo["id"]):
        errors.append("topology counts must agree with core topology id")
    n = topology(topo["id"])["ranks"]
    family, primitive, dtype = spec["family"], spec["primitive"], spec["dtype"]
    if family not in {"collective", "p2p", "moe"}:
        errors.append("family must be collective, p2p, or moe")
    elif (family == "collective" and primitive not in PRIMITIVES
          or family == "p2p" and primitive != "send_recv"
          or family == "moe" and primitive != "moe"):
        errors.append("primitive does not belong to the selected family")
    if dtype not in DTYPES and not (family == "p2p" and dtype == "uint8_bytes"):
        errors.append("dtype is outside the core family vocabulary")
    try:
        normalized = normalize_mapping(topology(topo["id"]), spec["physical_mapping"])
        if spec["physical_mapping"] != normalized:
            errors.append("physical mapping must be present and rank ordered")
    except (KeyError, TypeError, ValueError):
        errors.append("physical mapping is invalid")
    if not isinstance(spec["backend"], str) or not spec["backend"]:
        errors.append("backend must be a non-empty identifier")
    if not isinstance(spec["environment_id"], str) or not spec["environment_id"]:
        errors.append("environment_id must be a non-empty identifier")
    if primitive in ROOTED and (type(spec["root"]) is not int or not 0 <= spec["root"] < n):
        errors.append("root must be in communicator ranks")
    if primitive not in ROOTED and spec["root"] is not None:
        errors.append("root supplied to a non-rooted operation")
    if primitive in REDUCTION and spec["reduce_op"] not in REDUCE_OPS:
        errors.append("reduction operation missing or outside the supported reduction vocabulary")
    if primitive not in REDUCTION and spec["reduce_op"] is not None:
        errors.append("reduce_op supplied to a non-reduction operation")
    if spec["buffer_mode"] not in {"out_of_place", "in_place"}:
        errors.append("buffer mode must be out_of_place or in_place")
    if spec["buffer_mode"] == "in_place" and primitive not in INPLACE_CANDIDATES:
        errors.append("core in_place is not a candidate for this primitive")
    size = spec["size"]
    if not isinstance(size, dict):
        return errors + ["size must be an object"]
    if type(size.get("requested")) is not int or size["requested"] < 0:
        errors.append("requested size must be a non-negative integer")
    if size.get("actual") is not None and (type(size["actual"]) is not int or size["actual"] < 0):
        errors.append("actual size must be a non-negative integer or unresolved null")
    expected_element_bytes = 1 if dtype == "uint8_bytes" else DTYPES.get(dtype)
    if expected_element_bytes is not None and size.get("element_bytes") != expected_element_bytes:
        errors.append("element bytes must agree with dtype")
    for field in ("input_bytes_by_rank", "output_bytes_by_rank"):
        if field in size and (not isinstance(size[field], list) or len(size[field]) != n
                              or any(type(value) is not int or value < 0 for value in size[field])):
            errors.append(f"{field} must contain one non-negative byte count per rank")
    if family == "moe":
        moe = spec["moe"]
        valid_moe = (isinstance(moe, dict)
                     and all(type(moe.get(key)) is int and moe[key] > 0 for key in ("hidden", "top_k", "experts")))
        if not valid_moe:
            errors.append("MoE hidden, Top-K, and experts must be positive integers")
        elif not 1 <= moe["top_k"] <= moe["experts"]:
            errors.append("Top-K must select distinct experts within E")
        if valid_moe and moe["experts"] % n:
            errors.append("uniform expert placement requires E divisible by EP")
        route_id = spec["route"].get("id") if isinstance(spec["route"], dict) else None
        allowed_routes = MOE_COMMON_ROUTES + (MOE_REMOTE_ROUTES if topo["id"] == "AB16" else ())
        if route_id not in allowed_routes:
            errors.append("MoE route is outside this topology's core route set")
        if valid_moe and route_id in {"node_local", "node_remote"} and moe["top_k"] > moe["experts"] // 2:
            errors.append("Top-K exceeds eligible experts on one node")
        if type(size.get("actual")) is not int or size["actual"] <= 0:
            errors.append("MoE core requires a positive token count")
        if size.get("kind") != "tokens_per_rank" or size.get("unit") != "tokens":
            errors.append("MoE size must be source tokens per rank")
        if spec["layout_mode"] not in {"fresh", "cached"}:
            errors.append("MoE layout must be fresh or cached")
        if spec["metric"] not in {"dispatch", "combine", "roundtrip"}:
            errors.append("MoE metric must be dispatch, combine, or roundtrip")
    elif spec["layout_mode"] is not None or spec["moe"] is not None:
        errors.append("MoE-only fields supplied to a non-MoE case")
    if family == "p2p":
        participants = spec["participants"]
        if (not isinstance(participants, list) or len(participants) != 2
                or any(type(p) is not int or not 0 <= p < n for p in participants)
                or participants[0] == participants[1]):
            errors.append("P2P requires two distinct in-range ranks")
        if spec["direction"] not in {"unidirectional", "bidirectional"}:
            errors.append("P2P direction is invalid")
    elif spec["participants"] is not None or spec["direction"] is not None:
        errors.append("P2P-only fields supplied to a non-P2P case")
    if family != "moe" and primitive != "all_to_all_v":
        if size.get("kind") != "S_bytes" or size.get("unit") != "bytes":
            errors.append("fixed communication size must be S bytes")
        quantum = expected_element_bytes * (n if primitive in DIVIDED else 1) if expected_element_bytes else None
        if (quantum is not None and (size.get("legal_quantum_bytes") != quantum
                or type(size.get("api_count")) is not int or size["api_count"] < 0
                or size.get("actual") != size.get("api_count", -1) * quantum)):
            errors.append("fixed size must equal legal API count times its quantum")
    if primitive == "all_to_all_v":
        route = spec["route"]
        weights = route.get("weight_matrix") if isinstance(route, dict) else None
        if not isinstance(route, dict) or "weight_matrix" not in route:
            errors.append("AllToAllv requires an explicit route matrix or unresolved contract")
        elif weights is not None:
            if (not isinstance(weights, list) or len(weights) != n
                    or any(not isinstance(row, list) or len(row) != n
                           or any(type(value) is not int or value < 0 for value in row) for row in weights)):
                errors.append("AllToAllv weights must be a non-negative NxN integer matrix")
        if size.get("kind") != "Q_bytes" or size.get("unit") != "bytes":
            errors.append("AllToAllv size must be Q bytes")
    return errors


def _identity_spec(spec: dict[str, Any], *, with_size: bool = True) -> dict[str, Any]:
    result = copy.deepcopy(spec)
    if not with_size:
        result.pop("size", None)
    else:
        size = result["size"]
        # Requested aliases and audit-only calculated metadata do not split
        # physically identical sizes. Unknown actual sizes retain the request.
        result["size"] = {key: value for key, value in size.items()
                          if key in {"kind", "actual", "api_count", "element_bytes", "semantics", "weight_unit_count"}}
        if size.get("actual") is None:
            result["size"]["requested"] = size["requested"]
    return result


def build_case(spec: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    errors = validate_spec(spec)
    case_hash = digest({"schema_version": SCHEMA_VERSION, "spec": _identity_spec(spec)})
    config_hash = digest({"schema_version": SCHEMA_VERSION, "spec": _identity_spec(spec, with_size=False)})
    checks = ["installed_backend_version", "topology", "dtype", "shape", "capacity", "mapping_and_environment"]
    if spec["reduce_op"]:
        checks.append("reduction_op_dtype_combination")
    if spec["buffer_mode"] == "in_place":
        checks.append("native_in_place_alias_contract")
    if spec["layout_mode"] == "cached":
        checks.append("cached_layout_contract")
    if spec.get("route") and spec["route"].get("weight_matrix", True) is None:
        checks.append("message_or_expert_pack_contract")
    if spec["family"] == "moe":
        checks.extend(["route_materialization", "dispatch_contract", "combine_contract"])
    run_group = copy.deepcopy(spec)
    if spec["family"] == "moe":
        run_group["metric"] = "all_three_metrics"
        run_group["measurement_variant"] = "shared_preparation_only"
    zero_payload = spec["size"].get("actual") == 0
    return {
        "schema_version": SCHEMA_VERSION, "plan_version": PLAN_VERSION,
        "case_id": f"{spec['family']}-{case_hash[:20]}", "case_hash": case_hash,
        "config_id": f"config-{config_hash[:20]}",
        "run_group_id": f"run-{digest(_identity_spec(run_group))[:20]}",
        **spec, "status": "invalid_combination" if errors else "planned",
        "semantic_validity": "invalid_combination" if errors else "valid",
        "invalid_reasons": errors,
        "capability": {"status": "unverified", "checks": checks, "evidence": []},
        "platform_status": "not_applicable" if zero_payload else "not_reached",
        "official_status": "reference_unverified",
        "extension_policy": {
            "required": not zero_payload, "kind": "adaptive_doubling_then_confirmation",
            "next_requested": (65536 if spec["family"] == "moe" else 2**31),
            "preallocation_budget_fraction": 0.60, "capacity_must_be_verified": True,
            "fixed_base_sizes_must_not_be_skipped": True,
        },
    }


def collective_cases(*, mappings: dict[str, list[dict[str, Any]]] | None = None,
                     environment_id: str = "UNVERIFIED", seed: int = 20261014) -> Iterator[dict[str, Any]]:
    for topo_id in ("A8", "AB16"):
        n = topology(topo_id)["ranks"]
        for primitive in PRIMITIVES:
            for dtype in DTYPES:
                ops = REDUCE_OPS if primitive in REDUCTION else (None,)
                roots = range(n) if primitive in ROOTED else (None,)
                modes = ("out_of_place", "in_place") if primitive in INPLACE_CANDIDATES else ("out_of_place",)
                routes = list(variable_routes(n, seed)) if primitive == "all_to_all_v" else [None]
                for op in ops:
                    for root in roots:
                        for mode in modes:
                            for route in routes:
                                spec = new_spec("collective", primitive, topo_id, dtype,
                                                mappings=mappings, environment_id=environment_id)
                                spec.update(reduce_op=op, root=root, buffer_mode=mode, route=route)
                                sizes = variable_sizes(route, n, dtype) if route else fixed_sizes(primitive, n, dtype)
                                for size in sizes:
                                    spec["size"] = size if route else fixed_size_for_spec(size, spec)
                                    yield build_case(spec)


def moe_cases(*, mappings: dict[str, list[dict[str, Any]]] | None = None,
              environment_id: str = "UNVERIFIED", seed: int = 20261014) -> Iterator[dict[str, Any]]:
    for topo_id in ("A8", "AB16"):
        routes = MOE_COMMON_ROUTES + (MOE_REMOTE_ROUTES if topo_id == "AB16" else ())
        for hidden in (2048, 4096, 7168, 8192):
            for top_k in (1, 2, 4, 8):
                for experts in (64, 128, 256):
                    for route in routes:
                        for tokens in BASE_TOKENS:
                            for layout in ("fresh", "cached"):
                                for metric in ("dispatch", "combine", "roundtrip"):
                                    spec = new_spec("moe", "moe", topo_id, "bf16",
                                                    mappings=mappings, environment_id=environment_id)
                                    spec.update(layout_mode=layout, metric=metric,
                                                measurement_variant="direct_roundtrip" if metric == "roundtrip" else "standalone_continuous",
                                                moe={"hidden": hidden, "top_k": top_k, "experts": experts,
                                                     "dispatch_dtype": "bf16", "combine_dtype": "bf16"},
                                                route={"id": route, "seed": seed, "materialization_status": "unresolved",
                                                       "expert_placement": "uniform_contiguous_by_rank",
                                                       "rule_version": "v2-prefix-stable-constraints"})
                                    spec["size"] = {
                                        "kind": "tokens_per_rank", "requested": tokens, "actual": tokens, "unit": "tokens",
                                        "semantics": "source_tokens_per_rank", "element_bytes": 2,
                                        "input_bytes_per_rank": tokens * hidden * 2,
                                        "logical_expert_expanded_bytes_per_rank": tokens * hidden * top_k * 2,
                                        "actual_route_payload_bytes": None, "actual_peak_memory_bytes": None,
                                        "capacity_status": "unverified", "busbw_factor": None,
                                    }
                                    yield build_case(spec)


def p2p_cases(*, mappings: dict[str, list[dict[str, Any]]] | None = None,
              environment_id: str = "UNVERIFIED", **_: Any) -> Iterator[dict[str, Any]]:
    """A intra-node and all cross-node pairs. B intra-node is a separate diagnostic."""
    for topo_id in ("A8", "AB16"):
        n = topology(topo_id)["ranks"]
        for direction in ("unidirectional", "bidirectional"):
            for src in range(n):
                for dst in range(n):
                    if src == dst or (topo_id == "AB16" and src // 8 == dst // 8):
                        continue
                    if direction == "bidirectional" and src > dst:
                        continue
                    for dtype in (*DTYPES, "uint8_bytes"):
                        spec = new_spec("p2p", "send_recv", topo_id, dtype,
                                        mappings=mappings, environment_id=environment_id)
                        spec.update(participants=[src, dst], direction=direction, metric=f"p2p_{direction}")
                        for size in fixed_sizes("send_recv", n, dtype):
                            spec["size"] = fixed_size_for_spec(size, spec)
                            yield build_case(spec)


def append_extension(case: dict[str, Any], size: dict[str, Any]) -> dict[str, Any]:
    """Create a distinct immutable case for a prevalidated dynamic size.

    This is not a memory allocator and never infers available memory. Caller
    supplies corrected counts/bytes and retains the capacity proof separately.
    """
    spec = {key: copy.deepcopy(case[key]) for key in SPEC_FIELDS}
    spec["size"] = copy.deepcopy(size)
    result = build_case(spec)
    result["parent_case_id"] = case["case_id"]
    result["coverage_tier"] = "adaptive_extension"
    return result


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Publish a whole new manifest without overwriting an existing one.

    A duplicate or invalid record leaves no partial target. Link publication is
    atomic on the local filesystem and fails if another writer won the name.
    """
    from .schema import assert_valid_case

    if path.exists():
        raise FileExistsError(f"refusing to overwrite catalog: {path}")
    count, hashes = 0, hashlib.sha256()
    statuses, topologies, operations = Counter(), Counter(), Counter()
    case_ids: set[str] = set()
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                assert_valid_case(record)
                if record["case_id"] in case_ids:
                    raise ValueError(f"duplicate case_id: {record['case_id']}")
                case_ids.add(record["case_id"])
                line = canonical(record) + "\n"
                stream.write(line)
                hashes.update(line.encode("utf-8"))
                count += 1
                statuses[record["status"]] += 1
                topologies[record["topology"]["id"]] += 1
                operations[record["primitive"]] += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(path), "records": count, "sha256": hashes.hexdigest(),
            "statuses": dict(statuses), "topologies": dict(topologies), "primitives": dict(operations)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("musacomm-catalog"))
    parser.add_argument("--families", default="collective,moe,p2p")
    parser.add_argument("--mapping", type=Path, help="JSON object keyed by A8/AB16, not an SSH target")
    parser.add_argument("--environment-id", default="UNVERIFIED")
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    generators = {"collective": collective_cases, "moe": moe_cases, "p2p": p2p_cases}
    families = args.families.split(",")
    if not families or any(family not in generators for family in families) or len(set(families)) != len(families):
        parser.error("families must be unique values from collective,moe,p2p")
    mappings = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for target in [args.out_dir / f"{family}.jsonl" for family in families] + [args.out_dir / "manifest_summary.json"]:
        if target.exists():
            parser.error(f"refusing to overwrite existing output: {target}; choose a new --out-dir")
    summary = {"schema_version": SCHEMA_VERSION, "plan_version": PLAN_VERSION,
               "environment_id": args.environment_id, "manifest_files": {},
               "moe_contract_counts": {"base_combinations": 8448, "layout_configs": 16896, "metric_rows": 50688},
               "not_materialized_by_this_core_catalog": [
                   "adaptive sizes need per-configuration measured capacity and platform feedback",
                   "B intra-node all pairs; 2/4/3/7/15-rank diagnostics; concurrent pair matchings",
                   "boundary/alignment/zero-count extensions outside variable all-zero control",
                   "conditional backends/dtypes; latency samples; soak stress schedules",
                   "MoE actual routes and AllToAllv message-hot packed layout need execution contracts",
               ]}
    for family in families:
        summary["manifest_files"][family] = write_jsonl(
            args.out_dir / f"{family}.jsonl", generators[family](mappings=mappings,
                environment_id=args.environment_id, seed=args.seed))
    summary_path = args.out_dir / "manifest_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
