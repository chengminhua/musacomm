"""Stdlib schema and consistency validation for immutable *planned* cases.

Execution status is intentionally a different document, managed by state.py.
Neither this schema nor a generated row certifies installed hardware capability.
The schema version is declared in catalog.SCHEMA_VERSION; a row contains all
SPEC_FIELDS plus stable identity, validity, capability, and extension metadata.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from . import catalog


HASH = re.compile(r"^[0-9a-f]{64}$")
STRING_FIELDS = {
    "family", "primitive", "backend", "environment_id", "dtype", "buffer_mode",
    "metric", "measurement_variant",
}
NULLABLE_STRING_FIELDS = {"reduce_op", "layout_mode", "direction"}
GENERATED_FIELDS = {
    "schema_version", "plan_version", "case_id", "case_hash", "config_id",
    "run_group_id", "status", "semantic_validity", "invalid_reasons",
    "capability", "platform_status", "official_status", "extension_policy",
}


def spec_from_case(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the immutable identity-bearing fields only."""
    return {key: copy.deepcopy(record[key]) for key in catalog.SPEC_FIELDS}


def _shape_errors(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return ["case must be an object"]
    missing = set(catalog.SPEC_FIELDS) | GENERATED_FIELDS
    missing -= record.keys()
    if missing:
        return [f"missing fields: {','.join(sorted(missing))}"]
    errors = []
    for key in STRING_FIELDS:
        if not isinstance(record[key], str) or not record[key]:
            errors.append(f"{key} must be a non-empty string")
    for key in NULLABLE_STRING_FIELDS:
        if record[key] is not None and not isinstance(record[key], str):
            errors.append(f"{key} must be a string or null")
    for key in ("topology", "size", "capability", "extension_policy"):
        if not isinstance(record[key], dict):
            errors.append(f"{key} must be an object")
    for key in ("route", "moe"):
        if record[key] is not None and not isinstance(record[key], dict):
            errors.append(f"{key} must be an object or null")
    if not isinstance(record["physical_mapping"], list):
        errors.append("physical_mapping must be an array")
    if record["participants"] is not None and not isinstance(record["participants"], list):
        errors.append("participants must be an array or null")
    if record["root"] is not None and type(record["root"]) is not int:
        errors.append("root must be an integer or null")
    if not isinstance(record["invalid_reasons"], list) or any(not isinstance(reason, str) for reason in record["invalid_reasons"]):
        errors.append("invalid_reasons must be a string array")
    if not isinstance(record["case_hash"], str) or not HASH.fullmatch(record["case_hash"]):
        errors.append("case_hash must be a full SHA-256 hex digest")
    return errors


def _derived_size_errors(record: dict[str, Any]) -> list[str]:
    """Audit derived byte counts without including audit data in case identity."""
    errors = []
    primitive, size, n = record["primitive"], record["size"], record["topology"]["ranks"]
    actual = size["actual"]
    expected: dict[str, Any] = {}
    if record["family"] == "moe":
        h, k, d = record["moe"]["hidden"], record["moe"]["top_k"], size["element_bytes"]
        expected.update(input_bytes_per_rank=actual * h * d,
                        logical_expert_expanded_bytes_per_rank=actual * h * k * d,
                        semantics="source_tokens_per_rank", busbw_factor=None)
    elif primitive == "all_to_all_v":
        weights = record["route"]["weight_matrix"]
        if weights is None:
            expected.update(actual=None, materialization_status="unresolved", busbw_factor=None)
        else:
            unit = size.get("weight_unit_count")
            if type(unit) is not int or unit < 0:
                return ["weight_unit_count must be a non-negative integer"]
            byte_unit = unit * size["element_bytes"]
            send = [sum(row) * byte_unit for row in weights]
            recv = [sum(weights[src][dst] for src in range(n)) * byte_unit for dst in range(n)]
            expected.update(actual=max(send + recv), input_bytes_by_rank=send, output_bytes_by_rank=recv,
                            global_nonself_payload_bytes=sum(weights[i][j] for i in range(n) for j in range(n) if i != j) * byte_unit,
                            cross_node_payload_bytes=sum(weights[i][j] for i in range(n) for j in range(n) if i // 8 != j // 8) * byte_unit,
                            bandwidth_applicable=any(send), busbw_factor=None)
    else:
        src_bytes = [actual] * n
        dst_bytes = [actual] * n
        root = record["root"]
        if primitive == "all_gather":
            src_bytes = [actual // n] * n
        elif primitive == "reduce_scatter":
            dst_bytes = [actual // n] * n
        elif primitive == "broadcast":
            src_bytes = [actual if rank == root else 0 for rank in range(n)]
        elif primitive == "reduce":
            dst_bytes = [actual if rank == root else 0 for rank in range(n)]
        elif primitive == "gather":
            src_bytes = [actual // n] * n
            dst_bytes = [actual if rank == root else 0 for rank in range(n)]
        elif primitive == "scatter":
            src_bytes = [actual if rank == root else 0 for rank in range(n)]
            dst_bytes = [actual // n] * n
        elif primitive == "send_recv":
            src, dst = record["participants"]
            senders = {src, dst} if record["direction"] == "bidirectional" else {src}
            receivers = {src, dst} if record["direction"] == "bidirectional" else {dst}
            src_bytes = [actual if rank in senders else 0 for rank in range(n)]
            dst_bytes = [actual if rank in receivers else 0 for rank in range(n)]
        expected.update(input_bytes_by_rank=src_bytes, output_bytes_by_rank=dst_bytes,
                        per_peer_bytes=actual // n if primitive in catalog.DIVIDED else None,
                        root_total_bytes=actual if primitive in catalog.ROOTED else None,
                        busbw_factor=None if record["direction"] == "bidirectional" else catalog.busbw_factor(primitive, n))
    for key, value in expected.items():
        if size.get(key) != value:
            errors.append(f"size.{key} disagrees with operation semantics")
    return errors


def validate_case(record: Any, *, check_identity: bool = True) -> list[str]:
    """Return schema errors; a correctly labelled invalid combination is valid.

    `invalid_combination` records preserve rejected parameter combinations for
    audit, but must not become executable. Actual execution statuses live in a
    separate StateStore and never mutate the catalog's planned capability.
    """
    errors = _shape_errors(record)
    if errors:
        return errors
    if record["schema_version"] != catalog.SCHEMA_VERSION:
        errors.append("schema_version does not match this validator")
    if record["plan_version"] != catalog.PLAN_VERSION:
        errors.append("plan_version does not match this catalog")
    spec = spec_from_case(record)
    try:
        invalid = catalog.validate_spec(spec)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return errors + [f"malformed nested spec: {type(exc).__name__}"]
    expected_status = "invalid_combination" if invalid else "planned"
    if record["status"] != expected_status:
        errors.append("immutable catalog status disagrees with semantic validity")
    if record["semantic_validity"] != ("invalid_combination" if invalid else "valid"):
        errors.append("semantic_validity disagrees with spec validation")
    if record["invalid_reasons"] != invalid:
        errors.append("invalid_reasons does not preserve the validator's reasons")
    capability = record["capability"]
    if capability.get("status") != "unverified" or capability.get("evidence") != []:
        errors.append("planned catalog capability must remain unverified without evidence")
    checks = capability.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(check, str) for check in checks):
        errors.append("capability checks must be a non-empty string array")
    if record["official_status"] != "reference_unverified":
        errors.append("a plan must not certify an official bandwidth comparison")
    zero_payload = record["size"].get("actual") == 0
    if record["platform_status"] != ("not_applicable" if zero_payload else "not_reached"):
        errors.append("a plan must not certify a measured bandwidth platform")
    policy = record["extension_policy"]
    if policy.get("required") is not (not zero_payload) or policy.get("capacity_must_be_verified") is not True:
        errors.append("adaptive extension must retain its measured capacity gate")
    if policy.get("fixed_base_sizes_must_not_be_skipped") is not True:
        errors.append("fixed base sizes must not be skipped")
    if not invalid:
        try:
            errors.extend(_derived_size_errors(record))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"malformed derived size: {type(exc).__name__}")
    if check_identity:
        try:
            rebuilt = catalog.build_case(spec)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"cannot reconstruct immutable identity: {type(exc).__name__}")
        else:
            for key in ("case_hash", "case_id", "config_id", "run_group_id"):
                if record[key] != rebuilt[key]:
                    errors.append(f"{key} does not match the canonical identity")
    return errors


def assert_valid_case(record: Any, *, check_identity: bool = True) -> None:
    errors = validate_case(record, check_identity=check_identity)
    if errors:
        raise ValueError("; ".join(errors))


def alltoallv_counts(record: dict[str, Any]) -> dict[str, Any]:
    """Materialize matching element counts/displacements from saved weights.

    This creates CPU integer arrays only. It is not a backend capability test,
    pack/unpack implementation, memory allocator, or executable job request.
    """
    assert_valid_case(record)
    if record["primitive"] != "all_to_all_v" or record["status"] != "planned":
        raise ValueError("requires a semantically valid AllToAllv planned case")
    weights = record["route"]["weight_matrix"]
    if weights is None:
        raise ValueError("message-hot pack contract is unresolved")
    n, units = record["topology"]["ranks"], record["size"]["weight_unit_count"]
    send = [[weight * units for weight in row] for row in weights]
    recv = [[send[src][dst] for src in range(n)] for dst in range(n)]

    def displacements(counts: list[list[int]]) -> list[list[int]]:
        result = []
        for row in counts:
            position, offsets = 0, []
            for count in row:
                offsets.append(position)
                position += count
            result.append(offsets)
        return result

    return {"units": "elements", "sendcounts": send, "recvcounts": recv,
            "senddispls": displacements(send), "recvdispls": displacements(recv),
            "capability_status": "unverified"}
