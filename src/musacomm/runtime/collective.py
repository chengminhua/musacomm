#!/usr/bin/env python3
"""Catalog-driven MUSA communication benchmark with isolated CPU control.

No torch/MUSA import occurs on import, --help, or --validate-only. The caller
must reserve the host and launch one rank per GPU. Supplied reservation evidence
is recorded, not authenticated; this module does not provide resource isolation.
Native MCCL uses torch_musa-owned tensors and an explicit MUSA stream. Gloo
handles IDs, status, readiness, errors and timing aggregation, never MCCL.
"""
from __future__ import annotations

import argparse
import ctypes as C
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback

from . import helpers as helper_module
from .helpers import (
    REDUCTIONS, audit_manifest, catalog_modules, choose_iterations, input_segments,
    iter_cases, make_rank_plan, memory_decision, output_segments, performance_constant,
    summarize_blocks, validate_visible_mapping,
)


class PhaseFailure(RuntimeError):
    def __init__(self, phase, reports):
        self.phase, self.reports = phase, reports
        super().__init__(f"distributed phase failed: {phase}")


class Journal:
    def __init__(self, directory, rank):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"rank-{rank:04d}.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.rank = rank

    def write(self, record):
        record = {"rank": self.rank, "utc": datetime.now(timezone.utc).isoformat(), **record}
        self.file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


class Control:
    def __init__(self, dist, world):
        self.dist, self.world = dist, world

    def gather(self, value):
        values = [None] * self.world
        self.dist.all_gather_object(values, value)  # Default group is Gloo/CPU.
        return values

    def barrier(self):
        self.dist.barrier()  # Never insert a GPU collective into measured work.

    def phase(self, name, function):
        value, error = None, None
        try:
            value = function()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        reports = self.gather({"error": error})
        if any(report["error"] for report in reports):
            raise PhaseFailure(name, reports)
        return value


class UniqueId(C.Structure):
    _fields_ = [("internal", C.c_char * 128)]


class NativeMCCL:
    """ABI checked against the installed header; no compiler or private symbols."""
    DTYPE = {"uint8_bytes": 1, "fp16": 6, "fp32": 7, "bf16": 9}
    OP = {"sum": 0, "prod": 1, "max": 2, "min": 3, "avg": 4}

    def __init__(self, library_path, header_path):
        library_path = Path(library_path).resolve(strict=True)
        header_path = Path(header_path).resolve(strict=True)
        header = header_path.read_text()
        self.header = header
        for name, value in {"MCCL_UNIQUE_ID_BYTES": 128, "mcclUint8": 1,
                            "mcclFloat16": 6, "mcclFloat32": 7, "mcclBfloat16": 9,
                            "mcclSum": 0, "mcclProd": 1, "mcclMax": 2,
                            "mcclMin": 3, "mcclAvg": 4}.items():
            if not re.search(r"\b" + name + r"\s*(?:=\s*)?" + str(value) + r"\b", header):
                raise ValueError(f"installed header ABI check failed: {name}={value}")
        self.lib = C.CDLL(str(library_path))
        self.functions = {}
        ptr, count, integer, stream = C.c_void_p, C.c_size_t, C.c_int, C.c_void_p
        signatures = {
            "mcclGetVersion": [C.POINTER(integer)],
            "mcclGetUniqueId": [C.POINTER(UniqueId)],
            "mcclCommInitRank": [C.POINTER(ptr), integer, UniqueId, integer],
            "mcclCommDestroy": [ptr], "mcclCommAbort": [ptr],
            "mcclCommGetAsyncError": [ptr, C.POINTER(integer)],
            "mcclGroupStart": [], "mcclGroupEnd": [],
            "mcclAllReduce": [ptr, ptr, count, integer, integer, ptr, stream],
            "mcclReduce": [ptr, ptr, count, integer, integer, integer, ptr, stream],
            "mcclBroadcast": [ptr, ptr, count, integer, integer, ptr, stream],
            "mcclAllGather": [ptr, ptr, count, integer, ptr, stream],
            "mcclReduceScatter": [ptr, ptr, count, integer, integer, ptr, stream],
            "mcclAlltoAll": [ptr, ptr, count, integer, ptr, stream],
            "mcclGather": [ptr, ptr, count, integer, integer, ptr, stream],
            "mcclScatter": [ptr, ptr, count, integer, integer, ptr, stream],
            "mcclSend": [ptr, count, integer, integer, ptr, stream],
            "mcclRecv": [ptr, count, integer, integer, ptr, stream],
            "mcclAllToAllv": [ptr, C.POINTER(count), C.POINTER(count), ptr,
                              C.POINTER(count), C.POINTER(count), integer, ptr, stream],
        }
        for name, args in signatures.items():
            if re.search(r"\b" + name + r"\s*\(", header) and hasattr(self.lib, name):
                fn = getattr(self.lib, name)
                fn.argtypes, fn.restype = args, integer
                self.functions[name] = fn
        self.lib.mcclGetErrorString.argtypes = [integer]
        self.lib.mcclGetErrorString.restype = C.c_char_p
        self.comm = ptr()
        self.version = integer()
        self.call("mcclGetVersion", C.byref(self.version))
        header_version = re.search(r"#define\s+MCCL_VERSION_CODE\s+(\d+)", header)
        if not header_version or self.version.value != int(header_version.group(1)):
            raise ValueError("runtime MCCL version differs from supplied installed header")
        self.metadata = {"library": str(library_path), "header": str(header_path),
                         "library_sha256": file_sha256(library_path),
                         "header_sha256": hashlib.sha256(header.encode()).hexdigest(),
                         "version_code": self.version.value,
                         "exposed_declared_functions": sorted(self.functions)}

    def check(self, result, name):
        if result != 0:
            message = self.lib.mcclGetErrorString(result)
            raise RuntimeError(f"{name}: MCCL {result}: {(message or b'unknown').decode(errors='replace')}")

    def call(self, name, *args):
        if name not in self.functions:
            raise RuntimeError(f"installed public API is not exposed: {name}")
        self.check(self.functions[name](*args), name)

    def initialize(self, rank, world, control):
        uid = UniqueId()
        control.phase("unique_id", lambda: self.call("mcclGetUniqueId", C.byref(uid)) if rank == 0 else None)
        ids = [C.string_at(C.byref(uid), 128) if rank == 0 else None]
        control.dist.broadcast_object_list(ids, src=0)
        C.memmove(C.byref(uid), ids[0], 128)
        control.phase("native_communicator_init", lambda: self.call("mcclCommInitRank", C.byref(self.comm), world, uid, rank))

    def check_async(self):
        code = C.c_int()
        self.call("mcclCommGetAsyncError", self.comm, C.byref(code))
        self.check(code.value, "asynchronous communication")

    def close(self, abort=False):
        if self.comm.value:
            self.call("mcclCommAbort" if abort else "mcclCommDestroy", self.comm)
            self.comm = C.c_void_p()


class Buffers:
    def __init__(self, torch, plan, dtype, device):
        self.plan, self.owned = plan, []
        def alloc(n):
            tensor = torch.empty(max(1, n), device=device, dtype=dtype)
            self.owned.append(tensor)
            return tensor
        if plan.alias == "same":
            self.input = self.output = alloc(max(plan.input_elements, plan.output_elements))
        elif plan.alias == "input_in_output":
            self.output = alloc(plan.output_elements)
            self.input = self.output.narrow(0, plan.alias_offset, plan.input_elements)
        elif plan.alias == "output_in_input":
            self.input = alloc(plan.input_elements)
            self.output = self.input.narrow(0, plan.alias_offset, plan.output_elements)
        else:
            self.input, self.output = alloc(plan.input_elements), alloc(plan.output_elements)
        # Empty views are materialized before timing, while native calls keep
        # valid pointers to their one-element dummy storage.
        self.logical_input = self.input.narrow(0, 0, plan.input_elements)
        self.logical_output = self.output.narrow(0, 0, plan.output_elements)


def cpu_pattern(torch, source, destination, start, count):
    index = torch.arange(start, start + count, dtype=torch.int64, device="cpu")
    return (source * 37 + destination * 13 + index * 17 + (index // 251) * 29 + 23).remainder(251)


def cpu_reduction_input(torch, rank, start, count, op, case):
    index = torch.arange(start, start + count, dtype=torch.int64, device="cpu")
    if op in {"prod", "min", "max"}:
        n = case["topology"]["ranks"]
        if case["primitive"] == "reduce_scatter":
            c = case["size"]["api_count"]
            winner = (index // c + 1 + index.remainder(c).remainder(n - 1)).remainder(n)
        elif case["primitive"] == "reduce":
            winner = (case["root"] + 1 + index.remainder(n - 1)).remainder(n)
        else:
            winner = index.remainder(n)
        if op == "prod":
            # The sole magnitude-2 factor belongs to a non-root/non-receiver
            # for rooted/partitioned reductions. No-op outputs remain +/-1.
            sign = torch.where((index + rank).remainder(3) == 0, -1, 1)
            return sign * torch.where(winner == rank, 2, 1)
        delta = (rank - winner).remainder(n)
        return 1 + index.remainder(7) + (delta if op == "min" else n - 1 - delta)
    return 1 + index.remainder(7) + rank % 3


def cpu_reference(torch, plan, case, start, count):
    offset = plan.rank * plan.api_count if plan.primitive == "reduce_scatter" else 0
    op = case["reduce_op"]
    # CPU FP64 reference is independent of communicated values and device kernels.
    values = [cpu_reduction_input(torch, r, offset + start, count, op, case).to(torch.float64)
              for r in range(plan.world)]
    result = values[0]
    for value in values[1:]:
        if op in {"sum", "avg"}:
            result = result + value
        elif op == "prod":
            result = result * value
        elif op == "min":
            result = torch.minimum(result, value)
        elif op == "max":
            result = torch.maximum(result, value)
    return result / plan.world if op == "avg" else result


def initialize_buffers(torch, buffers, case, chunk_elements, timing=False):
    plan = buffers.plan
    constant = performance_constant(case) if timing else None
    for tensor in buffers.owned:
        tensor.fill_(251 if tensor.dtype == torch.uint8 else -19)
    if constant is not None:
        buffers.input.fill_(constant)
    elif plan.primitive in REDUCTIONS:
        for start in range(0, plan.input_elements, chunk_elements):
            length = min(chunk_elements, plan.input_elements - start)
            values = cpu_reduction_input(torch, plan.rank, start, length, case["reduce_op"], case)
            buffers.input.narrow(0, start, length).copy_(values.to(buffers.input.dtype))
    else:
        for offset, count, source, destination in input_segments(case, plan):
            for start in range(0, count, chunk_elements):
                length = min(chunk_elements, count - start)
                values = cpu_pattern(torch, source, destination, start, length)
                buffers.input.narrow(0, offset + start, length).copy_(values.to(buffers.input.dtype))
    torch.musa.synchronize()


def validate_buffers(torch, buffers, case, chunk_elements, timing=False):
    plan = buffers.plan
    constant = performance_constant(case) if timing else None
    report = {"valid": True, "checked_output_elements": 0, "checked_input_elements": 0,
              "wrong_elements": 0, "max_absolute_error": 0.0,
              "reference": "CPU_FP64_exact_representable_pattern", "tolerance": 0}

    def check(tensor, offset, values, is_input=False):
        actual = tensor.narrow(0, offset, values.numel()).cpu().to(torch.float64)
        expected = values.to(torch.float64)
        finite = torch.isfinite(actual)
        mismatch = (~finite) | (actual != expected)
        wrong = int(mismatch.sum().item())
        report["wrong_elements"] += wrong
        report["valid"] &= wrong == 0
        report["checked_input_elements" if is_input else "checked_output_elements"] += actual.numel()
        error = torch.abs(actual[finite] - expected[finite])
        if error.numel():
            report["max_absolute_error"] = max(report["max_absolute_error"], float(error.max().item()))

    if plan.primitive in REDUCTIONS:
        for start in range(0, plan.output_elements, chunk_elements):
            length = min(chunk_elements, plan.output_elements - start)
            expected = (torch.full((length,), constant, dtype=torch.float64) if constant is not None
                        else cpu_reference(torch, plan, case, start, length))
            check(buffers.output, start, expected)
    else:
        for offset, count, source, destination in output_segments(case, plan):
            for start in range(0, count, chunk_elements):
                length = min(chunk_elements, count - start)
                check(buffers.output, offset + start, cpu_pattern(torch, source, destination, start, length))
    # Out-of-place calls must not overwrite the independently allocated input.
    if case["buffer_mode"] == "out_of_place":
        if plan.primitive in REDUCTIONS:
            for start in range(0, plan.input_elements, chunk_elements):
                length = min(chunk_elements, plan.input_elements - start)
                check(buffers.input, start, cpu_reduction_input(torch, plan.rank, start, length, case["reduce_op"], case), True)
        else:
            for offset, count, source, destination in input_segments(case, plan):
                for start in range(0, count, chunk_elements):
                    length = min(chunk_elements, count - start)
                    check(buffers.input, offset + start, cpu_pattern(torch, source, destination, start, length), True)
    return report


def implementation_contract(native, case, args):
    p = case["primitive"]
    names = {"all_reduce": "mcclAllReduce", "all_gather": "mcclAllGather",
             "reduce_scatter": "mcclReduceScatter", "broadcast": "mcclBroadcast",
             "reduce": "mcclReduce", "all_to_all": "mcclAlltoAll",
             "gather": "mcclGather", "scatter": "mcclScatter"}
    if p == "all_to_all_v":
        if args.alltoallv_path == "torch":
            return "torch_musa_all_to_all_single_splits", False, None
        if args.alltoallv_path == "native":
            if "global_sdispls" not in native.header or "mcclAllToAllv" not in native.functions:
                return None, False, "installed_AllToAllv_global_displacements_ABI_not_confirmed"
            return "native_mcclAllToAllv_global_displacements", True, None
        return "composite_mccl_grouped_send_recv", False, None
    if p == "send_recv":
        return "native_mccl_grouped_p2p", True, None
    # A catalog backend is a requested measurement contract, not just a label.
    # Keep an explicitly selected composite path stable across SDK versions
    # that may additionally expose a native entry point.
    if p in {"all_to_all", "gather", "scatter"} and case["backend"] == "mccl_grouped_send_recv":
        return "composite_mccl_grouped_send_recv", False, None
    if names.get(p) in native.functions:
        return "native_" + names[p], True, None
    if p in {"all_to_all", "gather", "scatter"} and args.missing_native == "grouped":
        return "composite_mccl_grouped_send_recv", False, None
    return None, False, "installed_public_native_API_not_exposed"


def prepare_submit(native, buffers, case, stream, implementation, torch, dist, torch_group):
    """Build every argument/array/view once; returned callable only communicates."""
    plan = buffers.plan
    n, r, p, c, d = plan.world, plan.rank, plan.primitive, plan.api_count, plan.element_bytes
    dtype, op, root = native.DTYPE[case["dtype"]], native.OP.get(case["reduce_op"], 0), case["root"]
    sp, rp = buffers.input.data_ptr(), buffers.output.data_ptr()
    stream_pointer = C.c_void_p(int(stream.musa_stream))
    comm = native.comm
    if implementation == "torch_musa_all_to_all_single_splits":
        send_splits, receive_splits = list(plan.sendcounts), list(plan.recvcounts)
        def submit_torch():
            work = dist.all_to_all_single(buffers.logical_output, buffers.logical_input,
                                         output_split_sizes=receive_splits, input_split_sizes=send_splits,
                                         group=torch_group, async_op=True)
            work.wait()  # Establish backend-stream -> calling-stream completion dependency.
        return submit_torch
    if implementation == "native_mcclAllToAllv_global_displacements":
        count_array = lambda values: (C.c_size_t * len(values))(*values)
        arrays = [count_array(values) for values in (plan.sendcounts, plan.global_sdispls,
                                                    plan.recvcounts, plan.recvdispls)]
        args = (sp, arrays[0], arrays[1], rp, arrays[2], arrays[3], dtype, comm, stream_pointer)
        return lambda: native.call("mcclAllToAllv", *args)
    if implementation.startswith("native_") and p != "send_recv":
        fn = implementation[len("native_"):]
        tail = (comm, stream_pointer)
        if p in {"all_reduce", "reduce_scatter"}:
            call_args = (sp, rp, c, dtype, op) + tail
        elif p == "reduce":
            call_args = (sp, rp, c, dtype, op, root) + tail
        elif p in {"broadcast", "gather", "scatter"}:
            call_args = (sp, rp, c, dtype, root) + tail
        else:
            call_args = (sp, rp, c, dtype) + tail
        return lambda: native.call(fn, *call_args)
    calls = []
    def send(peer, offset, count):
        if count:
            calls.append(("mcclSend", (sp + offset * d, count, dtype, peer, comm, stream_pointer)))
    def recv(peer, offset, count):
        if count:
            calls.append(("mcclRecv", (rp + offset * d, count, dtype, peer, comm, stream_pointer)))
    if p == "all_to_all":
        for peer in range(n):
            send(peer, peer * c, c)
            recv(peer, peer * c, c)
    elif p == "all_to_all_v":
        for peer in range(n):
            send(peer, plan.senddispls[peer], plan.sendcounts[peer])
            recv(peer, plan.recvdispls[peer], plan.recvcounts[peer])
    elif p == "gather":
        send(root, 0, c)
        if r == root:
            for peer in range(n):
                recv(peer, peer * c, c)
    elif p == "scatter":
        if r == root:
            for peer in range(n):
                send(peer, peer * c, c)
        recv(root, 0, c)
    elif p == "send_recv":
        src, dst = case["participants"]
        if r == src:
            send(dst, 0, c)
            if case["direction"] == "bidirectional":
                recv(dst, 0, c)
        elif r == dst:
            recv(src, 0, c)
            if case["direction"] == "bidirectional":
                send(src, 0, c)
    else:
        raise ValueError(f"no composite route for {p}")
    def submit_grouped():
        if not calls:
            return
        native.call("mcclGroupStart")
        try:
            for fn, params in calls:
                native.call(fn, *params)
        finally:
            native.call("mcclGroupEnd")
    return submit_grouped


def timed_block(torch, control, native, submit, stream, start_event, stop_event,
                iterations, timeout_seconds):
    # Synchronization/readiness is deliberately performed by the caller in a
    # separate coordinated phase, so local preparation errors cannot cause
    # mismatched Gloo collectives inside this function.
    host_start = time.perf_counter()
    start_event.record(stream)
    for _ in range(iterations):
        submit()
    stop_event.record(stream)
    deadline = host_start + timeout_seconds
    while not stop_event.query():
        if time.perf_counter() > deadline:
            raise TimeoutError("GPU completion event did not complete before deadline")
        time.sleep(0.0001)
    host_seconds = time.perf_counter() - host_start
    event_seconds = start_event.elapsed_time(stop_event) / 1000.0
    native.check_async()
    return {"rank": control.dist.get_rank(), "event_seconds": event_seconds,
            "host_seconds": host_seconds}


def run_case(case, args, rank, device, torch, dist, control, native, torch_group, schema, journal):
    case_id = case["case_id"]
    record = {"record": "case_result", "case_id": case_id, "case_hash": case["case_hash"],
              "config_id": case["config_id"], "primitive": case["primitive"],
              "requested_backend": case["backend"], "dtype": case["dtype"],
              "reduce_op": case["reduce_op"], "root": case["root"],
              "buffer_mode": case["buffer_mode"], "size": case["size"],
              "participants": case["participants"], "direction": case["direction"],
              "platform_status": "not_reached", "official_status": "reference_unverified"}
    reason = None
    if case["status"] != "planned":
        reason = "invalid_combination"
    elif case["family"] not in {"collective", "p2p"}:
        reason = "not_a_collective_case"
    elif case["topology"]["ranks"] != control.world:
        reason = "world_size_mismatch"
    elif case["environment_id"] != args.environment_id or args.environment_id == "UNVERIFIED":
        reason = "environment_binding_missing_or_mismatched"
    elif case["size"]["actual"] is None:
        reason = "unresolved_route_materialization"
    elif case["backend"] not in {"mccl_native", "mccl_grouped_send_recv", "torch_musa_mccl"}:
        reason = "backend_not_implemented_by_this_adapter"
    else:
        binding = case["physical_mapping"][rank]
        if binding["host"] != args.host_id or binding["physical_gpu"] != args.bound_physical_gpu:
            reason = "physical_host_GPU_binding_mismatch"
    reasons = control.gather(reason)
    if any(reasons):
        record.update(status="invalid_combination" if reason == "invalid_combination" else "pending",
                      requested_case_status="pending", reasons_by_rank=reasons)
        journal.write(record)
        return record
    plan = control.phase("rank_shape_plan", lambda: make_rank_plan(case, rank, schema))
    implementation, native_api, unavailable = implementation_contract(native, case, args)
    reasons = control.gather(unavailable)
    if any(reasons):
        record.update(status="pending", requested_case_status="pending", reasons_by_rank=reasons,
                      capability_status="unverified")
        journal.write(record)
        return record
    requested_satisfied = (native_api and case["backend"] == "mccl_native"
                           or implementation.startswith("composite_") and case["backend"] == "mccl_grouped_send_recv"
                           or implementation.startswith("torch_") and case["backend"] == "torch_musa_mccl")
    record.update(implementation=implementation, native_public_api=native_api,
                  requested_backend_satisfied=bool(requested_satisfied),
                  timing_input="constant_zero_or_one_to_prevent_recursive_reduction_overflow"
                  if performance_constant(case) is not None else "fixed_exact_pattern",
                  rank_plan=asdict(plan))
    contract = {"runner_sha256": args.runner_sha256, "support_code_hashes": args.support_code_hashes,
                "implementation": implementation, "settings": args.measurement_settings,
                "communication_environment": args.communication_environment,
                "mccl": native.metadata, "torch": str(torch.__version__),
                "environment_id": args.environment_id, "measurement": "5plus_blocks_MUSA_events_and_host_completion_v1"}
    record["execution_signature"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    journal.write({"record": "case_phase", "case_id": case_id, "phase": "memory_preflight"})
    def preflight_memory():
        torch.musa.empty_cache()
        free, total = torch.musa.mem_get_info(device)
        return memory_decision(plan, free_bytes=int(free), total_bytes=int(total), fraction=args.memory_fraction,
                               workspace_reserve=args.workspace_reserve_bytes,
                               extra_backend_bytes=plan.storage_bytes if implementation.startswith("torch_") else 0)
    memory = control.phase("memory_query", preflight_memory)
    memory_reports = control.gather(memory)
    record["memory_preflight_by_rank"] = memory_reports
    if not all(row["allowed"] for row in memory_reports):
        record.update(status="capacity_limited", requested_case_status="capacity_limited",
                      reason="all_ranks_agreed_before_allocation", capability_status="unverified")
        journal.write(record)
        return record
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32, "uint8_bytes": torch.uint8}[case["dtype"]]
    buffers = control.phase("preallocation", lambda: Buffers(torch, plan, dtype, device))
    stream = torch.musa.current_stream(device)
    submit = control.phase("prepare_callable", lambda: prepare_submit(native, buffers, case, stream,
                            implementation, torch, dist, torch_group))
    def check_once(label, timing=False, initialize=True):
        journal.write({"record": "case_phase", "case_id": case_id, "phase": label})
        if initialize:
            control.phase(label + "_initialize", lambda: initialize_buffers(torch, buffers, case,
                                                                           args.chunk_elements, timing))
            control.barrier()
            control.phase(label + "_communicate", lambda: (submit(), torch.musa.synchronize(), native.check_async()))
        validation = control.phase(label + "_validate", lambda: validate_buffers(torch, buffers, case,
                                                                                 args.chunk_elements, timing))
        reports = control.gather(validation)
        record[label] = reports
        if not all(value["valid"] for value in reports):
            record.update(status="failed_correctness", requested_case_status="failed_correctness",
                          capability_status="unverified")
            journal.write(record)
            raise PhaseFailure("correctness", reports)
    check_once("correctness_pre")
    control.phase("timing_initialize", lambda: initialize_buffers(torch, buffers, case,
                                                                   args.chunk_elements, timing=True))
    def warmup():
        for _ in range(args.warmup):
            submit()
        torch.musa.synchronize()
    control.phase("warmup", warmup)
    # MUSA Event constructors may be lazy. Materialize both outside any measurement.
    def prepare_events():
        start, stop = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
        start.record(stream)
        stop.record(stream)
        stop.synchronize()
        return start, stop
    start_event, stop_event = control.phase("materialize_events", prepare_events)
    def measure(iters, phase):
        journal.write({"record": "case_phase", "case_id": case_id, "phase": phase, "iterations": iters})
        control.phase(phase + "_ready", torch.musa.synchronize)
        control.barrier()
        local = control.phase(phase, lambda: timed_block(torch, control, native, submit, stream,
                              start_event, stop_event, iters, args.operation_timeout_seconds))
        return control.gather(local)
    if case["size"]["actual"] == 0:
        # Empty routes have no meaningful device throughput or saturation
        # window. Preserve their invocation/completion overhead separately;
        # an empty grouped route is an adapter no-op, not a native API proof.
        rows = [{"iterations": 200, "ranks": measure(200, f"zero_control_block_{i}")}
                for i in range(args.blocks)]
        check_once("correctness_post")
        record.update(status="passed", requested_case_status="passed" if requested_satisfied else "pending",
                      control_only=True, formal_timing_qualified=False,
                      capability_status="unverified_by_zero_payload",
                      platform_status="not_applicable", bandwidth_applicable=False,
                      zero_payload_control_blocks=rows,
                      zero_payload_note="adapter invocation overhead only; no link bandwidth or payload correctness evidence")
        journal.write(record)
        return record
    pilot = measure(args.pilot_iterations, "pilot")
    record["pilot"] = {"iterations": args.pilot_iterations, "ranks": pilot}
    iterations = choose_iterations(max(x["event_seconds"] for x in pilot), args.pilot_iterations,
                                   case["size"]["actual"], args.block_seconds,
                                   maximum=args.max_iterations)
    blocks, attempts = [], []
    for index in range(args.blocks):
        for retry in range(5):
            rows = measure(iterations, f"block_{index}_attempt_{retry}")
            block = {"iterations": iterations, "ranks": rows}
            duration = max(x["event_seconds"] for x in rows)
            if duration >= args.block_seconds:
                blocks.append(block)
                break
            attempts.append({"reason": "duration_below_target", **block})
            iterations = choose_iterations(duration, iterations, case["size"]["actual"],
                                           args.block_seconds, maximum=args.max_iterations)
        else:
            raise PhaseFailure("measurement_window_not_met", attempts)
    check_once("timing_output_validation", timing=True, initialize=False)
    check_once("correctness_post")
    record.update(status="passed", requested_case_status="passed" if requested_satisfied else "pending",
                  capability_status="supported_for_this_execution_contract_only",
                  timing_blocks=blocks, rejected_timing_attempts=attempts,
                  summary=summarize_blocks(case, blocks), formal_timing_qualified=True,
                  gpu_completion="MUSA_current_stream_event_query_and_native_async_error_check",
                  control_backend="gloo_cpu", lease_ids=args.lease_ids,
                  reservation_evidence=args.reservation_evidence)
    if implementation.startswith("composite_"):
        record["native_coverage_note"] = "composite result is not evidence of a native entrypoint"
    journal.write(record)
    return record


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def communication_environment(environ):
    """Capture effective MCCL knobs, including inherited vendor-specific ones.

    Never dump the general process environment: credentials for unrelated
    services do not belong in benchmark journals. Runtime journals may still
    contain network-interface names or paths and should be reviewed before sharing.
    """
    keys = {"MUSA_VISIBLE_DEVICES", "MCCL_ALGOS", "MCCL_PROTOS", "MCCL_ALGO", "MCCL_PROTO",
            "MCCL_SOCKET_IFNAME", "MCCL_IB_HCA", "MCCL_IB_GID_INDEX", "MCCL_NET_SHARED_BUFFERS",
            "MCCL_DEBUG", "MCCL_MIN_NCHANNELS", "MCCL_MAX_NCHANNELS", "MCCL_P2P_DISABLE",
            "MCCL_P2P_LEVEL", "MCCL_IB_DISABLE", "MCCL_NET"}
    keys.update(key for key in environ if key.startswith("MCCL_"))
    return {key: environ.get(key) for key in sorted(keys)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--catalog-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--environment-id")
    parser.add_argument("--lease-ids", nargs="+", help="external orchestrator evidence; does not acquire, verify, or renew leases")
    parser.add_argument("--reservation-evidence", help="external reservation/preflight attestation; not verified resource isolation")
    parser.add_argument("--host-id", default=os.environ.get("MUSACOMM_HOST_ID", socket.gethostname()))
    parser.add_argument("--physical-gpu-map", help="verified host physical IDs in LOCAL_RANK order, e.g. 0,1,2,3,4,5,6,7")
    parser.add_argument("--mccl-library", type=Path)
    parser.add_argument("--mccl-header", type=Path)
    parser.add_argument("--alltoallv-path", choices=("torch", "grouped", "native"), default="torch")
    parser.add_argument("--missing-native", choices=("grouped", "pending"), default="grouped")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pilot-iterations", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--block-seconds", type=float, default=0.2)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument("--memory-fraction", type=float, default=0.60)
    parser.add_argument("--workspace-reserve-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--chunk-elements", type=int, default=65536)
    parser.add_argument("--operation-timeout-seconds", type=float, default=120)
    parser.add_argument("--control-timeout-seconds", type=float, default=600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.blocks < 5 or args.block_seconds < 0.2 or args.warmup < 20:
        parser.error("formal mode requires >=5 blocks, >=0.2s/block, >=20 warmup operations")
    if args.pilot_iterations < 1 or args.max_iterations < 200 or args.chunk_elements < 1:
        parser.error("invalid iteration/chunk settings")
    if not 0 < args.memory_fraction <= 0.60 or args.workspace_reserve_bytes < 0:
        parser.error("memory fraction must be in (0,0.60], reserve non-negative")
    if not all(math.isfinite(v) and v > 0 for v in
               (args.block_seconds, args.operation_timeout_seconds, args.control_timeout_seconds)):
        parser.error("timeouts and measurement durations must be finite and positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.validate_only:
        for name in ("outdir", "environment_id", "physical_gpu_map", "mccl_library", "mccl_header"):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for GPU execution")
        if not args.lease_ids and not args.reservation_evidence:
            parser.error("--lease-ids or --reservation-evidence is required for GPU execution")
    return args


def validate_execution_scope(args, manifest, schema):
    """Fail before GPU imports if a manifest exceeds the public v0.1 scope."""
    if manifest["world_sizes"] != [8]:
        raise ValueError("musacomm v0.1 GPU execution requires exactly eight ranks on one node")
    physical = [int(part) for part in args.physical_gpu_map.split(",")]
    if len(physical) != 8 or len(set(physical)) != 8 or any(p < 0 for p in physical):
        raise ValueError("musacomm v0.1 requires eight unique physical GPU IDs")
    for case in iter_cases(args.cases, schema):
        if case["topology"]["nodes"] != 1 or case["family"] not in {"collective", "p2p"}:
            raise ValueError("multi-node and MoE execution are not supported by musacomm v0.1")
        if any(item["host"] != args.host_id or item["physical_gpu"] != physical[item["rank"]]
               for item in case["physical_mapping"]):
            raise ValueError("every manifest rank must bind to this host and the supplied GPU map")
    return physical


def main(argv=None):
    args = parse_args(argv)
    catalog, schema = catalog_modules(args.catalog_dir)
    manifest = audit_manifest(args.cases, schema)
    if args.validate_only:
        print(json.dumps({"CPU_manifest_validation": "passed", **manifest}, sort_keys=True))
        return 0
    physical = validate_execution_scope(args, manifest, schema)
    rank, world, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 8 or not 0 <= rank < 8 or local_rank != rank:
        raise ValueError("launch with one local rank per GPU: torchrun --standalone --nproc_per_node=8")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "8")) != 8:
        raise ValueError("all eight ranks must belong to one node")
    import torch
    import torch.distributed as dist
    import torch_musa

    args.bound_physical_gpu = physical[local_rank]
    args.runner_sha256 = file_sha256(__file__)
    args.support_code_hashes = {"helpers": file_sha256(helper_module.__file__),
                               "catalog": file_sha256(catalog.__file__), "schema": file_sha256(schema.__file__)}
    args.communication_environment = communication_environment(os.environ)
    args.measurement_settings = {key: getattr(args, key) for key in
                                ("warmup", "pilot_iterations", "blocks", "block_seconds",
                                 "memory_fraction", "workspace_reserve_bytes", "chunk_elements")}
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.control_timeout_seconds))
    control = Control(dist, world)
    journal, native, fatal = None, None, False
    counts, requested_counts = {}, {}
    try:
        manifests = control.gather(manifest)
        if any(value != manifest for value in manifests) or manifest["world_sizes"] != [world]:
            raise ValueError("all ranks must use the same manifest and matching world size")
        journal = control.phase("journal_create", lambda: Journal(args.outdir, rank))
        mapping_evidence = control.phase("visible_device_binding", lambda: validate_visible_mapping(
            physical, os.environ.get("MUSA_VISIBLE_DEVICES"), torch.musa.device_count(), local_rank))
        torch.musa.set_device(local_rank)
        device = torch.device("musa", local_rank)
        native = control.phase("native_library_binding", lambda: NativeMCCL(args.mccl_library, args.mccl_header))
        versions = control.gather({key: native.metadata[key] for key in
                                   ("version_code", "header_sha256", "library_sha256")})
        if any(value != versions[0] for value in versions):
            raise ValueError("native MCCL header/library evidence differs across ranks")
        native.initialize(rank, world, control)
        torch_group = (dist.new_group(backend="mccl", timeout=timedelta(seconds=args.operation_timeout_seconds))
                       if args.alltoallv_path == "torch" else None)
        journal.write({"record": "run_environment", "manifest": manifest, "runner_sha256": args.runner_sha256,
                       "support_code_hashes": args.support_code_hashes,
                       "settings": args.measurement_settings,
                       "communication_environment": args.communication_environment,
                       "torch_version": str(torch.__version__), "torch_musa_version": str(getattr(torch_musa, "__version__", "unknown")),
                       "native_library": native.metadata, "host_id": args.host_id,
                       "local_rank": local_rank, "physical_gpu": args.bound_physical_gpu,
                       "mapping_evidence": mapping_evidence,
                       "environment_id": args.environment_id, "lease_ids": args.lease_ids,
                       "reservation_evidence": args.reservation_evidence,
                       "control_backend": "gloo", "alltoallv_path": args.alltoallv_path})
        for index, case in enumerate(iter_cases(args.cases, schema)):
            if args.limit is not None and index >= args.limit:
                break
            try:
                result = run_case(case, args, rank, device, torch, dist, control, native, torch_group, schema, journal)
            except Exception as exc:
                fatal = True
                journal.write({"record": "case_abort" if getattr(exc, "phase", "") == "correctness" else "case_result",
                               "case_id": case["case_id"], "status": "runtime_error",
                               "requested_case_status": "runtime_error", "capability_status": "unverified",
                               "phase": getattr(exc, "phase", "local"),
                               "error": {"type": type(exc).__name__, "message": str(exc)[:4000]},
                               "rank_reports": getattr(exc, "reports", None),
                               "traceback": traceback.format_exc(limit=5)})
                raise
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            requested_counts[result["requested_case_status"]] = requested_counts.get(result["requested_case_status"], 0) + 1
            gc.collect()
        journal.write({"record": "run_summary", "status_counts": counts,
                       "requested_case_status_counts": requested_counts,
                       "manifest_cases": manifest["cases"], "recorded_cases": sum(counts.values()),
                       "all_requested_cases_passed": requested_counts.get("passed", 0) == manifest["cases"],
                       "note": "per-row requested_case_status is authoritative; composite fallback cannot complete native coverage"})
        control.barrier()
    except Exception:
        fatal = True
        raise
    finally:
        if native is not None:
            try:
                native.close(abort=fatal)
            except Exception as exc:
                if journal is not None:
                    journal.write({"record": "cleanup_error", "error": str(exc)[:2000]})
        if journal is not None:
            journal.close()
        # On a fatal distributed failure, let torchrun terminate only this job;
        # graceful group destruction can otherwise wait on failed peers forever.
        if not fatal and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
