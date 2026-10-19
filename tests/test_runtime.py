"""Pure CPU tests: no torch, MUSA, network, subprocess launch or compiler."""
from __future__ import annotations

import ast
import copy
import inspect
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from musacomm.runtime import collective as runner, helpers

catalog, schema = helpers.catalog_modules()
HERE = Path(__file__).resolve().parent


def case_for(primitive="all_reduce", dtype="bf16", mode="out_of_place", root=None,
             op="sum", topo="A8", size_index=1):
    family = "p2p" if primitive == "send_recv" else "collective"
    spec = catalog.new_spec(family, primitive, topo, dtype, environment_id="fixture")
    spec.update(buffer_mode=mode, root=root,
                reduce_op=op if primitive in catalog.REDUCTION else None)
    if family == "p2p":
        spec.update(participants=[1, 5], direction="unidirectional")
    spec["size"] = catalog.fixed_size_for_spec(list(catalog.fixed_sizes(
        primitive, spec["topology"]["ranks"], dtype))[size_index], spec)
    return catalog.build_case(spec)


def variable_case(route_id="random_unequal", topo="A8"):
    spec = catalog.new_spec("collective", "all_to_all_v", topo, "bf16")
    spec["route"] = next(r for r in catalog.variable_routes(spec["topology"]["ranks"]) if r["id"] == route_id)
    spec["size"] = next(catalog.variable_sizes(spec["route"], spec["topology"]["ranks"], "bf16"))
    return catalog.build_case(spec)


class Tensor:
    def __init__(self, count, address=1000, width=2, dtype="bf16"):
        self.count, self.address, self.width, self.dtype = count, address, width, dtype
    def data_ptr(self):
        return self.address
    def narrow(self, axis, offset, count):
        assert axis == 0 and offset >= 0 and count >= 0 and offset + count <= self.count
        return Tensor(count, self.address + offset * self.width, self.width, self.dtype)


class FakeTorch:
    def __init__(self):
        self.allocations = []
    def empty(self, count, *, device, dtype):
        tensor = Tensor(count, 1000 + 10000 * len(self.allocations), dtype=dtype)
        self.allocations.append(tensor)
        return tensor


class FakeNative:
    DTYPE, OP = runner.NativeMCCL.DTYPE, runner.NativeMCCL.OP
    def __init__(self, exposed=()):
        self.functions = dict.fromkeys(exposed, True)
        self.header = "global_sdispls"
        self.comm = 123
        self.calls = []
    def call(self, function, *args):
        self.calls.append((function, args))


class ArithmeticVector:
    """Tiny CPU stand-in to execute the runner's vectorized oracle itself."""
    def __init__(self, values):
        self.values = list(values)
    def combine(self, other, function):
        rhs = other.values if isinstance(other, ArithmeticVector) else [other] * len(self.values)
        return ArithmeticVector(function(a, b) for a, b in zip(self.values, rhs))
    def __add__(self, other):
        return self.combine(other, lambda a, b: a + b)
    __radd__ = __add__
    def __sub__(self, other):
        return self.combine(other, lambda a, b: a - b)
    def __rsub__(self, other):
        return self.combine(other, lambda a, b: b - a)
    def __mul__(self, other):
        return self.combine(other, lambda a, b: a * b)
    def __floordiv__(self, other):
        return self.combine(other, lambda a, b: a // b)
    def __eq__(self, other):
        return self.combine(other, lambda a, b: a == b)
    def remainder(self, other):
        return self.combine(other, lambda a, b: a % b)


class ArithmeticTorch:
    int64 = "int64"
    @staticmethod
    def arange(start, end, **kwargs):
        assert kwargs["device"] == "cpu"
        return ArithmeticVector(range(start, end))
    @staticmethod
    def where(condition, yes, no):
        return ArithmeticVector(yes if flag else no for flag in condition.values)


class VerifiedCPUContracts(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        self.assertNotIn("torch", vars(runner))
        self.assertNotIn("torch", vars(helpers))

    def test_catalog_modules_use_installed_package_without_sys_path_mutation(self):
        original = list(sys.path)
        self.assertEqual(helpers.catalog_modules(), (catalog, schema))
        self.assertEqual(helpers.catalog_modules(Path(catalog.__file__).parent), (catalog, schema))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                helpers.catalog_modules(directory)
        self.assertEqual(sys.path, original)

    def test_public_execution_scope_is_one_host_eight_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            case = case_for()
            spec = schema.spec_from_case(case)
            spec["physical_mapping"] = [
                {"rank": rank, "node": "A", "host": "fixture-host", "physical_gpu": rank}
                for rank in range(8)
            ]
            case = catalog.build_case(spec)
            path.write_text(json.dumps(case) + "\n")
            args = SimpleNamespace(cases=path, host_id="fixture-host", physical_gpu_map="0,1,2,3,4,5,6,7")
            manifest = helpers.audit_manifest(path, schema)
            self.assertEqual(manifest["case_ids"], [case["case_id"]])
            self.assertEqual(runner.validate_execution_scope(args, manifest, schema), list(range(8)))
            args.host_id = "another-host"
            with self.assertRaises(ValueError):
                runner.validate_execution_scope(args, manifest, schema)
            with self.assertRaises(ValueError):
                runner.validate_execution_scope(args, {"world_sizes": [16]}, schema)

    def test_runtime_source_hash_paths_are_installed_package_files(self):
        self.assertEqual(len(runner.file_sha256(runner.__file__)), 64)
        self.assertEqual(len(runner.file_sha256(runner.helper_module.__file__)), 64)

    def test_environment_evidence_includes_extra_mccl_but_not_unrelated_secrets(self):
        environment = {"MCCL_BUFFSIZE": "1048576", "MUSA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                       "UNRELATED_API_KEY": "not-for-journals"}
        evidence = runner.communication_environment(environment)
        self.assertEqual(evidence["MCCL_BUFFSIZE"], "1048576")
        self.assertNotIn("UNRELATED_API_KEY", evidence)
        environment["MCCL_BUFFSIZE"] = "2097152"
        self.assertNotEqual(evidence, runner.communication_environment(environment))

    def test_all_fixed_shapes_and_aliases(self):
        checked = 0
        for topo in ("A8", "AB16"):
            n = catalog.topology(topo)["ranks"]
            for p in catalog.CORE_PRIMITIVES:
                if p == "all_to_all_v":
                    continue
                for dtype in ("bf16", "fp16", "fp32"):
                    for mode in (("out_of_place", "in_place") if p in catalog.INPLACE_CANDIDATES else ("out_of_place",)):
                        root = n - 1 if p in catalog.ROOTED else None
                        case = case_for(p, dtype, mode, root=root, topo=topo)
                        for rank in range(n):
                            plan = helpers.make_rank_plan(case, rank, schema)
                            self.assertEqual(plan.input_elements * plan.element_bytes, case["size"]["input_bytes_by_rank"][rank])
                            self.assertEqual(plan.output_elements * plan.element_bytes, case["size"]["output_bytes_by_rank"][rank])
                            buffers = runner.Buffers(FakeTorch(), plan, dtype, "fake")
                            self.assertEqual(sum(t.count for t in buffers.owned), plan.allocated_elements)
                            if plan.alias == "input_in_output":
                                self.assertEqual(buffers.input.data_ptr(), buffers.output.data_ptr() + plan.alias_offset * 2)
                            if plan.alias == "output_in_input":
                                self.assertEqual(buffers.output.data_ptr(), buffers.input.data_ptr() + plan.alias_offset * 2)
                            checked += 1
        self.assertGreater(checked, 1000)

    def test_rooted_memory_ownership(self):
        case = case_for("gather", mode="in_place", root=3)
        root = helpers.make_rank_plan(case, 3, schema)
        peer = helpers.make_rank_plan(case, 0, schema)
        self.assertEqual(root.alias, "input_in_output")
        self.assertIsNone(peer.alias)
        self.assertEqual(peer.output_elements, 0)
        self.assertEqual(root.storage_bytes, case["size"]["actual"])

    def test_variable_routes_match_and_oracles_cover_every_receive(self):
        for topo in ("A8", "AB16"):
            n = catalog.topology(topo)["ranks"]
            for route in catalog.variable_routes(n):
                if route["weight_matrix"] is None:
                    continue
                case = variable_case(route["id"], topo)
                plans = [helpers.make_rank_plan(case, r, schema) for r in range(n)]
                for src in range(n):
                    for dst in range(n):
                        self.assertEqual(plans[src].sendcounts[dst], plans[dst].recvcounts[src])
                    self.assertEqual(len(plans[src].global_sdispls), n * n)
                    self.assertEqual(len(helpers.expected_output_values(case, plans[src])), plans[src].output_elements)

    def test_unresolved_route_is_not_executable(self):
        with self.assertRaises(ValueError):
            helpers.make_rank_plan(variable_case("message_hot_rank_balanced"), 0, schema)

    def test_transport_oracle_is_source_and_offset_sensitive(self):
        case = case_for("all_gather", size_index=3)
        plan = helpers.make_rank_plan(case, 4, schema)
        values = helpers.expected_output_values(case, plan)
        c = plan.api_count
        for source in range(plan.world):
            for index in range(c):
                self.assertEqual(values[source * c + index], helpers.pattern_value(source, 0, index))
        self.assertNotEqual(values[0], values[c])

    def test_exact_reduction_oracle_and_stable_timing_input(self):
        for op in catalog.REDUCE_OPS:
            for topo in ("A8", "AB16"):
                case = case_for(op=op, topo=topo)
                n = case["topology"]["ranks"]
                for index in range(50):
                    vals = [helpers.reduction_input_value(r, index, op, n) for r in range(n)]
                    expected = {"sum": sum(vals), "avg": sum(vals) / n,
                                "prod": math.prod(vals), "min": min(vals), "max": max(vals)}[op]
                    self.assertEqual(helpers.reduction_reference_value(index, n, op), expected)
                self.assertIsNone(helpers.performance_constant(case))
                inplace = case_for(mode="in_place", op=op, topo=topo)
                self.assertEqual(helpers.performance_constant(inplace), 0 if op == "sum" else 1)

    def test_inplace_min_max_noop_cannot_pass_any_root_or_rs_rank(self):
        for op in ("min", "max"):
            for n in (8, 16):
                for root in range(n):
                    original = helpers.reduction_input_value(root, 0, op, n, root=root)
                    expected = helpers.reduction_reference_value(0, n, op, root=root)
                    self.assertNotEqual(original, expected)
                for c in (1, 2, 7):
                    for rank in range(n):
                        index = rank * c
                        original = helpers.reduction_input_value(rank, index, op, n, receive_count=c)
                        expected = helpers.reduction_reference_value(index, n, op, receive_count=c)
                        self.assertNotEqual(original, expected)

    def test_inplace_prod_reduce_noop_fails_all_roots_and_small_counts(self):
        for n in (8, 16):
            for root in range(n):
                for count in (1, 2, 3, 7, 16):
                    for index in range(count):
                        values = [helpers.reduction_input_value(r, index, "prod", n, root=root)
                                  for r in range(n)]
                        expected = helpers.reduction_reference_value(index, n, "prod", root=root)
                        self.assertEqual(sum(abs(value) == 2 for value in values), 1)
                        self.assertEqual(abs(values[root]), 1)
                        self.assertEqual(abs(expected), 2)
                        self.assertNotEqual(values[root], expected)

    def test_inplace_prod_reduce_scatter_noop_fails_every_output_slot(self):
        for n in (8, 16):
            for count in (1, 2, 3, 7, 16):
                for receiver in range(n):
                    for within in range(count):
                        index = receiver * count + within
                        original = helpers.reduction_input_value(receiver, index, "prod", n,
                                                                 receive_count=count)
                        expected = helpers.reduction_reference_value(index, n, "prod",
                                                                     receive_count=count)
                        self.assertEqual(abs(original), 1)
                        self.assertEqual(abs(expected), 2)
                        self.assertNotEqual(original, expected)

    def test_runtime_vectorized_patterns_match_scalar_helpers(self):
        for n in (8, 16):
            for primitive in ("all_reduce", "reduce", "reduce_scatter"):
                for op in ("prod", "min", "max"):
                    for count in (1, 2, 7):
                        root = n - 1 if primitive == "reduce" else None
                        case = {"primitive": primitive, "topology": {"ranks": n},
                                "root": root, "size": {"api_count": count}}
                        for rank in range(n):
                            start = rank * count if primitive == "reduce_scatter" else 0
                            actual = runner.cpu_reduction_input(ArithmeticTorch, rank, start,
                                                                count, op, case).values
                            expected = [helpers.reduction_input_value(rank, index, op, n,
                                        root=root, receive_count=count if primitive == "reduce_scatter" else None)
                                        for index in range(start, start + count)]
                            self.assertEqual(actual, expected)

    def test_physical_mapping_must_match_runtime_mask(self):
        self.assertEqual(helpers.validate_visible_mapping(list(range(8)), None, 8, 7)["physical_gpu"], 7)
        self.assertEqual(helpers.validate_visible_mapping([2, 3], "2,3", 2, 1)["physical_gpu"], 3)
        for mapping, mask, count in (([2, 3], None, 2), ([0, 1], "1,0", 2), ([0, 1], None, 8)):
            with self.assertRaises(ValueError):
                helpers.validate_visible_mapping(mapping, mask, count, 0)

    def test_reduce_scatter_reference_uses_rank_slice(self):
        case = case_for("reduce_scatter", op="sum", size_index=3)
        plan = helpers.make_rank_plan(case, 5, schema)
        self.assertEqual(helpers.expected_output_values(case, plan),
                         [helpers.reduction_reference_value(5 * plan.api_count + i, 8, "sum")
                          for i in range(plan.output_elements)])

    def test_memory_limit_is_total_peak_not_only_new_tensor(self):
        plan = helpers.make_rank_plan(case_for(), 0, schema)
        good = helpers.memory_decision(plan, free_bytes=9000, total_bytes=10000,
                                       fraction=0.6, workspace_reserve=100)
        bad = helpers.memory_decision(plan, free_bytes=3000, total_bytes=10000,
                                      fraction=0.6, workspace_reserve=100)
        self.assertTrue(good["allowed"])
        self.assertFalse(bad["allowed"])
        with self.assertRaises(ValueError):
            helpers.memory_decision(plan, free_bytes=9000, total_bytes=10000, fraction=0.9, workspace_reserve=0)

    def test_standard_s_corrects_old_allgather_scatter_alltoall_numerator(self):
        for primitive in ("all_gather", "reduce_scatter", "all_to_all", "gather", "scatter"):
            case = case_for(primitive, root=0 if primitive in catalog.ROOTED else None)
            result = helpers.bandwidth_metrics(case, 1)
            self.assertEqual(result["algbw_GBps"], case["size"]["actual"] / 1e9)
            self.assertEqual(result["busbw_GBps"], case["size"]["actual"] / 1e9 * 7 / 8)

    def test_alltoallv_does_not_turn_Q_into_bus_bandwidth(self):
        case = variable_case()
        result = helpers.bandwidth_metrics(case, 0.5)
        self.assertIsNone(result["algbw_GBps"])
        self.assertIsNone(result["busbw_GBps"])
        self.assertEqual(result["global_nonself_effective_GBps"],
                         case["size"]["global_nonself_payload_bytes"] / 0.5 / 1e9)
        self.assertFalse(helpers.bandwidth_metrics(variable_case("all_zero"), 1)["bandwidth_applicable"])

    def test_p2p_direction_is_not_ring_or_undocumented_duplex_sum(self):
        case = case_for("send_recv")
        self.assertEqual(helpers.make_rank_plan(case, 0, schema).input_elements, 0)
        spec = schema.spec_from_case(case)
        spec["direction"] = "bidirectional"
        spec["size"] = catalog.fixed_size_for_spec(spec["size"], spec)
        bidir = catalog.build_case(spec)
        rate = helpers.bandwidth_metrics(bidir, 1)
        self.assertIsNone(rate["busbw_GBps"])
        self.assertEqual(rate["bidirectional_sum_effective_GBps"], 2 * rate["per_direction_effective_GBps"])

    def test_adaptive_iterations_obey_minimum_and_guard(self):
        self.assertEqual(helpers.choose_iterations(5, 5, 1024), 200)
        self.assertEqual(helpers.choose_iterations(5, 5, 2**30), 20)
        self.assertGreaterEqual(helpers.choose_iterations(0.001, 5, 1024), 1100)
        with self.assertRaises(ValueError):
            helpers.choose_iterations(1e-12, 5, 1024)

    def test_block_statistics_use_cross_rank_max_and_not_single_op_quantiles(self):
        case = case_for()
        blocks = [{"iterations": 200,
                   "ranks": [{"rank": r, "event_seconds": (r + 1) * 0.2, "host_seconds": (r + 1) * 0.3}
                             for r in range(8)]} for _ in range(5)]
        result = helpers.summarize_blocks(case, blocks)
        self.assertEqual(result["median_max_rank_device_seconds_per_op"], 1.6 / 200)
        self.assertFalse(result["single_iteration_percentiles_available"])
        blocks[0]["ranks"].pop()
        with self.assertRaises(ValueError):
            helpers.summarize_blocks(case, blocks)

    def test_native_out_of_place_uses_two_preallocated_pointers(self):
        case = case_for()
        plan = helpers.make_rank_plan(case, 0, schema)
        buffers = runner.Buffers(FakeTorch(), plan, "bf16", "fake")
        native = FakeNative(("mcclAllReduce",))
        submit = runner.prepare_submit(native, buffers, case, SimpleNamespace(musa_stream=9),
                                       "native_mcclAllReduce", None, None, None)
        self.assertFalse(native.calls)
        submit()
        fn, args = native.calls[-1]
        self.assertEqual(fn, "mcclAllReduce")
        self.assertNotEqual(args[0], args[1])
        self.assertEqual(args[2], plan.api_count)

    def test_grouped_alltoall_count_and_order(self):
        case = case_for("all_to_all")
        plan = helpers.make_rank_plan(case, 3, schema)
        buffers = runner.Buffers(FakeTorch(), plan, "bf16", "fake")
        native = FakeNative()
        submit = runner.prepare_submit(native, buffers, case, SimpleNamespace(musa_stream=0),
                                       "composite_mccl_grouped_send_recv", None, None, None)
        submit()
        self.assertEqual(native.calls[0][0], "mcclGroupStart")
        self.assertEqual(native.calls[-1][0], "mcclGroupEnd")
        self.assertEqual(len(native.calls), 18)
        self.assertEqual([params[3] for fn, params in native.calls if fn == "mcclSend"], list(range(8)))
        self.assertTrue(all(params[1] == plan.api_count for fn, params in native.calls if fn in ("mcclSend", "mcclRecv")))

    def test_native_variable_global_displacements_are_N_squared(self):
        case = variable_case()
        plan = helpers.make_rank_plan(case, 2, schema)
        buffers = runner.Buffers(FakeTorch(), plan, "bf16", "fake")
        native = FakeNative()
        submit = runner.prepare_submit(native, buffers, case, SimpleNamespace(musa_stream=0),
                                       "native_mcclAllToAllv_global_displacements", None, None, None)
        submit()
        self.assertEqual(len(native.calls[0][1][2]), 64)

    def test_torch_alltoallv_uses_per_rank_split_lists_and_wait(self):
        case = variable_case("zero_send_25_contiguous")
        plan = helpers.make_rank_plan(case, 2, schema)
        buffers = runner.Buffers(FakeTorch(), plan, "bf16", "fake")
        called = []
        def alltoall(*args, **kwargs):
            called.append(kwargs)
            return SimpleNamespace(wait=lambda: called.append("wait"))
        submit = runner.prepare_submit(FakeNative(), buffers, case, SimpleNamespace(musa_stream=0),
                                       "torch_musa_all_to_all_single_splits", None,
                                       SimpleNamespace(all_to_all_single=alltoall), "data-group")
        submit()
        self.assertEqual(called[0]["input_split_sizes"], list(plan.sendcounts))
        self.assertEqual(called[0]["output_split_sizes"], list(plan.recvcounts))
        self.assertEqual(called[-1], "wait")

    def test_missing_native_is_explicit_not_auto_supported(self):
        args = SimpleNamespace(missing_native="grouped", alltoallv_path="torch")
        implementation, native_api, error = runner.implementation_contract(FakeNative(), case_for("gather", root=0), args)
        self.assertEqual(implementation, "composite_mccl_grouped_send_recv")
        self.assertFalse(native_api)
        args.missing_native = "pending"
        self.assertIsNotNone(runner.implementation_contract(FakeNative(), case_for("scatter", root=0), args)[2])

    def test_requested_grouped_backend_is_stable_when_native_symbol_exists(self):
        args = SimpleNamespace(missing_native="grouped", alltoallv_path="grouped")
        for primitive, function in (("all_to_all", "mcclAlltoAll"),
                                    ("gather", "mcclGather"), ("scatter", "mcclScatter")):
            case = case_for(primitive, root=0 if primitive in catalog.ROOTED else None)
            native = FakeNative((function,))
            self.assertEqual(runner.implementation_contract(native, case, args)[0], "native_" + function)
            spec = schema.spec_from_case(case)
            spec["backend"] = "mccl_grouped_send_recv"
            grouped = catalog.build_case(spec)
            self.assertEqual(runner.implementation_contract(native, grouped, args),
                             ("composite_mccl_grouped_send_recv", False, None))

    def test_timed_loop_contains_only_communication(self):
        tree = ast.parse(inspect.getsource(runner.timed_block))
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For))
        self.assertEqual(len(loop.body), 1)
        self.assertEqual(loop.body[0].value.func.id, "submit")
        called = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for forbidden in ("empty", "full", "fill_", "copy_", "item", "cpu", "all_gather_object", "barrier"):
            self.assertNotIn(forbidden, called)

    def test_manifest_audit_identity_and_duplicate_rejection(self):
        case = case_for()
        with tempfile.TemporaryDirectory(prefix=".verified-cpu-") as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(case) + "\n")
            self.assertEqual(helpers.audit_manifest(path, schema)["cases"], 1)
            path.write_text((json.dumps(case) + "\n") * 2)
            with self.assertRaises(ValueError):
                helpers.audit_manifest(path, schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
