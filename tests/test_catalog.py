"""CPU-only stdlib tests for immutable catalog contracts."""
from __future__ import annotations

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from musacomm.catalog import catalog, schema
from musacomm.catalog.state import StateStore

HERE = Path(__file__).resolve().parent


def fixed_case(primitive="all_reduce", topo="A8", dtype="bf16", root=None, op=None):
    spec = catalog.new_spec("collective", primitive, topo, dtype)
    spec.update(root=root, reduce_op=op)
    size = next(catalog.fixed_sizes(primitive, spec["topology"]["ranks"], dtype))
    spec["size"] = catalog.fixed_size_for_spec(size, spec)
    return catalog.build_case(spec)


class CatalogTests(unittest.TestCase):
    def test_ten_primitive_vocabulary(self):
        self.assertEqual(len(catalog.CORE_PRIMITIVES), 10)
        self.assertEqual(set(catalog.CORE_PRIMITIVES), {
            "all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce",
            "all_to_all", "all_to_all_v", "gather", "scatter", "send_recv"})

    def test_canonical_identity_and_audit_metadata(self):
        case = fixed_case(op="sum")
        self.assertEqual(case, fixed_case(op="sum"))
        spec = schema.spec_from_case(case)
        spec["size"]["requests"].append({"kind": "audit_alias", "requested": 123})
        self.assertEqual(catalog.build_case(spec)["case_id"], case["case_id"])
        spec["environment_id"] = "verified-environment-sha256"
        self.assertNotEqual(catalog.build_case(spec)["case_id"], case["case_id"])
        changed = copy.deepcopy(case)
        changed["size"]["input_bytes_by_rank"][0] += 2
        self.assertTrue(schema.validate_case(changed))

    def test_fixed_sizes_are_legal_unique_and_cover_aliases(self):
        for primitive in catalog.CORE_PRIMITIVES:
            if primitive == "all_to_all_v":
                continue
            for n in (8, 16):
                for dtype in catalog.DTYPES:
                    rows = list(catalog.fixed_sizes(primitive, n, dtype))
                    actual = [row["actual"] for row in rows]
                    self.assertEqual(actual, sorted(set(actual)))
                    aliases = {(entry["kind"], entry["requested"]) for row in rows for entry in row["requests"]}
                    self.assertTrue({("S_bytes", size) for size in catalog.BASE_SIZES} <= aliases)
                    self.assertTrue({("api_count", count) for count in (1, 2, 3)} <= aliases)
                    for row in rows:
                        self.assertGreater(row["actual"], 0)
                        self.assertEqual(row["actual"], row["api_count"] * row["legal_quantum_bytes"])

    def test_root_ops_and_shape_schema(self):
        for topo in ("A8", "AB16"):
            n = catalog.topology(topo)["ranks"]
            for primitive in catalog.PRIMITIVES:
                if primitive == "all_to_all_v":
                    continue
                for root in range(n) if primitive in catalog.ROOTED else (None,):
                    for op in catalog.REDUCE_OPS if primitive in catalog.REDUCTION else (None,):
                        for dtype in catalog.DTYPES:
                            case = fixed_case(primitive, topo, dtype, root, op)
                            self.assertEqual(schema.validate_case(case), [], case["case_id"])
                            self.assertEqual(case["capability"]["status"], "unverified")

    def test_invalid_is_preserved_without_support_claim(self):
        case = fixed_case("broadcast", root=99)
        self.assertEqual(case["status"], "invalid_combination")
        self.assertTrue(case["invalid_reasons"])
        self.assertEqual(schema.validate_case(case), [])
        self.assertEqual(catalog.validate_spec({}), ["missing spec fields: " + ",".join(catalog.SPEC_FIELDS)])
        wrong = copy.deepcopy(case)
        wrong["status"] = "planned"
        self.assertTrue(schema.validate_case(wrong))
        wrong = copy.deepcopy(case)
        wrong["capability"]["status"] = "supported"
        self.assertTrue(schema.validate_case(wrong))

    def test_alltoallv_counts_transpose_displacements_and_routes(self):
        for topo in ("A8", "AB16"):
            n = catalog.topology(topo)["ranks"]
            routes = list(catalog.variable_routes(n))
            self.assertEqual(routes, list(catalog.variable_routes(n)))
            self.assertEqual(len({route["id"] for route in routes}), len(routes))
            for route in routes:
                spec = catalog.new_spec("collective", "all_to_all_v", topo, "bf16")
                spec["route"] = route
                sizes = list(catalog.variable_sizes(route, n, "bf16"))
                for size in (sizes[0], sizes[-1]):
                    spec["size"] = size
                    case = catalog.build_case(spec)
                    self.assertEqual(schema.validate_case(case), [])
                    if route["weight_matrix"] is None:
                        with self.assertRaises(ValueError):
                            schema.alltoallv_counts(case)
                        continue
                    counts = schema.alltoallv_counts(case)
                    for src in range(n):
                        for dst in range(n):
                            self.assertEqual(counts["sendcounts"][src][dst], counts["recvcounts"][dst][src])
                        self.assertEqual(counts["senddispls"][src][0], 0)
                    if route["id"].startswith("node_fraction_") and size["actual"]:
                        total = sum(size["input_bytes_by_rank"])
                        self.assertEqual(size["cross_node_payload_bytes"] / total, route["cross_node_payload_target"])

    def test_moe_contract_dimensions_and_metric_grouping(self):
        base_combinations = 4 * 4 * 3 * (4 + 7) * len(catalog.BASE_TOKENS)
        self.assertEqual(base_combinations, 8448)
        self.assertEqual(base_combinations * 2, 16896)
        self.assertEqual(base_combinations * 2 * 3, 50688)
        cases = list(itertools.islice(catalog.moe_cases(), 6))
        self.assertEqual({case["metric"] for case in cases}, {"dispatch", "combine", "roundtrip"})
        self.assertEqual(len({case["run_group_id"] for case in cases}), 2)
        self.assertEqual(len({case["case_id"] for case in cases}), 6)
        for case in cases:
            self.assertEqual(schema.validate_case(case), [])
        original = cases[0]
        size = copy.deepcopy(original["size"])
        size.update(requested=65536, actual=65536,
                    input_bytes_per_rank=65536 * original["moe"]["hidden"] * 2,
                    logical_expert_expanded_bytes_per_rank=65536 * original["moe"]["hidden"] * original["moe"]["top_k"] * 2)
        expanded = catalog.append_extension(original, size)
        self.assertEqual(schema.validate_case(expanded), [])
        self.assertNotEqual(expanded["case_id"], original["case_id"])
        self.assertEqual(expanded["config_id"], original["config_id"])

    def test_p2p_ownership(self):
        case = next(catalog.p2p_cases())
        self.assertEqual(schema.validate_case(case), [])
        src, dst = case["participants"]
        self.assertEqual(sum(case["size"]["input_bytes_by_rank"]), case["size"]["actual"])
        self.assertGreater(case["size"]["input_bytes_by_rank"][src], 0)
        self.assertGreater(case["size"]["output_bytes_by_rank"][dst], 0)

    def test_safe_manifest_publication(self):
        case = fixed_case(op="sum")
        with tempfile.TemporaryDirectory(prefix=".test-") as directory:
            path = Path(directory) / "manifest.jsonl"
            result = catalog.write_jsonl(path, [case])
            self.assertEqual(result["records"], 1)
            self.assertEqual(json.loads(path.read_text()), case)
            with self.assertRaises(FileExistsError):
                catalog.write_jsonl(path, [case])
            duplicate = Path(directory) / "duplicate.jsonl"
            with self.assertRaises(ValueError):
                catalog.write_jsonl(duplicate, [case, case])
            self.assertFalse(duplicate.exists())

    def test_state_gates_require_evidence(self):
        case = fixed_case(op="sum")
        with tempfile.TemporaryDirectory(prefix=".test-") as directory:
            store = StateStore(directory)
            self.assertEqual(store.register(case)["status"], "planned")
            with self.assertRaises(ValueError):
                store.update(case["case_id"], status="passed")
            with self.assertRaises(ValueError):
                store.update(case["case_id"], capability_status="supported")
            store.update(case["case_id"], capability_status="supported", evidence={"capability_evidence": "CPU test fixture only"})
            with self.assertRaises(ValueError):
                store.update(case["case_id"], status="running")
            with self.assertRaises(ValueError):
                store.update(case["case_id"], status="unsupported")
            self.assertFalse(store.summary()["all_performance_completed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
