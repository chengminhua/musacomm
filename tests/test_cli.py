import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from musacomm import cli


class CliTests(unittest.TestCase):
    def arguments(self, *extra):
        args = cli.build_parser().parse_args(["allreduce", "--dry-run", *extra])
        cli.validate_args(args)
        return args

    def test_exact_byte_parser(self):
        self.assertEqual(cli.parse_bytes("1MiB"), 2**20)
        self.assertEqual(cli.parse_bytes("1MB"), 10**6)
        self.assertEqual(cli.parse_bytes("1.5KiB"), 1536)
        for value in ("0", "-1", "nan", "inf", "0.1KiB", "2e6", "1GiBjunk", "9999999999999999999999"):
            with self.subTest(value=value), self.assertRaises(Exception):
                cli.parse_bytes(value)

    def test_gpu_mapping_exact_and_explicit(self):
        self.assertEqual(cli.parse_gpus("7,6,5,4,3,2,1,0"), [7, 6, 5, 4, 3, 2, 1, 0])
        for value in ("0,1", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,7,", "-1,0,1,2,3,4,5,6"):
            with self.subTest(value=value), self.assertRaises(Exception):
                cli.parse_gpus(value)

    def test_reject_nonfinite_timing_and_invalid_limits(self):
        for values in (("--block-seconds", "nan"), ("--block-seconds", "inf"),
                       ("--launcher-timeout", "nan"), ("--launcher-timeout", "inf"),
                       ("--pilot-iterations", "0"), ("--limit", "0"), ("--max-cases", "0"),
                       ("--roots", "-1"), ("--sizes", "")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.arguments(*values)

    def test_all_reduce_operations_keep_all_sizes(self):
        args = self.arguments("--sizes", "1MiB,1GiB", "--reduce-op", "all")
        rows = cli.make_manifest(args, "example-host", list(range(8)), "test-environment")
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["reduce_op"] for row in rows}, set(cli.catalog.REDUCE_OPS))
        self.assertEqual({row["size"]["actual"] for row in rows}, {2**20, 2**30})

    def test_rooted_matrix_reuses_size_sequence(self):
        args = cli.build_parser().parse_args(["reduce", "--dry-run", "--sizes", "1MiB,1GiB",
                                              "--roots", "all", "--reduce-op", "all"])
        cli.validate_args(args)
        rows = cli.make_manifest(args, "example-host", list(range(8)), "test-environment")
        self.assertEqual(len(rows), 80)
        self.assertEqual(len({row["case_id"] for row in rows}), 80)

    def test_no_silent_missing_sizes_or_truncation(self):
        args = self.arguments("--sizes", "1MiB,3MiB")
        with self.assertRaisesRegex(ValueError, "unsupported exact byte sizes"):
            cli.make_manifest(args, "example-host", list(range(8)), "test-environment")
        args = self.arguments("--sizes", "1MiB,1GiB", "--max-cases", "1")
        with self.assertRaisesRegex(ValueError, "selection has 2 cases"):
            cli.make_manifest(args, "example-host", list(range(8)), "test-environment")
        args.allow_truncate = True
        self.assertEqual(len(cli.make_manifest(args, "example-host", list(range(8)), "test-environment")), 1)
        self.assertTrue(args.selection["truncated"])

    def test_timing_is_forwarded_and_module_is_packaged(self):
        args = self.arguments("--warmup", "27", "--blocks", "6", "--block-seconds", "0.3", "--pilot-iterations", "9")
        command = cli.runner_command(args, Path("/example/cases.jsonl"), Path("/example/output"),
                                     "test-environment", "example-host", list(range(8)), "example-evidence")
        self.assertIn("musacomm.runtime.collective", command)
        self.assertIn("--module", command)
        for key, value in (("--warmup", "27"), ("--blocks", "6"), ("--block-seconds", "0.3"), ("--pilot-iterations", "9")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("--catalog-dir", command)

    def test_portable_sdk_and_no_path_rewrite(self):
        args = self.arguments("--musa-home", "/opt/example-sdk")
        with mock.patch.dict(os.environ, {"PATH": "/example/bin", "MCCL_ALGOS": "1", "MCCL_ALGO": "Tree"}):
            env = cli.tuned_env(args, list(range(8)))
        self.assertEqual(env["PATH"], "/example/bin")
        self.assertEqual(env["MUSA_HOME"], "/opt/example-sdk")
        self.assertEqual(args.mccl_library, "/opt/example-sdk/lib/libmccl.so")
        self.assertNotIn("MCCL_ALGO", env)
        self.assertNotIn("MCCL_ALGOS", env)
        self.assertNotIn("musa-4.3.7", env["LD_LIBRARY_PATH"])

    def test_force_algorithm_env_is_explicit(self):
        args = self.arguments("--algo", "ring", "--proto", "simple")
        env = cli.tuned_env(args, list(range(8)))
        self.assertEqual((env["MCCL_ALGO"], env["MCCL_PROTO"]), ("Ring", "Simple"))
        self.assertNotIn("MCCL_ALGOS", env)
        args.mccl_env_style = "legacy"
        env = cli.tuned_env(args, list(range(8)))
        self.assertEqual((env["MCCL_ALGOS"], env["MCCL_PROTOS"]), ("1", "2"))
        self.assertNotIn("MCCL_ALGO", env)

    def test_reject_ignored_options(self):
        for values in (("--roots", "all"), ("--direction", "bidirectional"), ("--src", "3")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.arguments(*values)
        args = self.arguments("--backend", "torch")
        with self.assertRaisesRegex(ValueError, "supports only"):
            cli.make_manifest(args, "example-host", list(range(8)), "test-environment")

    def test_dry_run_without_gpu_or_torch_import(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "plan"
            code = (
                "import sys\n"
                "from musacomm.cli import main\n"
                f"rc = main(['allreduce', '--dry-run', '--sizes', '1MiB,1GiB', '--output', {str(output)!r}])\n"
                "assert 'torch' not in sys.modules\n"
                "raise SystemExit(rc)\n"
            )
            result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len((output / "cases.jsonl").read_text().splitlines()), 2)
            self.assertFalse((output / "results.csv").exists())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["allreduce", "--dry-run", "--output", str(output)]), 2)

    def test_all_primitive_plans(self):
        for primitive in set(cli.PRIMITIVE_ALIASES.values()):
            args = cli.build_parser().parse_args([primitive, "--dry-run", "--sizes", "1MiB"])
            cli.validate_args(args)
            with self.subTest(primitive=primitive):
                rows = cli.make_manifest(args, "example-host", list(range(8)), "test-environment")
                self.assertEqual(len(rows), 1)

    def test_skip_preflight_is_fully_diagnostic(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            library, header = root / "libmccl.so", root / "mccl.h"
            library.touch(); header.touch()
            args = self.arguments("--sizes", "1MiB", "--output", str(root / "out"),
                                  "--mccl-library", str(library), "--mccl-header", str(header), "--skip-preflight")
            args.dry_run = False
            outcome = {"exit_code": 0, "monitor_error": None, "interrupted": False,
                       "timed_out": False, "cleanup_error": None}
            with mock.patch.object(cli.safety, "sample") as sample, \
                 mock.patch.object(cli.safety, "require_host_namespace") as namespace, \
                 mock.patch.object(cli, "_execute", return_value=outcome) as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli._run_locked(args), 3)
            sample.assert_not_called(); namespace.assert_not_called()
            self.assertIsNone(execute.call_args.args[-1])
            audit = json.loads((root / "out/audit.json").read_text())
            self.assertEqual(audit["status"], "non_comparable")
            self.assertFalse((root / "out/results.csv").exists())

    def test_export_validation_failure_is_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            library, header = root / "libmccl.so", root / "mccl.h"
            library.touch(); header.touch()
            args = self.arguments("--sizes", "1MiB", "--output", str(root / "out"),
                                  "--mccl-library", str(library), "--mccl-header", str(header))
            args.dry_run = False
            outcome = {"exit_code": 0, "monitor_error": None, "interrupted": False,
                       "timed_out": False, "cleanup_error": None}
            with mock.patch.object(cli.safety, "require_host_namespace"), \
                 mock.patch.object(cli.safety, "resolve_gmi", return_value="/example/mthreads-gmi"), \
                 mock.patch.object(cli.safety, "sample", return_value={"status": "clean"}), \
                 mock.patch.object(cli.safety, "diagnostic_queries", return_value={}), \
                 mock.patch.object(cli, "_execute", return_value=outcome), \
                 mock.patch("musacomm.exporter.export_results", side_effect=ValueError("missing rank")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli._run_locked(args), 2)
            audit = json.loads((root / "out/audit.json").read_text())
            self.assertEqual(audit["status"], "export_validation_failed")

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "requires Linux /proc")
    def test_monitor_failure_interrupt_and_timeout_cleanup_own_group(self):
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        for failure in (RuntimeError("telemetry failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as root:
                args = self.arguments()
                args.launcher_timeout = 1
                with mock.patch.object(cli.safety, "sample", side_effect=failure):
                    outcome = cli._execute(args, command, Path(root), list(range(8)), "/example/mthreads-gmi")
                self.assertIsNone(outcome["cleanup_error"])
                self.assertIsNotNone(outcome["exit_code"])
                self.assertEqual(outcome["interrupted"], isinstance(failure, KeyboardInterrupt))
        with tempfile.TemporaryDirectory() as root:
            args = self.arguments(); args.launcher_timeout = .05
            outcome = cli._execute(args, command, Path(root), list(range(8)), None)
            self.assertTrue(outcome["timed_out"])
            self.assertIsNone(outcome["cleanup_error"])
            self.assertIsNotNone(outcome["exit_code"])


if __name__ == "__main__":
    unittest.main()
