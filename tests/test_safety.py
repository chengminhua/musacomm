import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from musacomm import safety


DEVICE = {"Index": "0", "Product Name": "Example MUSA GPU",
          "FB Memory Usage": {"Used": "0MiB", "Total": "81920MiB"},
          "Utilization": {"Gpu": "0%"}}
HEADER = "Processes:\nID PID Process name GPU Memory\nUsage\n+-----------------+\n"
EMPTY = HEADER + "No running processes found\n-------------------\n"
BUSY = HEADER + "0 23456 /usr/bin/python3 train.py --secret omitted 128MiB\n-------------------\n"


def device_json(device=None):
    return json.dumps({"Attached GPUs": "1", "GPU": [device or DEVICE]})


class SafetyTests(unittest.TestCase):
    def test_device_inventory(self):
        result = safety.parse_devices(device_json())
        self.assertEqual(result[0]["memory_used_mib"], 0)
        self.assertEqual(result[0]["memory_total_mib"], 81920)
        self.assertEqual(result[0]["utilization"], 0)

    def test_device_critical_fields_fail_closed(self):
        for section, key, value in (("FB Memory Usage", "Used", "N/A"),
                                    ("FB Memory Usage", "Total", "0MiB"),
                                    ("Utilization", "Gpu", "nan"), ("Utilization", "Gpu", "101%")):
            changed = copy.deepcopy(DEVICE); changed[section][key] = value
            with self.subTest(value=value), self.assertRaises(safety.SafetyError):
                safety.parse_devices(device_json(changed))
        for value in ("", "[]", "{}", '{"GPU": []}'):
            with self.subTest(value=value), self.assertRaises(safety.SafetyError):
                safety.parse_devices(value)

    def test_pid_is_second_column_not_gpu_id(self):
        rows = safety.parse_processes(BUSY, {0})
        self.assertEqual(rows[0]["pid"], 23456)
        self.assertEqual(rows[0]["gpu"], 0)
        self.assertEqual(rows[0]["executable"], "python3")
        self.assertNotIn("secret", repr(rows))
        wrapped = "\n".join("| " + line + " |" for line in BUSY.splitlines())
        self.assertEqual(safety.parse_processes(wrapped, {0})[0]["pid"], 23456)

    def test_empty_process_table_must_be_explicit(self):
        self.assertEqual(safety.parse_processes(EMPTY, {0}), [])
        invalid = ("", HEADER, "Processes:\nNo running processes found", BUSY + "garbage\n",
                   BUSY + "No running processes found\n", BUSY.replace("128MiB", "N/A"),
                   BUSY.replace("0 23456", "0 0"), BUSY.replace("0 23456", "7 23456"))
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(safety.SafetyError):
                safety.parse_processes(text, {0})

    def test_sample_command_failure_never_becomes_idle(self):
        with mock.patch.object(safety.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            with self.assertRaisesRegex(safety.SafetyError, "query failed"):
                safety.sample("/example/mthreads-gmi", [0])

    def test_sample_memory_busy_without_process(self):
        changed = copy.deepcopy(DEVICE); changed["FB Memory Usage"]["Used"] = "1MiB"
        values = [subprocess.CompletedProcess([], 0, device_json(changed), ""),
                  subprocess.CompletedProcess([], 0, EMPTY, "")]
        with mock.patch.object(safety.subprocess, "run", side_effect=values):
            with self.assertRaisesRegex(safety.SafetyError, "nonzero"):
                safety.sample("/example/mthreads-gmi", [0])

    def test_sample_only_own_gpu_pid_is_allowed_during_run(self):
        values = [subprocess.CompletedProcess([], 0, device_json(), ""),
                  subprocess.CompletedProcess([], 0, BUSY, "")]
        with mock.patch.object(safety.subprocess, "run", side_effect=values):
            self.assertEqual(safety.sample("/example/mthreads-gmi", [0], own_pids={23456})["status"], "clean")
        with mock.patch.object(safety.subprocess, "run", side_effect=values):
            with self.assertRaisesRegex(safety.SafetyError, "foreign GPU"):
                safety.sample("/example/mthreads-gmi", [0], own_pids={0})

    def test_allow_busy_does_not_allow_missing_telemetry(self):
        values = [subprocess.CompletedProcess([], 0, device_json(), ""),
                  subprocess.CompletedProcess([], 0, "", "")]
        with mock.patch.object(safety.subprocess, "run", side_effect=values):
            with self.assertRaises(safety.SafetyError):
                safety.sample("/example/mthreads-gmi", [0], allow_busy=True)

    def test_busy_device_without_attributable_process_is_unknown(self):
        changed = copy.deepcopy(DEVICE); changed["Utilization"]["Gpu"] = "25%"
        values = [subprocess.CompletedProcess([], 0, device_json(changed), ""),
                  subprocess.CompletedProcess([], 0, EMPTY, "")]
        with mock.patch.object(safety.subprocess, "run", side_effect=values):
            with self.assertRaisesRegex(safety.SafetyError, "no attributable"):
                safety.sample("/example/mthreads-gmi", [0], own_pids={23456})
        with mock.patch.object(safety.subprocess, "run", side_effect=values):
            result = safety.sample("/example/mthreads-gmi", [0], own_pids={23456}, allow_busy=True)
            self.assertEqual(result["status"], "diagnostic")
            self.assertIn("+00:00", result["timestamp_utc"])

    def test_container_is_not_assumed_same_pid_namespace(self):
        with mock.patch.object(safety.Path, "exists", return_value=True):
            with self.assertRaisesRegex(safety.SafetyError, "container PID"):
                safety.require_host_namespace()

    @unittest.skipUnless(os.name == "posix" and os.path.isdir("/proc"), "requires Linux /proc")
    def test_cleanup_terminates_owned_group_including_child(self):
        child_script = "import time; time.sleep(30)"
        parent_script = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_script!r}]); time.sleep(30)"
        proc = subprocess.Popen([sys.executable, "-c", parent_script], start_new_session=True)
        try:
            deadline = time.monotonic() + 2
            while len(safety.group_members(proc.pid)) < 2 and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertGreaterEqual(len(safety.group_members(proc.pid)), 2)
            safety.stop_process_group(proc, grace_seconds=.2)
            self.assertIsNotNone(proc.poll())
            self.assertEqual(safety.group_members(proc.pid), set())
        finally:
            if proc.poll() is None:
                safety.stop_process_group(proc, grace_seconds=.2)


if __name__ == "__main__":
    unittest.main()
