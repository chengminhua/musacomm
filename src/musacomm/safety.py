"""Conservative local GPU sampling and cleanup; no cluster or SSH integration.

The process table reports host PIDs.  A container can expose those PIDs while
its /proc exposes another namespace, so normal mode is deliberately host-only.
These checks detect activity; they are not a scheduler or an exclusive lease.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


class SafetyError(RuntimeError):
    """We cannot establish a trustworthy local measurement environment."""


_NUMBER = r"\d+(?:\.\d+)?"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _field(value: dict, name: str):
    for key, item in value.items():
        if str(key).strip().casefold() == name.casefold():
            return item
    return None


def _memory_mib(value) -> float:
    match = re.fullmatch(rf"({_NUMBER})\s*([KMGT]?i?B)?", str(value).strip(), re.I)
    if not match:
        raise SafetyError("GPU memory telemetry is missing or unrecognized")
    unit = (match[2] or "MiB").upper()
    power = "KMGT".index(unit[0]) + 1 if unit != "B" else 0
    result = float(match[1]) * (1024 if "I" in unit else 1000) ** power / 2**20
    if not math.isfinite(result):
        raise SafetyError("GPU memory telemetry is not finite")
    return result


def parse_devices(text: str) -> list[dict]:
    """Parse documented GMI JSON; unknown critical data is not idle."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise SafetyError("mthreads-gmi -q --json did not return JSON") from exc
    if not isinstance(data, dict):
        raise SafetyError("GPU JSON must be an object")
    entries = _field(data, "GPU") or _field(data, "GPUs")
    if not isinstance(entries, list) or not entries:
        raise SafetyError("GPU inventory is missing or empty")
    attached = _field(data, "Attached GPUs")
    if attached is not None and str(attached).strip() != str(len(entries)):
        raise SafetyError("attached GPU count disagrees with inventory")
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SafetyError("GPU inventory entry is not an object")
        raw_index = str(_field(entry, "Index")).strip()
        if not raw_index.isdecimal() or int(raw_index) in seen:
            raise SafetyError("GPU index is missing, invalid or duplicated")
        index = int(raw_index)
        seen.add(index)
        memory, utilization = _field(entry, "FB Memory Usage"), _field(entry, "Utilization")
        if not isinstance(memory, dict) or not isinstance(utilization, dict):
            raise SafetyError("GPU memory/utilization section is missing")
        used, total = _memory_mib(_field(memory, "Used")), _memory_mib(_field(memory, "Total"))
        util_match = re.fullmatch(rf"({_NUMBER})\s*%?", str(_field(utilization, "Gpu")).strip())
        if total <= 0 or used > total or not util_match or not 0 <= float(util_match[1]) <= 100:
            raise SafetyError("GPU memory/utilization is inconsistent or invalid")
        name = _field(entry, "Product Name") or _field(entry, "Name")
        if not isinstance(name, str) or not name.strip():
            raise SafetyError("GPU product name is missing")
        result.append({"index": index, "name": name.strip(), "memory_used_mib": used,
                       "memory_total_mib": total, "utilization": float(util_match[1])})
    return sorted(result, key=lambda row: row["index"])


def parse_processes(text: str, device_ids: set[int]) -> list[dict]:
    """GMI columns are GPU_ID, PID, executable, memory (not PID first)."""
    text = _ANSI.sub("", text)
    marker = re.search(r"\bProcesses\s*:", text, re.I)
    if marker is None:
        raise SafetyError("GPU process section is missing; cannot assume idle")
    header, header_seen, empty, rows, seen = "", False, False, [], set()
    for raw in text[marker.end():].splitlines():
        line = raw.strip().strip("|").strip()
        if not line or re.fullmatch(r"[+|=\-\s]+", line):
            continue
        if re.fullmatch(r"No running processes found[.!]?", line, re.I):
            empty = True
            continue
        if not header_seen:
            if re.match(r"\d+\s+\d+\b", line):
                raise SafetyError("GPU process rows have no recognized header")
            header += " " + line
            header_seen = all(re.search(rf"\b{word}\b", header, re.I)
                              for word in ("ID", "PID", "Process", "Memory"))
            continue
        if line.casefold() == "usage":
            continue
        match = re.fullmatch(rf"(\d+)\s+(\d+)\s+(.+?)\s+({_NUMBER}\s*[KMGT]?i?B)", line, re.I)
        if match is None:
            raise SafetyError("GPU process table has an unknown row")
        gpu, pid = int(match[1]), int(match[2])
        if gpu not in device_ids or pid <= 0 or (gpu, pid) in seen:
            raise SafetyError("GPU process table has an invalid or duplicate GPU/PID")
        seen.add((gpu, pid))
        # Arguments can contain secrets. Retain only an executable basename.
        executable = Path(match[3].split()[0].strip("\"'")).name
        rows.append({"gpu": gpu, "pid": pid, "executable": executable,
                     "memory_mib": _memory_mib(match[4])})
    if not header_seen or (not empty and not rows) or (empty and rows):
        raise SafetyError("GPU process table is incomplete or contradictory")
    return rows


def require_host_namespace() -> None:
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        raise SafetyError("container PID/device mapping cannot be verified; run on the host or use "
                          "--skip-preflight for explicitly non-comparable diagnostics")
    try:
        status = Path("/proc/self/status").read_text()
    except OSError as exc:
        raise SafetyError("cannot verify the local PID namespace") from exc
    nspid = re.search(r"^NSpid:\s*(.+)$", status, re.M)
    if nspid is None or len(nspid[1].split()) != 1:
        raise SafetyError("unknown or nested PID namespace; use the host for normal measurements")


def resolve_gmi(value: str | None = None) -> str:
    found = shutil.which(value or "mthreads-gmi")
    if found is None:
        raise SafetyError("mthreads-gmi is unavailable; --skip-preflight is diagnostic-only")
    return str(Path(found).resolve(strict=True))


def sample(gmi: str, selected: list[int], *, allow_busy: bool = False,
           own_pids: set[int] | None = None) -> dict:
    outputs = []
    for command in ([gmi, "-q", "--json"], [gmi]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode != 0:
            raise SafetyError(f"{Path(gmi).name} query failed with exit {result.returncode}")
        outputs.append(result.stdout)
    devices = parse_devices(outputs[0])
    if not set(selected).issubset({row["index"] for row in devices}):
        raise SafetyError("selected physical GPU IDs are missing from GMI inventory")
    processes = parse_processes(outputs[1], {row["index"] for row in devices})
    selected_devices = [row for row in devices if row["index"] in selected]
    # Entire host activity is relevant to a formal bandwidth measurement.
    foreign = [row for row in processes if row["pid"] not in (own_pids or set())]
    busy = [row["index"] for row in devices
            if row["memory_used_mib"] > 0 or row["utilization"] > 0]
    if foreign and not allow_busy:
        raise SafetyError("foreign GPU processes detected: " + ",".join(str(row["pid"]) for row in foreign))
    # During our run memory/utilization is expected; process ownership is checked.
    if own_pids is None and busy and not allow_busy:
        raise SafetyError(f"GPU memory or utilization is nonzero on devices {busy}")
    if own_pids is not None and not allow_busy:
        attributable = {row["gpu"] for row in processes if row["pid"] in own_pids}
        unexplained = sorted(set(busy) - attributable)
        if unexplained:
            raise SafetyError(f"busy GPUs have no attributable benchmark process: {unexplained}; "
                              "GPU PID visibility or namespace mapping may be inconsistent")
    return {"status": "diagnostic" if allow_busy else "clean", "gmi_executable": gmi,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "devices": devices, "selected_devices": selected_devices,
            "processes": processes, "foreign_gpu_pids": sorted({row["pid"] for row in foreign}),
            "busy_device_ids": busy}


def diagnostic_queries(gmi: str) -> dict:
    result = {}
    for name, tail in (("topology", ["topo", "-m"]), ("mtlink", ["mtlink", "-s"])):
        try:
            query = subprocess.run([gmi, *tail], capture_output=True, text=True, timeout=30, check=False)
            result[name] = {"returncode": query.returncode, "stdout": query.stdout, "stderr": query.stderr}
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = {"error": str(exc)}
    return result


def _process_rows() -> list[tuple[int, int, int, int, str]]:
    rows = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            rows.append((int(entry.name), int(fields[1]), int(fields[2]), int(fields[3]), fields[0]))
        except (OSError, ValueError, IndexError):
            continue
    return rows


def process_tree_pids(root_pid: int) -> set[int]:
    rows, children = _process_rows(), {root_pid}
    while True:
        found = {pid for pid, parent, _, _, state in rows if parent in children and state != "Z"}
        if found.issubset(children):
            return children
        children.update(found)


def group_members(group: int) -> set[int]:
    return {pid for pid, _, pgid, session, state in _process_rows()
            if pgid == group and session == group and state != "Z"}


def stop_process_group(proc: subprocess.Popen, grace_seconds: float = 5) -> None:
    """Terminate only the new session/process group started by this launcher."""
    if proc.pid == os.getpgrp():
        raise SafetyError("refusing to signal the caller process group")
    if group_members(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while group_members(proc.pid) and time.monotonic() < deadline:
        proc.poll()
        time.sleep(0.05)
    if group_members(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise SafetyError("launcher did not exit after process-group termination") from exc
    deadline = time.monotonic() + 5
    while group_members(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if group_members(proc.pid):
        raise SafetyError("benchmark descendants remain; retain any external reservation")
