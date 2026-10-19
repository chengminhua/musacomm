"""Independent append-audited case statuses. Never accesses resource databases."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .catalog import canonical

STATUSES = {"planned", "pending", "running", "passed", "failed_correctness", "timeout",
            "runtime_error", "unsupported", "invalid_combination", "capacity_limited", "contaminated"}
CAPABILITIES = {"unverified", "supported", "unsupported"}
PLATFORMS = {"confirmed", "candidate_unconfirmed", "not_reached", "not_applicable"}
OFFICIAL = {"matched", "below_reference", "incomparable", "reference_unverified"}
CASE_ID = re.compile(r"^(collective|moe|p2p)-[0-9a-f]{20}$")


class StateStore:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        if not CASE_ID.fullmatch(case_id):
            raise ValueError("invalid case_id; paths are not accepted")
        return self.directory / f"{case_id}.json"

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.directory / ".lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _commit(self, state: dict[str, Any], *, event_type: str) -> None:
        """Snapshot is authoritative; history may miss the last crash-time event.

        Atomic rename prevents half-written snapshots. An event sequence number
        makes any crash gap auditable; this is not a distributed execution lock.
        """
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        event = {"event": event_type, **state}
        fd, temp_name = tempfile.mkstemp(prefix=".state-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(canonical(state) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self._path(state["case_id"]))
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def register(self, case: dict[str, Any]) -> dict[str, Any]:
        with self._lock():
            path = self._path(case["case_id"])
            if path.exists():
                current = json.loads(path.read_text(encoding="utf-8"))
                if current["case_hash"] != case["case_hash"]:
                    raise ValueError("case_id collision with a different full hash")
                return current
            state = {"case_id": case["case_id"], "case_hash": case["case_hash"],
                     "status": case["status"], "sequence": 0,
                     "capability_status": case["capability"]["status"],
                     "platform_status": case["platform_status"], "official_status": case["official_status"],
                     "reason": "; ".join(case.get("invalid_reasons", [])), "evidence": {}}
            self._commit(state, event_type="registered")
            return state

    def get(self, case_id: str) -> dict[str, Any]:
        return json.loads(self._path(case_id).read_text(encoding="utf-8"))

    def update(self, case_id: str, *, status: str | None = None,
               capability_status: str | None = None, platform_status: str | None = None,
               official_status: str | None = None, reason: str = "",
               evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock():
            state = self.get(case_id)
            next_state = copy.deepcopy(state)
            for key, value, choices in (
                ("status", status, STATUSES), ("capability_status", capability_status, CAPABILITIES),
                ("platform_status", platform_status, PLATFORMS), ("official_status", official_status, OFFICIAL),
            ):
                if value is not None:
                    if value not in choices:
                        raise ValueError(f"invalid {key}: {value}")
                    next_state[key] = value
            supplied = copy.deepcopy(evidence or {})
            merged = {**state["evidence"], **supplied}
            new_status = next_state["status"]
            if capability_status is not None and capability_status != "unverified":
                if not supplied.get("capability_evidence"):
                    raise ValueError("support classification requires capability_evidence")
            if status == "running":
                if next_state["capability_status"] != "supported":
                    raise ValueError("cannot mark running while capability is unverified")
                if not supplied.get("leases") or not supplied.get("pids"):
                    raise ValueError("running requires current lease and PID evidence")
            if status == "passed":
                if next_state["capability_status"] != "supported":
                    raise ValueError("passed requires explicitly verified capability")
                if supplied.get("correctness_passed") is not True:
                    raise ValueError("passed requires fresh correctness_passed=true evidence")
                if not isinstance(supplied.get("valid_samples"), int) or supplied["valid_samples"] <= 0:
                    raise ValueError("passed requires a positive valid_samples count")
                if not supplied.get("results"):
                    raise ValueError("passed requires raw result references")
            if status == "unsupported":
                if next_state["capability_status"] != "unsupported" or not supplied.get("capability_evidence"):
                    raise ValueError("unsupported needs direct interface/version evidence, not just failure")
            if status in {"invalid_combination", "capacity_limited", "failed_correctness", "timeout", "runtime_error", "contaminated"}:
                if not reason or not supplied:
                    raise ValueError(f"{status} requires a reason and evidence")
            if platform_status == "confirmed":
                if new_status != "passed" or not supplied.get("platform_evidence"):
                    raise ValueError("platform confirmation needs passed performance and platform evidence")
            if official_status in {"matched", "below_reference"} and not supplied.get("official_comparison_evidence"):
                raise ValueError("official comparison needs same-scope reference evidence")
            next_state.update(reason=reason or state["reason"], evidence=merged, sequence=state["sequence"] + 1)
            self._commit(next_state, event_type="updated")
            return next_state

    def summary(self) -> dict[str, Any]:
        from collections import Counter
        statuses, capabilities, platforms, official = Counter(), Counter(), Counter(), Counter()
        for path in self.directory.glob("*.json"):
            if not CASE_ID.fullmatch(path.stem):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            statuses[record["status"]] += 1
            capabilities[record["capability_status"]] += 1
            platforms[record["platform_status"]] += 1
            official[record["official_status"]] += 1
        total = sum(statuses.values())
        return {"total_registered": total, "statuses": dict(statuses), "capabilities": dict(capabilities),
                "platforms": dict(platforms), "official": dict(official),
                "performance_passed_fraction_of_registered": statuses["passed"] / total if total else None,
                "all_performance_completed": bool(total) and statuses["passed"] == total,
                "warning": "registration total is not the complete plan total; compare against catalog manifests"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("manifest", type=Path)
    update = commands.add_parser("update")
    update.add_argument("case_id")
    update.add_argument("--status", choices=sorted(STATUSES))
    update.add_argument("--capability-status", choices=sorted(CAPABILITIES))
    update.add_argument("--platform-status", choices=sorted(PLATFORMS))
    update.add_argument("--official-status", choices=sorted(OFFICIAL))
    update.add_argument("--reason", default="")
    update.add_argument("--evidence", type=Path)
    commands.add_parser("summary")
    args = parser.parse_args()
    store = StateStore(args.state_dir)
    if args.command == "register":
        count = 0
        with args.manifest.open(encoding="utf-8") as stream:
            for line in stream:
                store.register(json.loads(line))
                count += 1
        print(json.dumps({"registered_or_existing": count}))
    elif args.command == "summary":
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    else:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else {}
        print(json.dumps(store.update(args.case_id, status=args.status, capability_status=args.capability_status,
                         platform_status=args.platform_status, official_status=args.official_status,
                         reason=args.reason, evidence=evidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
