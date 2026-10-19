"""Single-node MUSA communication benchmarks with auditable correctness gates.

Version 0.1 uses eight ranks (one per GPU). Reserve the entire machine through
your own scheduler before measuring. The local safety checks are not a lease.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from . import __version__
from .catalog import catalog, schema
from . import safety


TUNING_KEYS = (
    "MCCL_ALGO", "MCCL_ALGOS", "MCCL_PROTO", "MCCL_PROTOS", "MCCL_P2P_LEVEL",
    "MCCL_P2P_DISABLE", "MCCL_DEBUG", "MCCL_DEBUG_SUBSYS", "MCCL_MIN_NCHANNELS",
    "MCCL_MAX_NCHANNELS", "MCCL_NET_SHARED_BUFFERS", "MCCL_IB_GID_INDEX",
)
PRIMITIVE_ALIASES = {
    "allreduce": "all_reduce", "all_reduce": "all_reduce",
    "allgather": "all_gather", "all_gather": "all_gather",
    "reducescatter": "reduce_scatter", "reduce_scatter": "reduce_scatter",
    "broadcast": "broadcast", "reduce": "reduce",
    "alltoall": "all_to_all", "all_to_all": "all_to_all",
    "alltoallv": "all_to_all_v", "all_to_all_v": "all_to_all_v",
    "gather": "gather", "scatter": "scatter", "sendrecv": "send_recv", "send_recv": "send_recv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", value.lower())
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}")
    units = {"b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
             "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}
    size = Decimal(match[1]) * units[match[2] or "b"]
    if not size.is_finite() or size != int(size) or not 0 < size <= 2**63 - 1:
        raise argparse.ArgumentTypeError("size must be a positive integral byte count below 2^63")
    return int(size)


def parse_gpus(value: str) -> list[int]:
    if not re.fullmatch(r"\d+(?:\s*,\s*\d+){7}", value.strip()):
        raise argparse.ArgumentTypeError("v0.1 requires exactly eight comma-separated physical GPU IDs")
    gpus = [int(item) for item in value.split(",")]
    if len(set(gpus)) != 8:
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return gpus


def size_points(args) -> list[int]:
    if args.sizes:
        values = sorted(set(parse_bytes(value) for value in args.sizes.split(",")))
    else:
        minimum, maximum = parse_bytes(args.min_size), parse_bytes(args.max_size)
        if minimum > maximum:
            raise ValueError("--min-size cannot exceed --max-size")
        values, point = [], minimum
        while point <= maximum:
            values.append(point)
            point *= 2
        if values[-1] != maximum:
            values.append(maximum)
    original = len(values)
    if args.limit is not None:
        values = values[:args.limit]
    args.size_selection = {"requested_points": original, "selected_points": len(values)}
    return values


def make_manifest(args, host_id: str, gpus: list[int], environment_id: str) -> list[dict]:
    primitive = PRIMITIVE_ALIASES[args.primitive]
    requested = set(size_points(args))
    mapping = {"A8": [{"rank": rank, "node": "A", "physical_gpu": gpu, "host": host_id}
                       for rank, gpu in enumerate(gpus)]}
    roots = range(8) if args.roots == "all" else [int(args.roots)]
    operations = catalog.REDUCE_OPS if args.reduce_op == "all" else [args.reduce_op]
    default_backend = "grouped" if primitive in {"all_to_all", "all_to_all_v", "gather", "scatter"} else "native"
    backend = args.backend or default_backend
    if primitive != "all_to_all_v" and backend != default_backend:
        raise ValueError(f"{primitive} supports only --backend {default_backend} in v0.1")
    backend_name = {"native": "mccl_native", "torch": "torch_musa_mccl", "grouped": "mccl_grouped_send_recv"}[backend]
    route = next(row for row in catalog.variable_routes(8) if row["id"] == "equal") if primitive == "all_to_all_v" else None
    sizes = list(catalog.variable_sizes(route, 8, args.dtype) if route else catalog.fixed_sizes(primitive, 8, args.dtype))
    supported = {size["actual"] for size in sizes}
    missing = requested - supported
    if missing:
        raise ValueError(f"unsupported exact byte sizes {sorted(missing)}; v0.1 uses catalog sizes "
                         "(powers of two up to 1GiB plus legal small-element cases); no sizes were silently omitted")
    rows = []
    for operation in operations if primitive in catalog.REDUCTION else [None]:
        for root in roots if primitive in catalog.ROOTED else [None]:
            spec = catalog.new_spec("p2p" if primitive == "send_recv" else "collective", primitive, "A8", args.dtype,
                                    mappings=mapping, environment_id=environment_id)
            spec.update(reduce_op=operation, root=root, buffer_mode=args.buffer_mode, route=route, backend=backend_name)
            if primitive == "send_recv":
                spec.update(participants=[args.src, args.dst], direction=args.direction, metric=f"p2p_{args.direction}")
            for size in sizes:
                if size["actual"] in requested:
                    spec["size"] = size if route else catalog.fixed_size_for_spec(size, spec)
                    row = catalog.build_case(spec)
                    schema.assert_valid_case(row)
                    rows.append(row)
    rows.sort(key=lambda row: (row["size"]["actual"], row["case_id"]))
    original_count = len(rows)
    if original_count > args.max_cases:
        if not args.allow_truncate:
            raise ValueError(f"selection has {original_count} cases; raise --max-cases or explicitly use --allow-truncate")
        rows = rows[:args.max_cases]
    args.selection = {"selected_before_limit": original_count, "selected_after_limit": len(rows),
                      "truncated": original_count != len(rows)}
    return rows


def tuned_env(args, gpus: list[int]) -> dict[str, str]:
    env = os.environ.copy()
    for key in TUNING_KEYS:
        env.pop(key, None)
    env["MUSA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    env["MUSA_HOME"] = args.musa_home
    directories = [str(Path(args.mccl_library).resolve().parent), str(Path(args.musa_home) / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        directories.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(directories))
    # Do not modify PATH or insert a shim around the telemetry executable.
    env["MCCL_DEBUG"] = {"off": "WARN", "info": "INFO", "trace": "TRACE"}[args.debug]
    if args.debug != "off":
        env["MCCL_DEBUG_SUBSYS"] = "ENV,GRAPH,TUNING,COLL,P2P"
    if args.algo != "auto":
        if args.mccl_env_style == "legacy":
            env["MCCL_ALGOS"] = {"tree": "0", "ring": "1", "collnet": "2"}[args.algo]
        else:
            env["MCCL_ALGO"] = {"tree": "Tree", "ring": "Ring", "collnet": "CollNet"}[args.algo]
    if args.proto != "auto":
        if args.mccl_env_style == "legacy":
            env["MCCL_PROTOS"] = {"ll": "0", "simple": "2"}[args.proto]
        else:
            env["MCCL_PROTO"] = {"ll": "LL", "simple": "Simple"}[args.proto]
    if args.p2p != "auto":
        env["MCCL_P2P_LEVEL"] = "MTL" if args.p2p == "mtl" else "LOC"
    if args.p2p == "off":
        env["MCCL_P2P_DISABLE"] = "1"
    return env


def runner_command(args, manifest: Path, output: Path, environment_id: str, host_id: str,
                   gpus: list[int], evidence: str) -> list[str]:
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
               "--module", "musacomm.runtime.collective", "--cases", str(manifest),
               "--outdir", str(output / "results"), "--environment-id", environment_id,
               "--host-id", host_id, "--reservation-evidence", evidence,
               "--physical-gpu-map", ",".join(map(str, gpus)), "--mccl-library", args.mccl_library,
               "--mccl-header", args.mccl_header, "--alltoallv-path", args.backend or "grouped",
               "--warmup", str(args.warmup), "--blocks", str(args.blocks),
               "--block-seconds", str(args.block_seconds), "--pilot-iterations", str(args.pilot_iterations)]
    if PRIMITIVE_ALIASES[args.primitive] in {"all_to_all", "gather", "scatter"}:
        command += ["--missing-native", "grouped"]
    return command


def _execute(args, command: list[str], output: Path, gpus: list[int], gmi: str | None) -> dict:
    code, failure, interrupted, timed_out, cleanup_error = None, None, False, False, None
    with contextlib.ExitStack() as stack:
        log = stack.enter_context((output / "launcher.log").open("x", encoding="utf-8"))
        monitoring = stack.enter_context((output / "monitoring.jsonl").open("x", encoding="utf-8")) if gmi else None
        proc = subprocess.Popen(command, cwd=output, env=tuned_env(args, gpus), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.launcher_timeout
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if gmi is not None:
                    sample = safety.sample(gmi, gpus, allow_busy=args.allow_busy,
                                           own_pids=safety.process_tree_pids(proc.pid) | safety.group_members(proc.pid))
                    monitoring.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    monitoring.flush()
                try:
                    proc.wait(timeout=min(2, max(.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            code = proc.poll()
        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if monitoring is not None:
                monitoring.write(json.dumps({"status": "failed", "error": failure,
                                             "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
                monitoring.flush()
        finally:
            # Also clean descendants after a launcher that exits prematurely.
            try:
                safety.stop_process_group(proc)
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
            code = proc.returncode if code is None else code
    return {"exit_code": code, "monitor_error": failure, "interrupted": interrupted,
            "timed_out": timed_out, "cleanup_error": cleanup_error}


def _run_locked(args) -> int:
    gpus = parse_gpus(args.gpus)
    host_id = args.host_id or socket.gethostname()
    library = Path(args.mccl_library).resolve(strict=not args.dry_run)
    header = Path(args.mccl_header).resolve(strict=not args.dry_run)
    args.mccl_library, args.mccl_header = str(library), str(header)
    if not args.dry_run and (not library.is_file() or not header.is_file()):
        raise ValueError("MCCL library/header must be files")
    fingerprint = str(library) if args.dry_run else sha256(library)
    environment_id = args.environment_id or "musacomm-single-" + hashlib.sha256((host_id + fingerprint).encode()).hexdigest()[:16]
    rows = make_manifest(args, host_id, gpus, environment_id)
    output = Path(args.output or f"musacomm-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}").resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {output}")
    gmi, warnings = None, []
    if args.dry_run:
        preflight = {"status": "dry_run", "selected_physical_gpus": gpus}
    elif args.skip_preflight:
        warnings.append("All GPU preflight, live monitoring and postflight checks explicitly disabled; diagnostic only")
        preflight = {"status": "skipped", "selected_physical_gpus": gpus}
    else:
        safety.require_host_namespace()
        gmi = safety.resolve_gmi(args.gmi_bin)
        preflight = safety.sample(gmi, gpus, allow_busy=args.allow_busy)
        preflight.update(safety.diagnostic_queries(gmi))
    if args.allow_busy:
        warnings.append("--allow-busy permits contention; results are non-comparable diagnostics")
    if args.algo != "auto" or args.proto != "auto":
        warnings.append("MCCL overrides are requests, not proof of the selected algorithm/protocol; verify vendor logs")
    preflight.setdefault("timestamp_utc", dt.datetime.now(dt.timezone.utc).isoformat())
    output.mkdir(parents=True)
    manifest = output / "cases.jsonl"
    with manifest.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "preflight.json", preflight)
    evidence = f"local-observation-not-reservation:{sha256(output / 'preflight.json')}"
    command = runner_command(args, manifest, output, environment_id, host_id, gpus, evidence)
    configuration = {"version": __version__, "argv": command, "settings": vars(args), "warnings": warnings,
                     "manifest_sha256": sha256(manifest), "environment_id": environment_id,
                     "host_id": host_id, "selected_physical_gpus": gpus,
                     "environment": {key: value for key, value in tuned_env(args, gpus).items()
                                     if key.startswith("MCCL_") or key in {"MUSA_VISIBLE_DEVICES", "MUSA_HOME"}},
                     "selection": args.selection, "size_selection": args.size_selection}
    write_json(output / "config.json", configuration)
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "output": str(output), "cases": len(rows),
                          "command": command, "warnings": warnings}, ensure_ascii=False, indent=2))
        return 0
    outcome = _execute(args, command, output, gpus, gmi)
    postflight = {"status": "skipped"}
    if gmi is not None and not outcome["cleanup_error"]:
        try:
            postflight = safety.sample(gmi, gpus, allow_busy=args.allow_busy)
        except Exception as exc:
            postflight = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    audit = {**outcome, "status": "failed", "cases": len(rows), "warnings": warnings,
             "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "postflight": postflight, "output": str(output), "manifest_sha256": sha256(manifest),
             "selection": args.selection, "size_selection": args.size_selection,
             "physical_wire_rate_measured": False, "exclusive_reservation_verified": False,
             "note": "algbw/busbw are logical payload metrics; local checks are not hardware isolation"}
    ok = outcome["exit_code"] == 0 and not any(outcome[key] for key in
                                              ("monitor_error", "cleanup_error", "interrupted", "timed_out"))
    diagnostic = args.allow_busy or args.skip_preflight
    if outcome["interrupted"]:
        audit["status"], result = "interrupted", 130
    elif outcome["timed_out"]:
        audit["status"], result = "timeout", 124
    elif not ok or (not diagnostic and postflight.get("status") != "clean"):
        result = 2
    elif diagnostic:
        audit["status"], result = "non_comparable", 3
    else:
        try:
            from .exporter import export_results
            audit["export"] = export_results(output / "results", output / "results.csv")
            audit["status"], result = "passed", 0
        except Exception as exc:
            audit["status"], result = "export_validation_failed", 2
            audit["export_error"] = f"{type(exc).__name__}: {exc}"
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def run(args) -> int:
    if args.dry_run:
        return _run_locked(args)
    lock_path = Path(os.environ.get("MUSACOMM_LOCK", "/tmp/musacomm-8gpu.lock"))
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another musacomm process holds {lock_path}") from exc
        try:
            return _run_locked(args)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musacomm", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("primitive", choices=sorted(PRIMITIVE_ALIASES))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="exactly eight host physical GPU IDs, in rank order")
    parser.add_argument("--sizes", help="exact catalog sizes, e.g. 1MiB,16MiB,1GiB")
    parser.add_argument("--min-size", default="1MiB")
    parser.add_argument("--max-size", default="1GiB")
    parser.add_argument("--limit", type=int, help="explicitly limit selected size points")
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--allow-truncate", action="store_true", help="explicitly allow partial case coverage")
    parser.add_argument("--roots", default="0", help="root rank 0..7 or all, only for rooted operations")
    parser.add_argument("--reduce-op", choices=(*catalog.REDUCE_OPS, "all"), default="sum")
    parser.add_argument("--buffer-mode", choices=("out_of_place", "in_place"), default="out_of_place")
    parser.add_argument("--backend", choices=("native", "grouped", "torch"), help="AllToAllv path; fixed AllToAll/Gather/Scatter use grouped")
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--dst", type=int, default=1)
    parser.add_argument("--direction", choices=("unidirectional", "bidirectional"), default="unidirectional")
    parser.add_argument("--algo", choices=("auto", "ring", "tree", "collnet"), default="auto")
    parser.add_argument("--proto", choices=("auto", "simple", "ll"), default="auto")
    parser.add_argument("--mccl-env-style", choices=("modern", "legacy"), default="modern",
                        help="forced overrides: modern named variables (default) or explicitly opted-in legacy numeric variables")
    parser.add_argument("--p2p", choices=("auto", "mtl", "loc", "off"), default="auto")
    parser.add_argument("--debug", choices=("off", "info", "trace"), default="off")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--block-seconds", type=float, default=.2)
    parser.add_argument("--pilot-iterations", type=int, default=5)
    parser.add_argument("--host-id")
    parser.add_argument("--environment-id")
    parser.add_argument("--musa-home", default=os.environ.get("MUSA_HOME", "/usr/local/musa"))
    parser.add_argument("--mccl-library", help="default MUSA_HOME/lib/libmccl.so")
    parser.add_argument("--mccl-header", help="default MUSA_HOME/include/mccl.h")
    parser.add_argument("--gmi-bin", help="path to the real mthreads-gmi binary; PATH is never rewritten")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true", help="write a CPU-only plan without loading torch or contacting GPUs")
    parser.add_argument("--launcher-timeout", type=float, default=43200)
    parser.add_argument("--allow-busy", action="store_true", help="diagnostic only: allow existing GPU activity; no results.csv, exit 3 on completion")
    parser.add_argument("--skip-preflight", action="store_true", help="diagnostic only: disable ALL GPU environment checks and monitoring; no results.csv, exit 3 on completion")
    return parser


def validate_args(args) -> None:
    primitive = PRIMITIVE_ALIASES[args.primitive]
    parse_gpus(args.gpus)
    if not math.isfinite(args.block_seconds) or args.blocks < 5 or args.warmup < 20 or args.block_seconds < .2:
        raise ValueError("formal timing requires --warmup >=20, --blocks >=5, finite --block-seconds >=0.2")
    if args.pilot_iterations < 1 or args.max_cases < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("--pilot-iterations, --max-cases and --limit must be positive")
    if not math.isfinite(args.launcher_timeout) or args.launcher_timeout <= 0:
        raise ValueError("--launcher-timeout must be finite and positive")
    if args.roots != "all" and (not args.roots.isdecimal() or not 0 <= int(args.roots) < 8):
        raise ValueError("--roots must be a rank in 0..7 or all")
    if primitive not in catalog.ROOTED and args.roots != "0":
        raise ValueError("--roots applies only to rooted operations")
    if primitive not in catalog.REDUCTION and args.reduce_op != "sum":
        raise ValueError("--reduce-op applies only to reductions")
    if not 0 <= args.src < 8 or not 0 <= args.dst < 8 or args.src == args.dst:
        raise ValueError("--src and --dst must be distinct ranks in 0..7")
    if primitive != "send_recv" and (args.direction != "unidirectional" or args.src != 0 or args.dst != 1):
        raise ValueError("--src, --dst and --direction apply only to sendrecv")
    if args.buffer_mode == "in_place" and primitive not in catalog.INPLACE_CANDIDATES:
        raise ValueError(f"in_place is unsupported for {primitive}")
    if args.sizes is not None and not args.sizes.strip():
        raise ValueError("--sizes cannot be empty")
    args.musa_home = str(Path(args.musa_home).expanduser().resolve())
    args.mccl_library = args.mccl_library or str(Path(args.musa_home) / "lib/libmccl.so")
    args.mccl_header = args.mccl_header or str(Path(args.musa_home) / "include/mccl.h")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"musacomm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
