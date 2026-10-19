# musacomm

[简体中文](README.md)

A **MUSA communication benchmark** for Linux with **eight GPUs on one host**.
Generate a deterministic test plan, run MCCL collectives and point-to-point
transfers, verify every rank, and export timing and bandwidth results.

## Install and plan (no GPU needed)

```bash
python -m pip install --no-deps .
musacomm --help
musacomm allreduce --dry-run --sizes 1MiB,16MiB,1GiB --output plan-allreduce
```

Python 3.10+ is required. For GPU execution, install a compatible MUSA driver,
MCCL library and header, PyTorch, and torch_musa. Select the SDK with
`--musa-home` / `MUSA_HOME` or explicit `--mccl-library` and `--mccl-header` paths.

## Run on an exclusively reserved host

```bash
musacomm allreduce --dtype bf16 --reduce-op sum --output results/allreduce
musacomm alltoall --sizes 1MiB,64MiB,1GiB --output results/alltoall
musacomm sendrecv --src 0 --dst 1 --direction bidirectional --output results/p2p
```

Reserve the whole host through your scheduler first. Preflight checks GPU
availability; a local lock coordinates musacomm processes sharing that lock file.
Normal mode runs on the host and stops if GPU state is busy or cannot be confirmed.
Container diagnostics use `--skip-preflight` and are recorded as `non_comparable`.

Supported CLI primitives: AllReduce, AllGather, ReduceScatter, Broadcast,
Reduce, AllToAll, AllToAllv, Gather, Scatter and Send/Recv. AllToAll/Gather/Scatter
use grouped Send/Recv. AllToAllv currently exposes an equal route with
grouped/torch/native paths; native availability is runtime-dependent.
Available data types are BF16, FP16 and FP32. Each run checks the selected
operation, data type and runtime combination for correctness.

## Measurements and outputs

Default minimums: 20 warmups, five blocks, 200 ms per block. The runner uses
independent CPU control and full-rank correctness checks, reports device-event
and host-completion timing separately, and does not count validation work as
communication time. `algbw` is logical payload throughput; `busbw` applies the
collective's transfer-volume factor to that throughput.

Output includes an immutable manifest, configuration, preflight, rank journals,
launcher log and audit. The terminal prints the audit as JSON; `results.csv` holds
the per-case measurements. Normal CSV export requires a clean run and validated
results from all eight ranks; diagnostic overrides are recorded separately.
An existing output directory causes the command to stop, so choose a new directory
for each run. Review hostnames, paths and process details in the artifacts before
sharing them.

## Contributing

Repository skills: [usage](skills/musacomm-usage/SKILL.md) and
[development](skills/musacomm-development/SKILL.md). Start with [AGENTS.md](AGENTS.md)
when working with an AI coding agent in this checkout.

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/check_public_tree.py
```

CI performs CPU checks and package builds. See [usage](docs/usage.md),
[methodology](docs/methodology.md), [development](docs/development.md),
[security](SECURITY.md), [runtime dependencies](docs/third-party.md), and the
[validation record](docs/validation.md).

License: [MIT](LICENSE).
