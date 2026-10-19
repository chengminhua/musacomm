# Contributing

Start with [development notes](docs/development.md). Use Python 3.10 or newer.
CPU-only tests must remain usable without torch, torch_musa, MCCL, or a GPU.

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/check_public_tree.py
```

For a benchmark change, include the payload convention, dtype, reduction, buffer
layout, timing domain, synchronization, warmup, repetitions, correctness checks
and implementation path. Identify native and composite implementations explicitly.
Keep full-rank validation and stop tests when resource availability cannot be
confirmed. Run device tests on an authorized, exclusively reserved host. CI runs
the CPU tests and package build.

Before submitting, run the publication check and review the diff for credentials,
private hostnames, addresses and raw lab logs. Submit source code and tests;
install SDK libraries and headers through the runtime setup. Contributions use
the [MIT license](LICENSE). Preserve license and attribution information when
adding third-party material.
