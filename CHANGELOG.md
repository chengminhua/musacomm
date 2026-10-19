# Changelog

## 0.1.0 — 2026-10-19

- Extract the single-node communication launcher into an installable `musacomm`
  package with a console entry point and CPU-only planning.
- Retain MCCL-backed collective/P2P execution, deterministic cases, full-rank
  correctness checks and separate device / host-completion measurements.
- Make SDK library and header paths configurable.
- Add CPU tests, packaging CI, publication checks and usage/methodology docs.
