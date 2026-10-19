# Security and data handling

Reserve the entire host through your site's scheduler before running a formal
multi-GPU test. When using a container, confirm host-side GPU availability and
the physical GPU mapping before launching the benchmark.

Do not run untrusted code, manifests, shared libraries or headers. The MCCL
library is native executable code loaded into the process. Use only trusted SDK
installations. Do not execute GPU CI on public pull requests with access to
production hosts or secrets.

Measurement outputs can contain hostnames, absolute paths, GPU process details,
command lines and library fingerprints. Review artifacts before sharing and
redact private identifiers, paths and command arguments. Keep raw results outside
the source tree.

If GitHub private vulnerability reporting is enabled for the published repository,
use its **Security → Report a vulnerability** entry. Otherwise contact a maintainer
privately using a contact they explicitly provide. Keep credentials and
sensitive exploit details out of public issues.
