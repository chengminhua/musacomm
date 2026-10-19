#!/usr/bin/env python3
"""Remove local uid/gid, user/group names and timestamps from a built sdist.

Only regular files and directories are accepted. File contents are unchanged.
This normalizes archive metadata; it does not redact sensitive file contents.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


def normalize(path: Path) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.name.endswith(".tar.gz"):
        raise ValueError("expected a regular source .tar.gz archive")
    fd, temporary = tempfile.mkstemp(prefix=".normalized-", suffix=".tar.gz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw, tarfile.open(path, "r:gz") as source:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
                    for entry in source:
                        name = PurePosixPath(entry.name)
                        if name.is_absolute() or ".." in name.parts or not (entry.isfile() or entry.isdir()):
                            raise ValueError("sdist contains an unsafe path or non-regular entry")
                        clean = copy.copy(entry)
                        clean.uid = clean.gid = clean.mtime = 0
                        clean.uname = clean.gname = ""
                        clean.pax_headers = {key: value for key, value in entry.pax_headers.items()
                                             if key not in {"uid", "gid", "uname", "gname", "mtime", "atime", "ctime"}}
                        contents = source.extractfile(entry) if entry.isfile() else None
                        try:
                            target.addfile(clean, contents)
                        finally:
                            if contents is not None:
                                contents.close()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args(argv)
    for path in args.archives:
        normalize(path)
        print(f"Normalized source archive metadata: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
