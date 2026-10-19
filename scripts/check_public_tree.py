#!/usr/bin/env python3
"""Conservative, offline checks of files eligible for source publication.

This is a release checklist aid, not a guarantee that data is safe to publish.
It never modifies files and never prints the matched secret or source line.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT_FILES = {
    ".gitattributes", ".gitignore", "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE", "MANIFEST.in",
    "README.md", "README.en.md", "SECURITY.md", "pyproject.toml",
}
ROOT_DIRS = {".github", "docs", "scripts", "skills", "src", "tests"}
IGNORED_DIRS = {
    ".git", ".venv", "venv", "build", "dist", "__pycache__",
    ".pytest_cache", "htmlcov",
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".txt"}
RAW_SUFFIXES = {
    ".jsonl", ".log", ".csv", ".tsv", ".parquet", ".tar", ".gz", ".tgz",
    ".zip", ".7z", ".bz2", ".xz",
}
VENDOR_SUFFIXES = {
    ".h", ".hpp", ".cuh", ".so", ".dll", ".dylib", ".a", ".o", ".bin",
    ".deb", ".rpm", ".exe", ".whl",
}
MAX_TEXT_BYTES = 2 * 1024 * 1024
IPV4 = re.compile(r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])")
# Integer network bases avoid putting example private endpoints in this source.
PRIVATE_NETWORKS = tuple(ipaddress.IPv4Network(value) for value in (
    (0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16),
))
SENSITIVE = (
    ("user-home", re.compile(r"/(?:home|Users)/[^/\s\"'<>]+/")),
    ("private-host", re.compile(r"\bworker[0-9]{4,}\b", re.IGNORECASE)),
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
    ("credential-token", re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}"
        r"|xox[baprs]-[A-Za-z0-9-]{20,})\b")),
)
CREDENTIAL_ASSIGNMENT = re.compile(
    r"\b(?:password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)"
    r"\s*[:=]\s*(['\"])([^'\"\r\n]{8,})\1", re.IGNORECASE,
)
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    code: str
    message: str


def ignored_directory(name: str, *, at_root: bool = False) -> bool:
    return (name in IGNORED_DIRS or name.endswith(".egg-info")
            or (at_root and name.startswith(("plan", "result", "musacomm-"))))


def placeholder(value: str) -> bool:
    lowered = value.lower().strip()
    return (lowered.startswith(("<", "${", "your_", "example_", "placeholder"))
            or lowered in {"changeme", "not-a-secret", "redacted"})


def scan_text(text: str, display_path: str) -> list[Finding]:
    """Inspect text without returning matched content."""
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for match in IPV4.finditer(line):
            try:
                address = ipaddress.IPv4Address(match.group())
            except ipaddress.AddressValueError:
                continue
            if any(address in network for network in PRIVATE_NETWORKS):
                findings.append(Finding(display_path, line_number, "private-address",
                                        "private IPv4 address in publication candidate"))
                break
        for code, pattern in SENSITIVE:
            if pattern.search(line):
                findings.append(Finding(display_path, line_number, code,
                                        "potentially sensitive identifier in publication candidate"))
        if any(not placeholder(match.group(2)) for match in CREDENTIAL_ASSIGNMENT.finditer(line)):
            findings.append(Finding(display_path, line_number, "credential-assignment",
                                    "potential credential assignment; review without sharing its value"))
    return findings


def markdown_prose(text: str) -> str:
    """Keep line positions but omit fenced/inline code for link validation."""
    output = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            output.append("\n" if line.endswith("\n") else "")
        elif fence is not None:
            output.append("\n" if line.endswith("\n") else "")
        else:
            output.append(re.sub(r"`+[^`\n]*`+", "", line))
    return "".join(output)


def check_markdown_links(text: str, path: Path, root: Path) -> list[Finding]:
    findings = []
    prose = markdown_prose(text)
    for pattern in (INLINE_LINK, REFERENCE_LINK):
        for match in pattern.finditer(prose):
            raw = match.group(1).strip()
            target = raw[1:raw.index(">")] if raw.startswith("<") else raw.split()[0]
            parsed = urlsplit(target)
            # Remote links are not fetched. This check deliberately stays offline.
            if parsed.scheme and parsed.scheme != "file":
                continue
            if parsed.netloc or not parsed.path:
                continue
            relative = unquote(parsed.path)
            line = prose.count("\n", 0, match.start()) + 1
            display = path.relative_to(root).as_posix()
            if parsed.scheme == "file" or Path(relative).is_absolute():
                findings.append(Finding(display, line, "absolute-local-link",
                                        "use a repository-relative Markdown file link"))
                continue
            destination = (path.parent / relative).resolve()
            if not destination.is_relative_to(root):
                findings.append(Finding(display, line, "escaping-link",
                                        "Markdown file link escapes the repository"))
            elif not destination.exists():
                findings.append(Finding(display, line, "broken-link",
                                        "Markdown file link does not exist in the repository"))
    return findings


def candidate_files(root: Path) -> tuple[list[Path], list[Finding]]:
    files, findings = [], []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and ignored_directory(entry.name, at_root=True):
            continue
        if entry.name == ".coverage":
            continue
        if entry.is_symlink():
            findings.append(Finding(entry.name, 0, "symlink", "publication candidates must not be symlinks"))
        elif entry.is_file() and entry.name in ROOT_FILES:
            files.append(entry)
        elif entry.is_dir() and entry.name in ROOT_DIRS:
            for directory, dirnames, filenames in os.walk(entry, followlinks=False):
                base = Path(directory)
                kept = []
                for name in sorted(dirnames):
                    child = base / name
                    if child.is_symlink():
                        findings.append(Finding(child.relative_to(root).as_posix(), 0, "symlink",
                                                "publication candidates must not be symlinks"))
                    elif not ignored_directory(name):
                        kept.append(name)
                dirnames[:] = kept
                files.extend(base / name for name in sorted(filenames)
                             if not name.endswith((".pyc", ".pyo")))
        else:
            findings.append(Finding(entry.name, 0, "unexpected-root-path",
                                    "path is outside the explicit publication allowlist"))
    return files, findings


def check_tree(root: Path) -> list[Finding]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("publication root must be a directory")
    files, findings = candidate_files(root)
    for path in files:
        display = path.relative_to(root).as_posix()
        suffixes = set(path.suffixes)
        if path.is_symlink():
            findings.append(Finding(display, 0, "symlink", "publication candidates must not be symlinks"))
            continue
        if suffixes & VENDOR_SUFFIXES or path.name.startswith(("libmccl", "libmusa")):
            findings.append(Finding(display, 0, "vendor-or-binary-file",
                                    "headers, libraries and binary artifacts must not be bundled"))
            continue
        if suffixes & RAW_SUFFIXES or path.name in {"audit.json", "config.json", "preflight.json"}:
            findings.append(Finding(display, 0, "raw-result-or-archive",
                                    "raw measurements and archives must not be published with source"))
            continue
        if path.name not in ROOT_FILES and path.suffix not in TEXT_SUFFIXES:
            findings.append(Finding(display, 0, "unsupported-file-type",
                                    "file type is outside the source publication allowlist"))
            continue
        if path.stat().st_size > MAX_TEXT_BYTES:
            findings.append(Finding(display, 0, "oversize-file", "review unusually large source files manually"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(Finding(display, 0, "unreadable-text", "candidate is not readable UTF-8 text"))
            continue
        if "\x00" in text:
            findings.append(Finding(display, 0, "binary-content", "NUL bytes found in a text candidate"))
            continue
        findings.extend(scan_text(text, display))
        if path.suffix == ".md":
            findings.extend(check_markdown_links(text, path, root))
    return sorted(set(findings))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        findings = check_tree(args.root)
    except (OSError, ValueError) as exc:
        print(f"publication check failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    for item in findings:
        print(f"{item.path}:{item.line}: {item.code}: {item.message}")
    if findings:
        print(f"Publication check failed: {len(findings)} finding(s). Matched values were not printed.")
        return 1
    print("Publication source check passed (offline heuristics; manual review still required).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
