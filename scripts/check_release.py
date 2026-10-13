"""Fail closed on private artifacts, symlinks and common credential patterns.

Only paths, line numbers and finding categories are printed, never matched text.
Run this against the working tree AND against built source archives before release.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "release",
    "__pycache__",
    "node_modules",
    ".ruff_cache",
    ".pytest_cache",
}
SOURCE_DIRS = {"src", "tests", "docs", "scripts", "examples", "requirements", ".github"}
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
    "MANIFEST.in",
    "SOURCE_MANIFEST.json",
}
SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".json",
    ".html",
    ".css",
    ".js",
    ".swift",
    ".yml",
    ".yaml",
    ".command",
    ".png",
    ".svg",
}
FORBIDDEN_NAMES = {
    "contacts.json",
    "active-source.json",
    "bridge-session.json",
    "runtime-candidates.txt",
    "environment-report.json",
    "models-manifest.json",
    ".env",
}
PATTERNS = {
    "personal_absolute_path": re.compile(r"/(?:Users)/[^/\s\"'<>]+/|[A-Za-z]:\\Users\\[^\\\s\"'<>]+\\"),
    "wechat_account_id": re.compile(r"\bwxid_[A-Za-z0-9]{8,}\b"),
    "github_credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "hf_credential": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal_session_secret": re.compile(r"(?:token|secret|api_key)\s*[:=]\s*[\"'][A-Za-z0-9_-]{32,}[\"']", re.I),
}


def candidate_files(root=ROOT):
    files = set()
    for directory, folders, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        # Report directory symlinks too; never walk into their targets.
        for name in list(folders):
            if (parent / name).is_symlink():
                files.add(parent / name)
                folders.remove(name)
        folders[:] = [name for name in folders if name not in SKIP_DIRS and not name.endswith(".egg-info")]
        for name in names:
            if name != ".DS_Store" and not name.endswith((".pyc", ".pyo")):
                files.add(parent / name)
    # git add -f can defeat ignores. Always include tracked ignored artifacts.
    result = subprocess.run(["git", "ls-files", "-z", "--cached"], cwd=root, capture_output=True)
    if result.returncode == 0:
        files.update(root / name.decode() for name in result.stdout.split(b"\0") if name)
    return sorted(files)


def scan_paths(root, files, deny_words=()):
    findings = []
    for path in files:
        relative = path.relative_to(root)
        label = relative.as_posix()

        def finding(category, line=None):
            findings.append({"path": label, "category": category, **({"line": line} if line else {})})

        if path.is_symlink():
            finding("symlink_not_publishable")
            continue
        if not path.is_file():
            finding("missing_tracked_file")
            continue
        if relative.parts[0] not in SOURCE_DIRS | {"macos"} and label not in ROOT_FILES:
            finding("outside_source_allowlist")
        if (
            any(
                part in {"private", "models", "vendor", "data", "node_modules"} or part.endswith(".app")
                for part in relative.parts
            )
            or path.name in FORBIDDEN_NAMES
        ):
            finding("private_or_generated_artifact")
        if path.suffix not in SUFFIXES and label not in ROOT_FILES:
            finding("unapproved_file_type")
        if path.stat().st_size > 2 * 1024 * 1024:
            finding("oversized_source_artifact")
            continue
        if path.suffix == ".png":
            if relative.parts[:2] != ("docs", "assets"):
                finding("image_outside_documented_demo")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            finding("unexpected_binary")
            continue
        for category, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                finding(category, content.count("\n", 0, match.start()) + 1)
        for word in deny_words:
            start = content.find(word)
            if word and start >= 0:
                finding("private_deny_word", content.count("\n", 0, start) + 1)
    return findings


def scan(root=ROOT, deny_words=()):
    files = candidate_files(root)
    return files, scan_paths(root, files, deny_words)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--deny-words-file", type=Path, help="Private local newline-separated deny list (never publish it)"
    )
    args = parser.parse_args(argv)
    words = args.deny_words_file.read_text().splitlines() if args.deny_words_file else ()
    files, findings = scan(args.root.resolve(), words)
    print(
        json.dumps(
            {"passed": not findings, "files_scanned": len(files), "findings": findings}, ensure_ascii=False, indent=2
        )
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
