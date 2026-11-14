"""Thin wrappers around the git CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Status = Literal["added", "modified", "deleted", "renamed"]
_STATUS = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed", "T": "modified"}


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileChange:
    status: Status
    path: str  # path at head (the new path for renames)
    old_path: str | None = None  # path at base, for renames


def _git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def changed_files(repo: Path, base: str, head: str) -> list[FileChange]:
    """Files that differ between two refs, with renames detected."""
    out = _git(repo, "diff", "--name-status", "-z", "-M", base, head).split(b"\0")
    fields = [f.decode("utf-8") for f in out if f]
    changes: list[FileChange] = []
    i = 0
    while i < len(fields):
        code = fields[i][0]
        if code in ("R", "C"):
            old, new = fields[i + 1], fields[i + 2]
            changes.append(FileChange("renamed" if code == "R" else "added", new, old if code == "R" else None))
            i += 3
        else:
            changes.append(FileChange(_STATUS.get(code, "modified"), fields[i + 1]))
            i += 2
    return changes


def read_at(repo: Path, ref: str, path: str) -> str | None:
    """File contents at ``ref``, or None if it doesn't exist there or isn't UTF-8 text."""
    try:
        return _git(repo, "show", f"{ref}:{path}").decode("utf-8")
    except (GitError, UnicodeDecodeError):
        return None
