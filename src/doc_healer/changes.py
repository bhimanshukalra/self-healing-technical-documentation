"""Phase 2: find which code chunks a diff added, removed or modified."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .chunker import chunk_source, is_code_path
from .gitutil import changed_files, read_at
from .models import CodeChunk

ChangeType = Literal["added", "removed", "modified"]


@dataclass(frozen=True)
class ChunkChange:
    chunk_id: str
    change_type: ChangeType
    old: CodeChunk | None  # None when added
    new: CodeChunk | None  # None when removed


def detect_chunk_changes(repo: Path, base: str, head: str) -> list[ChunkChange]:
    """Compare the code chunks at ``base`` and ``head``.

    Works on whole chunks rather than diff lines, so removed code and unchanged
    siblings are handled exactly: a chunk is *modified* only if its own source differs
    (a class is not modified by an edit to one of its methods). Test files, vendored
    directories and non-Python files are ignored. A renamed file is compared under its
    new path, so a pure rename yields no changes.

    This layer reports every textual change, including comment- and whitespace-only
    edits; the meaningful-change filter (FR-2.2) decides which of them matter.
    """
    changes: list[ChunkChange] = []
    for file in changed_files(repo, base, head):
        if not is_code_path(file.path):
            continue
        old_source = None if file.status == "added" else read_at(repo, base, file.old_path or file.path)
        new_source = None if file.status == "deleted" else read_at(repo, head, file.path)
        old = {c.id: c for c in chunk_source(old_source, file.path)} if old_source is not None else {}
        new = {c.id: c for c in chunk_source(new_source, file.path)} if new_source is not None else {}

        for cid in sorted(old.keys() | new.keys()):
            if cid not in old:
                changes.append(ChunkChange(cid, "added", None, new[cid]))
            elif cid not in new:
                changes.append(ChunkChange(cid, "removed", old[cid], None))
            elif old[cid].diff_source != new[cid].diff_source:
                changes.append(ChunkChange(cid, "modified", old[cid], new[cid]))
    return changes
