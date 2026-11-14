"""Phase 2 end to end: a git range plus a link index in, suspect doc sections out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .changes import ChunkChange, detect_chunk_changes
from .docparser import is_doc_path, parse_markdown
from .gitutil import list_files, read_at
from .index import LinkIndex
from .meaningful import ChangeClassifier, Verdict, filter_changes
from .models import DocSection
from .suspects import SuspectReport, find_suspects


@dataclass(frozen=True)
class Analysis:
    changes: list[ChunkChange]  # every chunk that changed textually
    verdicts: list[Verdict]  # a verdict for each change
    report: SuspectReport


def docs_at(repo: Path, ref: str) -> dict[str, DocSection]:
    """All doc sections in the tree at ``ref``, by section id."""
    sections: dict[str, DocSection] = {}
    for path in list_files(repo, ref):
        if is_doc_path(path) and (text := read_at(repo, ref, path)) is not None:
            sections.update((s.id, s) for s in parse_markdown(text, path))
    return sections


def updated_section_ids(base_docs: dict[str, DocSection], head_docs: dict[str, DocSection]) -> set[str]:
    """Sections that are new or whose text changed between base and head."""
    return {
        sid for sid, section in head_docs.items()
        if sid not in base_docs or base_docs[sid].content != section.content
    }


def analyze_changes(
    repo: Path,
    base: str,
    head: str,
    index: LinkIndex,
    classifier: ChangeClassifier | None = None,
    min_score: float = 0.0,
) -> Analysis:
    """Detect code changes between ``base`` and ``head``, keep the meaningful ones, and
    find the doc sections they may have made stale (using the index built at base)."""
    changes = detect_chunk_changes(repo, base, head)
    verdicts = filter_changes(changes, index.linked_chunk_ids(), classifier)
    base_docs, head_docs = docs_at(repo, base), docs_at(repo, head)
    report = find_suspects(
        verdicts, index, head_docs, updated_section_ids(base_docs, head_docs), min_score
    )
    return Analysis(changes, verdicts, report)
