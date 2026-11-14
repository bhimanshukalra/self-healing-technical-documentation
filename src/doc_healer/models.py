"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChunkKind = Literal["function", "class", "endpoint", "config", "cli"]


@dataclass(frozen=True)
class CodeChunk:
    """A semantic unit of code: function, class, endpoint, config schema or CLI command."""

    id: str  # "<relative file path>::<qualified name>"
    kind: ChunkKind
    file_path: str
    name: str  # qualified name, e.g. "TaskStore.add_task"
    signature: str
    docstring: str | None
    source: str
    start_line: int
    end_line: int
    extra: dict[str, str] = field(default_factory=dict)  # e.g. route / http_method
    own_source: str | None = None  # classes only: source without method bodies

    @property
    def diff_source(self) -> str:
        """The text that decides whether this chunk itself changed."""
        return self.source if self.own_source is None else self.own_source


LinkMethod = Literal["heuristic", "embedding"]


@dataclass(frozen=True)
class Link:
    """An edge between a code chunk and a doc section."""

    code_chunk_id: str
    doc_section_id: str
    method: LinkMethod
    score: float  # 0..1; heuristic confidence or cosine similarity


@dataclass(frozen=True)
class DocSection:
    """A markdown section: the text under one heading, up to the next heading."""

    id: str  # "<relative doc path>#<heading path>"
    file_path: str
    heading_path: tuple[str, ...]  # e.g. ("Configuration", "Environment Variables")
    content: str  # raw markdown body, excluding the heading line and child sections
    code_refs: tuple[str, ...]  # names the section mentions, in order of first appearance
    start_line: int  # 1-based line of the heading
    end_line: int

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)
