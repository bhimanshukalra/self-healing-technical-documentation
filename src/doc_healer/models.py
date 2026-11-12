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
