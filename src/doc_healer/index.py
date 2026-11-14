"""The code-to-docs link graph, persisted as a JSON file."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from .models import Link

SCHEMA_VERSION = 1


class IndexVersionError(ValueError):
    """The index file was written with a different schema version."""


class LinkIndex:
    def __init__(self, links: Iterable[Link] = ()):
        self.links: list[Link] = sorted(links, key=lambda l: (l.doc_section_id, l.code_chunk_id, l.method))
        self._by_chunk: dict[str, list[Link]] = defaultdict(list)
        self._by_doc: dict[str, list[Link]] = defaultdict(list)
        for link in self.links:
            self._by_chunk[link.code_chunk_id].append(link)
            self._by_doc[link.doc_section_id].append(link)

    def docs_for_chunk(self, chunk_id: str) -> list[Link]:
        """Links to the doc sections that may describe this code chunk (the "suspects")."""
        return list(self._by_chunk.get(chunk_id, ()))

    def linked_chunk_ids(self) -> set[str]:
        """Every code chunk that at least one doc section is linked to."""
        return set(self._by_chunk)

    def chunks_for_doc(self, doc_section_id: str) -> list[Link]:
        return list(self._by_doc.get(doc_section_id, ()))

    def to_json(self) -> str:
        data = {"version": SCHEMA_VERSION, "links": [asdict(l) for l in self.links]}
        return json.dumps(data, indent=2) + "\n"  # stable output keeps diffs small

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinkIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != SCHEMA_VERSION:
            raise IndexVersionError(
                f"{path}: schema version {data.get('version')!r}, expected {SCHEMA_VERSION}; rebuild the index"
            )
        return cls(Link(**item) for item in data["links"])
