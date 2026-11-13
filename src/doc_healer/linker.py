"""Heuristic linking of doc sections to code chunks by the names they mention."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .models import CodeChunk, DocSection, Link

if TYPE_CHECKING:
    from .embeddings import Embedder, VectorStore

ROUTE_RE = re.compile(r"^([A-Z]+) (/\S*)$")
ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

SCORE_QUALIFIED = 1.0  # ref equals a chunk's qualified name (or an endpoint route)
SCORE_UNIQUE_SHORT = 0.9  # ref equals the final name part of exactly one chunk
SCORE_ENV_VAR = 0.9  # ALL_CAPS ref found verbatim in a chunk's source
SCORE_AMBIGUOUS_SHORT = 0.6  # ref equals the final name part of several chunks
SCORE_CONFIG_FIELD = 0.7  # ref equals a field name of a config class


DEFAULT_SIMILARITY_THRESHOLD = 0.70  # on the fixture: keeps all labeled links, ~half the extras of 0.65
DEFAULT_TOP_K = 5


def link_embeddings(
    sections: Iterable[DocSection],
    store: "VectorStore",
    embedder: "Embedder",
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[Link]:
    """Link each section to the code chunks whose cosine similarity meets ``threshold``.

    ``store`` must already hold the code chunks (see ``VectorStore.sync``).
    """
    from .embeddings import section_text

    sections = list(sections)
    if not sections:
        return []
    vectors = embedder.embed([section_text(s) for s in sections])
    links = [
        Link(chunk_id, section.id, "embedding", round(score, 4))
        for section, vector in zip(sections, vectors)
        for chunk_id, score in store.query(vector, top_k)
        if score >= threshold
    ]
    return sorted(links, key=lambda l: (l.doc_section_id, l.code_chunk_id))


def merge_links(*groups: Iterable[Link]) -> list[Link]:
    """Combine link lists, one link per (chunk, section) pair.

    Earlier groups win, so pass heuristic links first: a name match is more
    trustworthy than a similarity score.
    """
    merged: dict[tuple[str, str], Link] = {}
    for group in groups:
        for link in group:
            merged.setdefault((link.code_chunk_id, link.doc_section_id), link)
    return sorted(merged.values(), key=lambda l: (l.doc_section_id, l.code_chunk_id))


def link_heuristic(chunks: Iterable[CodeChunk], sections: Iterable[DocSection]) -> list[Link]:
    """Link each section to the chunks its ``code_refs`` point at.

    A pair linked by several refs keeps its highest score.
    """
    chunks = list(chunks)
    by_name = {c.name: c for c in chunks}
    by_short: dict[str, list[CodeChunk]] = defaultdict(list)
    for c in chunks:
        by_short[c.name.rsplit(".", 1)[-1]].append(c)
    routes = {
        (c.extra["http_method"], c.extra["route"]): c
        for c in chunks
        if c.kind == "endpoint"
    }
    config_chunks = [c for c in chunks if c.kind == "config"]

    links: dict[tuple[str, str], Link] = {}

    def add(chunk: CodeChunk, section: DocSection, score: float) -> None:
        key = (chunk.id, section.id)
        if key not in links or links[key].score < score:
            links[key] = Link(chunk.id, section.id, "heuristic", score)

    for section in sections:
        for ref in section.code_refs:
            if route := ROUTE_RE.match(ref):
                if chunk := routes.get((route.group(1), route.group(2))):
                    add(chunk, section, SCORE_QUALIFIED)
                continue
            if ref in by_name:
                add(by_name[ref], section, SCORE_QUALIFIED)
                continue
            if matches := by_short.get(ref):
                score = SCORE_UNIQUE_SHORT if len(matches) == 1 else SCORE_AMBIGUOUS_SHORT
                for chunk in matches:
                    add(chunk, section, score)
                continue
            if ENV_VAR_RE.match(ref):
                word = re.compile(rf"\b{re.escape(ref)}\b")
                for chunk in chunks:
                    if word.search(chunk.source):
                        add(chunk, section, SCORE_ENV_VAR)
            else:
                for chunk in config_chunks:
                    if ref in chunk.extra.get("fields", "").split(","):
                        add(chunk, section, SCORE_CONFIG_FIELD)

    return sorted(links.values(), key=lambda l: (l.doc_section_id, l.code_chunk_id))
