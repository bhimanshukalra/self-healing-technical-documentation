"""Heuristic linking of doc sections to code chunks by the names they mention."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from .models import CodeChunk, DocSection, Link

ROUTE_RE = re.compile(r"^([A-Z]+) (/\S*)$")
ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

SCORE_QUALIFIED = 1.0  # ref equals a chunk's qualified name (or an endpoint route)
SCORE_UNIQUE_SHORT = 0.9  # ref equals the final name part of exactly one chunk
SCORE_ENV_VAR = 0.9  # ALL_CAPS ref found verbatim in a chunk's source
SCORE_AMBIGUOUS_SHORT = 0.6  # ref equals the final name part of several chunks
SCORE_CONFIG_FIELD = 0.7  # ref equals a field name of a config class


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
