"""FR-2.3: from meaningful code changes to the doc sections that may now be stale."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from .index import LinkIndex
from .meaningful import Verdict
from .models import DocSection, Link

log = logging.getLogger(__name__)

# How strongly each kind of change suggests the docs are now wrong. A removed or re-signed
# chunk almost certainly breaks docs that name it; a body edit often does not.
REASON_WEIGHT = {
    "removed": 1.0,
    "api_changed": 1.0,
    "config_changed": 0.9,
    "added": 0.6,
    "body_changed": 0.6,
}
DEFAULT_WEIGHT = 0.6


@dataclass(frozen=True)
class Trigger:
    """One code change that points at a doc section, through one link."""

    verdict: Verdict
    link: Link


@dataclass(frozen=True)
class Suspect:
    section: DocSection
    triggers: tuple[Trigger, ...]
    priority: float  # link score x change weight, best trigger; orders suspects for LLM budgeting


@dataclass(frozen=True)
class SuspectReport:
    suspects: list[Suspect]  # most likely stale first
    already_updated: list[Suspect] = field(default_factory=list)  # section also edited in this PR
    unresolved: list[Link] = field(default_factory=list)  # links to sections that no longer exist


def find_suspects(
    verdicts: Iterable[Verdict],
    index: LinkIndex,
    sections: Mapping[str, DocSection],
    already_updated: Collection[str] = (),
    min_score: float = 0.0,
) -> SuspectReport:
    """Look up the doc sections linked to each meaningful change.

    ``sections`` are the docs as they are at the head of the PR. A section the PR also
    edited is reported separately instead of as a suspect: its author has probably
    handled it. A section is listed once however many changes point at it.
    """
    grouped: dict[str, list[Trigger]] = defaultdict(list)
    for verdict in verdicts:
        if not verdict.meaningful:
            continue
        for link in index.docs_for_chunk(verdict.change.chunk_id):
            if link.score >= min_score:
                grouped[link.doc_section_id].append(Trigger(verdict, link))

    suspects: list[Suspect] = []
    updated: list[Suspect] = []
    unresolved: list[Link] = []
    for section_id, triggers in grouped.items():
        section = sections.get(section_id)
        if section is None:
            log.warning("Index links to missing doc section %s; the index may be stale", section_id)
            unresolved.extend(t.link for t in triggers)
            continue
        priority = max(t.link.score * REASON_WEIGHT.get(t.verdict.reason, DEFAULT_WEIGHT) for t in triggers)
        suspect = Suspect(section, tuple(triggers), round(priority, 4))
        (updated if section_id in already_updated else suspects).append(suspect)

    order = lambda s: (-s.priority, s.section.id)  # noqa: E731
    return SuspectReport(sorted(suspects, key=order), sorted(updated, key=order), unresolved)
