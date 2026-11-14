"""FR-2.2: decide which chunk changes could make documentation stale."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol

from .changes import ChunkChange

CLASSIFIER_THRESHOLD = 0.5  # keep a change when P(meaningful) is at least this


class ChangeClassifier(Protocol):
    """FR-2.2a hook for the one judgment rules can't make: does this body edit change behavior?"""

    def meaningful_probability(self, change: ChunkChange) -> float:
        """Probability in [0, 1] that the edit changes behavior callers or docs would notice."""
        ...


@dataclass(frozen=True)
class Verdict:
    change: ChunkChange
    meaningful: bool
    reason: str
    confidence: float  # how sure we are of the verdict: 1.0 for rule-decided cases


def is_private(chunk_id: str) -> bool:
    """True if any part of the qualified name is ``_private`` (dunder names are public API)."""
    qualname = chunk_id.split("::", 1)[-1]
    return any(p.startswith("_") and not (p.startswith("__") and p.endswith("__")) for p in qualname.split("."))


def judge(
    change: ChunkChange,
    linked_ids: Collection[str] = (),
    classifier: ChangeClassifier | None = None,
) -> Verdict:
    """Classify one change. ``linked_ids`` are chunks the docs mention (the link graph)."""
    cid = change.chunk_id

    def verdict(meaningful: bool, reason: str, confidence: float = 1.0) -> Verdict:
        return Verdict(change, meaningful, reason, confidence)

    if is_private(cid) and cid not in linked_ids:
        return verdict(False, "private")

    if change.change_type == "added":
        return verdict(True, "added")
    if change.change_type == "removed":
        return verdict(True, "removed")

    old, new = change.old, change.new
    if old.fingerprint == new.fingerprint:
        # Same structure: only comments, whitespace or the docstring changed.
        return verdict(False, "docstring_only" if old.docstring != new.docstring else "cosmetic")
    if (old.signature, old.kind, old.extra, old.decorators) != (new.signature, new.kind, new.extra, new.decorators):
        return verdict(True, "api_changed")
    if new.kind == "config":
        return verdict(True, "config_changed")

    # Same interface, different body: behavior change or internal refactor?
    if classifier is None:
        # Err toward flagging; LLM staleness verification (FR-2.4) weeds out false positives.
        return verdict(True, "body_changed", 0.5)
    p = classifier.meaningful_probability(change)
    meaningful = p >= CLASSIFIER_THRESHOLD
    return verdict(meaningful, "body_changed" if meaningful else "internal_refactor", max(p, 1 - p))


def filter_changes(
    changes: Iterable[ChunkChange],
    linked_ids: Collection[str] = (),
    classifier: ChangeClassifier | None = None,
) -> list[Verdict]:
    """A verdict for every change; use ``meaningful_only`` to keep the ones that matter."""
    return [judge(c, linked_ids, classifier) for c in changes]


def meaningful_only(verdicts: Iterable[Verdict]) -> list[Verdict]:
    return [v for v in verdicts if v.meaningful]
