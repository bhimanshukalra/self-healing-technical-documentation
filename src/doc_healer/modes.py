"""FR-3.3: choose what to do with each stale section: auto-fix, draft for review, or flag.

* auto_fix: a simple change, a clean validated correction, and high confidence. Safe to apply.
* draft:    a usable correction, but a human must look: the change is complex or its complexity is
            unknown, confidence is below the threshold, or the diagnosis/edits were not airtight.
            The text carries TODO markers.
* flag:     no usable correction (none generated, or it failed or could not be checked). The section is
            reported with the diagnosis for a human to fix.

Confidence is the lowest of the signals that could be wrong: the verifier's, the validator's, and
(for simple changes) the classifier's. Nothing here trusts an LLM's self-reported number on its own.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from .changes import ChunkChange
from .classifier import COMPLEX, SIMPLE, Classification, Classifier, complexity_rules
from .llm import LLMError
from .repair import Correction
from .validation import Validation
from .verification import Verification, VerificationReport

log = logging.getLogger(__name__)

Mode = Literal["auto_fix", "draft", "flag"]
Complexity = Literal["simple", "complex", "unknown"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.8

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Decision:
    verification: Verification
    mode: Mode
    confidence: float  # min of the signals; 0.0 when there was nothing to score
    complexity: Complexity
    reasons: tuple[str, ...]  # why this mode, in plain words
    new_content: str | None  # section text to write: the fix (auto_fix), the marked draft (draft), None (flag)
    correction: Correction | None = None
    validation: Validation | None = None

    @property
    def section_id(self) -> str:
        return self.verification.section_id


def decide_modes(
    verifications: VerificationReport | Iterable[Verification],
    corrections: Iterable[Correction],
    validations: Iterable[Validation],
    classifier: Classifier | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Decision]:
    """One decision per stale verification, in order. The three inputs are matched by section id."""
    stale = verifications.stale if isinstance(verifications, VerificationReport) else [
        v for v in verifications if v.status == "stale"
    ]
    correction_by_id = {c.section.id: c for c in corrections}
    validation_by_id = {v.section_id: v for v in validations}
    cache: dict[str, Classification | None] = {}  # one complexity judgment per changed chunk
    return [
        _decide(v, correction_by_id.get(v.section_id), validation_by_id.get(v.section_id), classifier, threshold, cache)
        for v in stale
    ]


def _decide(v, correction, validation, classifier, threshold, cache) -> Decision:
    def flag(*reasons: str) -> Decision:
        return Decision(v, "flag", 0.0, "unknown", reasons, None, correction, validation)

    if correction is None:
        return flag("no correction was generated")
    if correction.status != "corrected":
        return flag(f"no usable correction: {correction.note or correction.status}")
    if validation is None:
        return flag("the correction was not validated")
    if validation.status == "unchecked":
        return flag(f"the correction could not be validated: {validation.note}")
    if validation.status == "failed":
        problems = [*validation.lint_problems, *validation.issues] or ["a validation check failed"]
        return flag("validation failed", *problems)

    complexity, complexity_confidence, complexity_notes = assess_complexity(v, classifier, cache)
    signals = [v.confidence, validation.confidence]
    if complexity == SIMPLE:
        signals.append(complexity_confidence)
    confidence = round(min(signals), 4)

    concerns: list[str] = []
    if complexity == COMPLEX:
        concerns.append("complex change: a person should confirm the wording")
    elif complexity == "unknown":
        concerns.append("could not establish that the change is simple")
    if confidence < threshold:
        concerns.append(f"confidence {confidence:.2f} is below the threshold {threshold:.2f}")
    if not v.grounded:
        concerns.append("the diagnosis did not quote text found in the section")
    if correction.rejected:
        concerns.append(f"{len(correction.rejected)} proposed edit(s) were rejected, so the fix may be incomplete")
    if not correction.anchored:
        concerns.append("the edits were not tied to specific wrong statements")

    if not concerns:
        reasons = ("simple change", f"confidence {confidence:.2f} >= {threshold:.2f}", "validation passed")
        return Decision(v, "auto_fix", confidence, "simple", reasons, correction.corrected, correction, validation)
    reasons = tuple(concerns) + tuple(complexity_notes)
    draft = add_todo_markers(correction, concerns)
    return Decision(v, "draft", confidence, complexity, reasons, draft, correction, validation)


def assess_complexity(
    verification: Verification,
    classifier: Classifier | None,
    cache: dict[str, Classification | None] | None = None,
) -> tuple[Complexity, float, list[str]]:
    """Simple only if every change behind the section is simple; anything doubtful counts as complex
    or unknown. Added/removed chunks are complex by rule; other changes need the classifier."""
    cache = {} if cache is None else cache
    results: list[Classification] = []
    notes: list[str] = []
    unknown = False
    seen: set[str] = set()
    for trigger in verification.suspect.triggers:
        change = trigger.verdict.change
        if change.chunk_id in seen:
            continue
        seen.add(change.chunk_id)
        result = _classify(change, classifier, cache)
        if result is None:
            unknown = True
            notes.append(f"no complexity judgment for {change.chunk_id}")
        else:
            results.append(result)
    if any(r.label == COMPLEX for r in results):
        return "complex", 1.0, notes
    if unknown or not results:
        return "unknown", 0.0, notes
    return "simple", min(r.probability for r in results), notes


def _classify(change: ChunkChange, classifier: Classifier | None, cache) -> Classification | None:
    if change.chunk_id in cache:
        return cache[change.chunk_id]
    result = complexity_rules(change)
    if result is None and classifier is not None:
        try:
            result = classifier.classify_complexity(change)
        except LLMError as exc:  # provider down, budget spent, bad reply: assume the worst
            log.warning("Complexity classification failed for %s: %s", change.chunk_id, exc)
    cache[change.chunk_id] = result
    return result


# -- TODO markers ------------------------------------------------------------------------

def add_todo_markers(correction: Correction, reasons: Sequence[str]) -> str:
    """The corrected section with an HTML-comment TODO line above each edited region.

    Comments don't change how the page renders but are plain in the diff and the source. A marker
    never lands inside a code block or table; it goes above the block instead, so fences and
    tables stay intact. Text outside the markers is exactly ``correction.corrected``.
    """
    corrected = correction.corrected
    spans, shift = [], 0
    for edit in sorted(correction.edits, key=lambda e: e.start):  # offsets in the corrected text
        spans.append(edit.start + shift)
        shift += len(edit.replace) - (edit.end - edit.start)

    lines = corrected.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    targets = {_marker_line(lines, bisect.bisect_right(line_starts, s) - 1) for s in spans}

    text = "TODO(doc-healer): review this automatic correction. " + "; ".join(reasons)
    if correction.verification.explanation:
        text += f". Diagnosis: {correction.verification.explanation}"
    marker = f"<!-- {_comment_safe(text)} -->"
    for index in sorted(targets, reverse=True):
        lines.insert(index, marker)
    return "\n".join(lines)


def _marker_line(lines: list[str], index: int) -> int:
    """The line a marker for an edit on ``lines[index]`` may safely sit above."""
    fence, opener = None, 0
    for i in range(index + 1):
        m = _FENCE_RE.match(lines[i])
        if m and fence is None:
            fence, opener = m.group(1)[0], i
            if i == index:
                return i  # the edit is on an opening fence line
        elif m and fence == m.group(1)[0]:
            if i == index:
                return opener  # on a closing fence line: above the whole block
            fence = None
        elif i == index and fence is not None:
            return opener  # inside a code block: above the block
    if lines[index].lstrip().startswith("|"):  # a table row: above the table
        while index > 0 and lines[index - 1].lstrip().startswith("|"):
            index -= 1
    return index


def _comment_safe(text: str) -> str:
    return re.sub(r"-{2,}", "-", " ".join(text.split()))[:400]
