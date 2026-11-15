"""The driver: one call that runs the whole pipeline on a git range and reports what it found.

detect changes -> keep meaningful ones -> find suspect doc sections -> verify with an LLM ->
repair -> validate -> choose a mode per section. Nothing here writes to the repository or talks
to GitHub; the Action turns the result into a pull request and a comment (FR-4.2, FR-4.3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .analysis import Analysis, analyze_changes
from .classifier import build_classifier
from .embeddings import Embedder
from .gitutil import checkout, read_at
from .index import IndexVersionError, LinkIndex
from .llm import LLMClient
from .modes import DEFAULT_CONFIDENCE_THRESHOLD, Decision, decide_modes
from .pipeline import build_index
from .repair import Correction, repair_all
from .validation import Validation, validate_all
from .verification import Verification, VerificationReport, verify_suspects

log = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = ".doc-healer/index.json"
IndexSource = Literal["committed", "built"]


@dataclass(frozen=True)
class RunConfig:
    base: str  # the PR's target branch or commit
    head: str = "HEAD"
    index_path: str = DEFAULT_INDEX_PATH  # repo-relative; read from the tree at ``base``
    classifier_backend: str = "llm"  # llm | none (laya and jev are Phase 5 experiments)
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    use_embeddings: bool = True  # only matters when the index has to be built
    store_path: Path | None = None  # vector cache for that build


@dataclass(frozen=True)
class Summary:
    suspects: int  # doc sections that code changes may have made stale
    accurate: int  # checked, still fine
    stale: int  # confirmed stale
    auto_fix: int  # ... of which fixed automatically
    draft: int  # ... of which have a draft for review
    flag: int  # ... of which need a person to write the fix
    unverified: int  # could not be checked (no LLM, rate limits, budget): needs a person
    already_updated: int  # the PR itself edited these sections

    @property
    def needs_review(self) -> int:
        return self.draft + self.flag + self.unverified


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    index_source: IndexSource
    analysis: Analysis
    verification: VerificationReport
    corrections: list[Correction]
    validations: list[Validation]
    decisions: list[Decision]
    summary: Summary
    llm_calls: dict[str, Any] = field(default_factory=dict)  # provider usage, for the PR comment

    def decision_for(self, section_id: str) -> Decision | None:
        return next((d for d in self.decisions if d.section_id == section_id), None)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready view (Action outputs, logs). The full corrected text stays on the Decision objects."""
        decisions = {d.section_id: d for d in self.decisions}
        sections = []
        for v in self.verification.results:
            d = decisions.get(v.section_id)
            sections.append({
                "section": v.section_id,
                "file": v.suspect.section.file_path,
                "heading": v.suspect.section.heading,
                "priority": v.suspect.priority,
                "status": v.status,
                "mode": d.mode if d else None,
                "confidence": d.confidence if d else v.confidence,
                "explanation": v.explanation,
                "stale_statements": list(v.stale_statements),
                "reasons": list(d.reasons) if d else [],
                "triggers": [
                    {"chunk": t.verdict.change.chunk_id, "change": t.verdict.reason, "link": t.link.method}
                    for t in v.suspect.triggers
                ],
                "diff": d.correction.diff() if d and d.mode != "flag" and d.correction else None,
            })
        report = self.analysis.report
        return {
            "base": self.config.base,
            "head": self.config.head,
            "index_source": self.index_source,
            "summary": {**vars(self.summary), "needs_review": self.summary.needs_review},
            "sections": sections,
            "stale_sections": [s["section"] for s in sections if s["status"] == "stale"],
            "already_updated": [s.section.id for s in report.already_updated],
            "unresolved_links": [l.doc_section_id for l in report.unresolved],
            "llm_calls": self.llm_calls,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"


def resolve_index(
    repo: Path,
    base: str,
    index_path: str = DEFAULT_INDEX_PATH,
    *,
    use_embeddings: bool = True,
    store_path: Path | None = None,
    embedder: Embedder | None = None,
) -> tuple[LinkIndex, IndexSource]:
    """The link index as it was at ``base``: the committed one if usable, else one built from a
    temporary checkout of ``base`` (so the PR's own changes never leak into the links)."""
    if (text := read_at(repo, base, index_path)) is not None:
        try:
            return LinkIndex.from_json(text, f"{base}:{index_path}"), "committed"
        except (IndexVersionError, ValueError, KeyError, TypeError) as exc:
            log.warning("Committed index %s is unusable (%s); rebuilding", index_path, exc)
    else:
        log.info("No committed index at %s:%s; building one", base, index_path)
    with checkout(repo, base) as tree:
        result = build_index(tree, store_path=store_path, embedder=embedder, use_embeddings=use_embeddings)
    return result.index, "built"


def run_check(
    repo: Path,
    config: RunConfig,
    client: LLMClient | None = None,
    embedder: Embedder | None = None,
) -> RunResult:
    """Run every stage on ``config.base..config.head``.

    Without an ``LLMClient`` the run stops after finding suspects: each is reported as
    unverified for a person to review. LLM failures (rate limits, budget, bad replies) never raise
    out of here; the affected sections are reported as unverified, flagged or drafts instead.
    """
    # An unknown or unimplemented backend should fail even in a dry run; only "llm" needs a client.
    needs_client = config.classifier_backend == "llm"
    classifier = None if (needs_client and client is None) else build_classifier(config.classifier_backend, client)
    index, source = resolve_index(
        repo, config.base, config.index_path,
        use_embeddings=config.use_embeddings, store_path=config.store_path, embedder=embedder,
    )
    analysis = analyze_changes(repo, config.base, config.head, index, classifier)
    suspects = analysis.report.suspects

    if client is None:
        verification = VerificationReport([
            Verification(s, "unverified", "not verified: no LLM is configured") for s in suspects
        ])
        corrections: list[Correction] = []
        validations: list[Validation] = []
        decisions: list[Decision] = []
    else:
        verification = verify_suspects(suspects, client)
        corrections = repair_all(verification, client)
        validations = validate_all(corrections, client)
        decisions = decide_modes(verification, corrections, validations, classifier, config.confidence_threshold)

    summary = Summary(
        suspects=len(suspects),
        accurate=len(verification.accurate),
        stale=len(verification.stale),
        auto_fix=sum(d.mode == "auto_fix" for d in decisions),
        draft=sum(d.mode == "draft" for d in decisions),
        flag=sum(d.mode == "flag" for d in decisions),
        unverified=len(verification.unverified),
        already_updated=len(analysis.report.already_updated),
    )
    stats = {}
    if client is not None:
        stats = {
            "calls": client.stats.calls,
            "retries": client.stats.retries,
            "fallback_calls": client.stats.fallback_calls,
            "by_provider": dict(client.stats.by_provider),
        }
    return RunResult(config, source, analysis, verification, corrections, validations, decisions, summary, stats)
