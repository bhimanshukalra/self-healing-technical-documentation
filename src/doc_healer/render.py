"""Markdown for the fix pull request and for the comment listing sections that need a person.

Everything an LLM wrote (explanations, quotes, reasons) is untrusted: it is flattened to one line,
and characters that would ping people, make links or inject HTML are neutralized before it lands
in a GitHub comment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote

from .apply import ApplyResult, SkippedFix
from .driver import RunResult
from .modes import Decision
from .models import DocSection

FIX_MARKER = "<!-- doc-healer:fix -->"
REVIEW_MARKER = "<!-- doc-healer:review -->"

Kind = Literal["draft", "flag", "unverified", "not_applied"]
_KIND_LABEL = {
    "draft": "draft for review",
    "flag": "needs a manual fix",
    "unverified": "could not be checked",
    "not_applied": "fix could not be applied",
}


@dataclass(frozen=True)
class LinkContext:
    """Where links should point. Without a repository, sections are named by file and line only."""

    repo: str | None = None  # "owner/name"
    ref: str = "HEAD"  # commit sha or branch the line numbers refer to
    server: str = "https://github.com"


@dataclass(frozen=True)
class ReviewItem:
    section: DocSection
    kind: Kind
    explanation: str
    reasons: tuple[str, ...]
    confidence: float | None = None
    diff: str | None = None  # a suggested edit, when there is one worth showing


def section_link(section: DocSection, ctx: LinkContext) -> str:
    label = f"`{section.file_path}` – {_plain(section.heading, 120)}"
    if ctx.repo is None:
        return f"{label} (line {section.start_line})"
    url = f"{ctx.server}/{ctx.repo}/blob/{quote(ctx.ref, safe='')}/{quote(section.file_path)}#L{section.start_line}"
    return f"[{label}]({url})"


def review_items(result: RunResult, skipped: Iterable[SkippedFix] = ()) -> list[ReviewItem]:
    """Every section a person has to look at, in priority order."""
    decisions = {d.section_id: d for d in result.decisions}
    skipped_by_id = {s.section_id: s.reason for s in skipped}
    items: list[ReviewItem] = []
    for v in result.verification.results:
        section, d = v.suspect.section, decisions.get(v.section_id)
        if v.status == "unverified":
            items.append(ReviewItem(section, "unverified", "", (_plain(v.explanation),)))
        elif v.status == "stale" and d is not None:
            if v.section_id in skipped_by_id:
                items.append(ReviewItem(section, "not_applied", v.explanation, (skipped_by_id[v.section_id],), d.confidence))
            elif d.mode == "draft":
                items.append(ReviewItem(section, "draft", v.explanation, d.reasons, d.confidence,
                                        d.correction.diff() if d.correction else None))
            elif d.mode == "flag":
                items.append(ReviewItem(section, "flag", v.explanation, d.reasons, None,
                                        d.correction.diff() if d.correction and d.correction.status == "corrected" else None))
    return items


def render_fix_pr(
    decisions: Iterable[Decision],
    applied: ApplyResult,
    ctx: LinkContext,
    *,
    pr_number: int | None = None,
) -> tuple[str, str]:
    """Title and body for the pull request that carries the automatic fixes."""
    done = {a.section_id for a in applied.applied}
    fixes = [d for d in decisions if d.mode == "auto_fix" and d.section_id in done]
    source = f"#{pr_number}" if pr_number else f"`{ctx.ref[:7]}`"
    title = f"docs: update {len(fixes)} section{'s' if len(fixes) != 1 else ''} for changes in {source.strip('`')}"

    parts = [
        FIX_MARKER,
        f"doc-healer found documentation that changes in {source} made stale, and fixed "
        f"{len(fixes)} section{'s' if len(fixes) != 1 else ''} automatically. Each fix is a small, targeted edit "
        "that was checked against the new code before this PR was opened. Please review the diffs.",
    ]
    for n, d in enumerate(fixes, 1):
        section = d.verification.suspect.section
        triggers = ", ".join(
            f"`{t.verdict.change.chunk_id}` ({t.verdict.reason.replace('_', ' ')})" for t in d.verification.suspect.triggers
        )
        parts.append(
            f"#### {n}. {section_link(section, ctx)}\n"
            f"**What was wrong:** {_plain(d.verification.explanation)}\n"
            f"**Triggered by:** {triggers}\n"
            f"**Confidence:** {d.confidence:.2f} · validation passed\n\n"
            f"<details><summary>Diff</summary>\n\n{_fenced(d.correction.diff(), 'diff')}\n\n</details>"
        )
    return title, "\n\n".join(parts) + "\n"


def render_review_comment(items: list[ReviewItem], ctx: LinkContext, fix_pr_url: str | None = None) -> str:
    """The comment for the original PR: what needs a person, with links to each section."""
    parts = [REVIEW_MARKER, "## doc-healer"]
    if fix_pr_url:
        parts.append(f"Some stale documentation was fixed automatically in {fix_pr_url}.")
    if not items:
        return "\n\n".join(parts) + "\n"
    parts.append(
        f"### Needs human review ({len(items)})\n\n"
        "Code changes in this PR may have made these documentation sections stale, and doc-healer "
        "could not safely fix them on its own."
    )
    for n, item in enumerate(items, 1):
        lines = [f"#### {n}. {section_link(item.section, ctx)}"]
        status = _KIND_LABEL[item.kind] + (f" · confidence {item.confidence:.2f}" if item.confidence else "")
        lines.append(f"**Status:** {status}")
        if item.explanation:
            lines.append(f"**What changed:** {_plain(item.explanation)}")
        if item.reasons:
            lines.append("**Why it needs a person:** " + "; ".join(_plain(r) for r in item.reasons))
        if item.diff:
            lines.append(f"\n<details><summary>Suggested edit</summary>\n\n{_fenced(item.diff, 'diff')}\n\n</details>")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def _plain(text: str, limit: int = 400) -> str:
    """One safe line from untrusted text: no pings, links, HTML or Markdown structure."""
    flat = " ".join(str(text).split())[:limit]
    flat = flat.replace("@", "@​").replace("[", "\\[").replace("]", "\\]")
    return flat.replace("<", "&lt;").replace(">", "&gt;")


def _fenced(text: str, lang: str = "") -> str:
    """``text`` in a code fence long enough that nothing inside can close it early."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text.rstrip(chr(10))}\n{fence}"
