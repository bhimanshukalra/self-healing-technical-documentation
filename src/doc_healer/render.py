"""Markdown for the fix pull request and for the comment listing sections that need a person.

Everything an LLM wrote (explanations, quotes, reasons) is untrusted: it is flattened to one line,
and characters that would ping people, make links or inject HTML are neutralized before it lands
in a GitHub comment.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote

from .apply import ApplyResult, SkippedFix
from .driver import RunResult
from .modes import Decision
from .models import DocSection

FIX_MARKER = "<!-- doc-healer:fix -->"
COMMENT_MARKER = "<!-- doc-healer:summary -->"

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
    heading = " > ".join(_plain(part, 60) for part in section.heading_path)  # escape each title, not our separator
    label = f"`{section.file_path}` – {heading}"
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
                diff = d.correction.diff() if d.correction and d.correction.status == "corrected" else None
                items.append(ReviewItem(section, "not_applied", v.explanation, (skipped_by_id[v.section_id],), d.confidence, diff))
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
            f"<details><summary>Diff</summary>\n\n{_fenced(_clip_diff(d.correction.diff()), 'diff')}\n\n</details>"
        )
    return title, "\n\n".join(parts) + "\n"


MAX_COMMENT_CHARS = 60000  # GitHub rejects comments over 65536
MAX_DIFF_CHARS = 3000
MAX_LISTED = 50


@dataclass(frozen=True)
class Entry:
    section: DocSection
    note: str = ""
    provider: str = ""  # which LLM provider checked it


@dataclass(frozen=True)
class CommentData:
    accurate: list[Entry]
    fixed: list[Entry]
    review: list[ReviewItem]
    already_updated: list[Entry]
    unresolved_links: int = 0
    llm_calls: dict = field(default_factory=dict)
    llm_configured: bool = True
    fix_ref: str | None = None  # markdown for the fix PR, e.g. "[#42](url)"
    providers: dict = field(default_factory=dict)  # section id -> provider that verified it
    notices: list[str] = field(default_factory=list)  # things that went wrong, e.g. a PR that could not be opened

    @property
    def multiple_providers(self) -> bool:
        return len(self.llm_calls.get("by_provider", {})) > 1


def comment_data(
    result: RunResult, applied: ApplyResult, fix_ref: str | None = None, notices: Iterable[str] = ()
) -> CommentData:
    """Everything the summary comment reports, taken from a run and what was actually applied."""
    done = {a.section_id for a in applied.applied}
    decisions = {d.section_id: d for d in result.decisions}
    accurate = [Entry(v.suspect.section, v.explanation, v.provider) for v in result.verification.accurate]
    fixed = [
        Entry(v.suspect.section, v.explanation, v.provider)
        for v in result.verification.stale if v.section_id in done and decisions.get(v.section_id)
    ]
    updated = [Entry(s.section) for s in result.analysis.report.already_updated]
    return CommentData(
        accurate, fixed, review_items(result, applied.skipped), updated,
        len(result.analysis.report.unresolved), result.llm_calls, bool(result.llm_calls), fix_ref,
        {v.section_id: v.provider for v in result.verification.results if v.provider},
        list(notices),
    )


def summary_line(data: CommentData) -> str:
    """The one-line headline, e.g. "3 sections verified accurate, 1 auto-fixed (see #42), 2 flagged for review"."""
    def n(count: int) -> str:
        return f"{count} section{'s' if count != 1 else ''}"

    parts = []
    if data.accurate:
        parts.append(f"{n(len(data.accurate))} verified accurate")
    if data.fixed:
        ref = f" (see {data.fix_ref})" if data.fix_ref else ""
        parts.append(f"{len(data.fixed)} auto-fixed{ref}")
    if data.review:
        parts.append(f"{len(data.review)} flagged for review")
    if data.already_updated:
        parts.append(f"{len(data.already_updated)} already updated in this PR")
    if not parts:
        return "**Doc Check Results:** no documentation sections were affected by this PR's code changes."
    return "**Doc Check Results:** " + ", ".join(parts) + "."


def render_summary_comment(data: CommentData, ctx: LinkContext) -> str:
    """The comment posted on every PR the Action runs on, within GitHub's size limit.

    If the full text is too long it is cut back in steps (fewer sections, then no diffs) rather than
    posting nothing.
    """
    text = ""
    for max_items, diffs in ((MAX_LISTED, True), (20, True), (10, False), (5, False), (2, False), (0, False)):
        text = _render_summary(data, ctx, max_items, diffs)
        if len(text) <= MAX_COMMENT_CHARS:
            return text
    return text[:MAX_COMMENT_CHARS]


def _render_summary(data: CommentData, ctx: LinkContext, max_items: int, with_diffs: bool) -> str:
    def tag(entry_provider: str) -> str:
        return f" · via `{_plain(entry_provider, 40)}`" if data.multiple_providers and entry_provider else ""

    parts = [COMMENT_MARKER, "## doc-healer", summary_line(data)]
    for notice in data.notices:
        parts.append(f"> ⚠️ {_plain(notice, 300)}")
    if not data.llm_configured and (data.review or data.accurate):
        parts.append("No LLM was configured, so these sections were found but not verified.")

    if data.review:
        shown = data.review[:max_items]
        block = [f"### Needs human review ({len(data.review)})", (
            "Code changes in this PR may have made these documentation sections stale, and doc-healer "
            "could not safely fix them on its own."
        )]
        for n, item in enumerate(shown, 1):
            lines = [f"#### {n}. {section_link(item.section, ctx)}"]
            status = _KIND_LABEL[item.kind] + (f" · confidence {item.confidence:.2f}" if item.confidence else "")
            lines.append(f"**Status:** {status}{tag(data.providers.get(item.section.id, ''))}")
            if item.explanation:
                lines.append(f"**What changed:** {_plain(item.explanation)}")
            if item.reasons:
                lines.append("**Why it needs a person:** " + "; ".join(_plain(r) for r in item.reasons))
            if item.diff and with_diffs:
                diff = _clip_diff(item.diff)
                lines.append(f"\n<details><summary>Suggested edit</summary>\n\n{_fenced(diff, 'diff')}\n\n</details>")
            block.append("\n".join(lines))
        if len(data.review) > len(shown):
            block.append(f"…and {len(data.review) - len(shown)} more (the list was shortened to fit GitHub's comment limit).")
        parts.append("\n\n".join(block))

    for title, entries, with_notes in (
        ("Auto-fixed", data.fixed, True),
        ("Verified accurate", data.accurate, True),
        ("Already updated in this PR", data.already_updated, False),
    ):
        if entries:
            shown = entries[:max_items]
            lines = []
            for e in shown:
                note = f" – {_plain(e.note, 160)}" if with_notes and e.note else ""
                lines.append(f"- {section_link(e.section, ctx)}{note}{tag(e.provider)}")
            if len(entries) > len(shown):
                lines.append(f"- …and {len(entries) - len(shown)} more")
            parts.append(f"<details><summary>{title} ({len(entries)})</summary>\n\n" + "\n".join(lines) + "\n\n</details>")

    if data.unresolved_links:
        parts.append(
            f"_The link index refers to {data.unresolved_links} documentation link(s) that no longer exist; "
            "it may be out of date._"
        )
    footer = _usage_footer(data)
    if footer:
        parts.append(footer)
    return "\n\n".join(parts) + "\n"


def _usage_footer(data: CommentData) -> str:
    by_provider = data.llm_calls.get("by_provider", {})
    if not by_provider:
        return ""
    used = ", ".join(f"`{_plain(name, 40)}` ({count})" for name, count in sorted(by_provider.items()))
    note = f"_Checked with {used}._"
    if data.llm_calls.get("fallback_calls"):
        note += " _The fallback provider was used after the primary ran out of capacity._"
    return note


def _clip_diff(diff: str) -> str:
    return diff if len(diff) <= MAX_DIFF_CHARS else diff[:MAX_DIFF_CHARS].rstrip() + "\n[... diff shortened ...]"


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
