"""FR-4.2: turn a run's decisions into a fix branch and pull request, and a comment for the rest.

* auto-fixes are written to the doc files on a new branch, committed, pushed, and opened as a pull
  request into the original PR's branch (merging it updates the original PR);
* drafts, flags, unverified sections and fixes that could not be applied are listed in a comment on
  the original PR, with links to each section.

Re-running on the same PR refreshes the fix branch, the fix PR and the comment instead of duplicating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .apply import ApplyResult, apply_decisions
from .driver import RunResult
from .github import GitHubClient, PullRequest
from .gitutil import commit_paths, current_ref, push_branch, rev_parse, switch, switch_new_branch
from .render import (
    REVIEW_MARKER, LinkContext, ReviewItem, render_fix_pr, render_review_comment, review_items,
)

_SHA_RE = re.compile(r"[0-9a-f]{40}")
BOT_NAME = "doc-healer[bot]"
BOT_EMAIL = "doc-healer@users.noreply.github.com"


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class PRContext:
    repo: str | None  # "owner/name"; None for a local dry run
    number: int | None  # the original PR's number
    head_branch: str  # the original PR's branch; the fix PR targets it
    head_sha: str  # the commit that was analyzed and is checked out


@dataclass(frozen=True)
class WorkflowResult:
    fix_branch: str | None
    fix_pr: PullRequest | None
    apply: ApplyResult
    review_items: list[ReviewItem]
    fix_pr_title: str | None
    fix_pr_body: str | None
    review_comment: str | None  # the comment text, if one is warranted
    comment_url: str | None
    dry_run: bool


def fix_branch_name(ctx: PRContext) -> str:
    suffix = str(ctx.number) if ctx.number else ctx.head_sha[:8]
    return "doc-healer/fix-" + re.sub(r"[^A-Za-z0-9._-]", "-", suffix)


def run_pr_workflow(
    repo: Path,
    result: RunResult,
    ctx: PRContext,
    github: GitHubClient | None,
    *,
    dry_run: bool = False,
    remote: str = "origin",
    branch: str | None = None,
) -> WorkflowResult:
    """Apply the auto-fixes and report the rest. ``repo`` must be checked out at ``ctx.head_sha``.

    With ``dry_run`` nothing is written, pushed or posted; the result still holds the rendered text.
    The checkout is left on the commit and branch it started on.
    """
    if github is None and not dry_run:
        raise WorkflowError("a GitHubClient is required unless dry_run is set")
    links = LinkContext(ctx.repo, ctx.head_sha)
    autos = [d for d in result.decisions if d.mode == "auto_fix"]
    branch = branch or fix_branch_name(ctx)

    applied = ApplyResult([], [], [])
    fix_pr = title = body = None
    if autos:
        preview = apply_decisions(repo, autos, dry_run=True)  # find out what applies before touching git
        applied = preview
        if preview.applied and not dry_run:
            if rev_parse(repo) != ctx.head_sha:
                raise WorkflowError(f"the checkout is not at the analyzed commit {ctx.head_sha[:8]}")
            applied = _commit_fixes(repo, autos, ctx, branch, remote)
        if applied.applied:
            title, body = render_fix_pr(autos, applied, links, pr_number=ctx.number)
            if not dry_run:
                fix_pr = github.open_or_update_pr(title, body, head=branch, base=ctx.head_branch)

    items = review_items(result, applied.skipped)
    comment = comment_url = None
    if items or applied.applied:
        fix_url = fix_pr.url if fix_pr else ("a pull request (not opened in a dry run)" if applied.applied else None)
        comment = render_review_comment(items, links, fix_url)
        if not dry_run and ctx.number is not None:
            comment_url = github.upsert_comment(ctx.number, comment, REVIEW_MARKER)

    return WorkflowResult(
        branch if applied.applied else None, fix_pr, applied, items, title, body, comment, comment_url, dry_run
    )


def _commit_fixes(repo: Path, autos, ctx: PRContext, branch: str, remote: str) -> ApplyResult:
    original = current_ref(repo)
    switch_new_branch(repo, branch)
    try:
        applied = apply_decisions(repo, autos)
        if applied.changed_files:
            lines = "\n".join(f"- {a.file_path}: {a.section_id.split('#', 1)[-1]}" for a in applied.applied)
            count = len(applied.applied)
            message = f"docs: fix {count} stale section{'s' if count != 1 else ''}\n\n{lines}\n"
            commit_paths(repo, applied.changed_files, message, BOT_NAME, BOT_EMAIL)
            push_branch(repo, branch, remote, force=True)  # a re-run replaces the bot's own earlier fix branch
        return applied
    finally:
        switch(repo, original, detach=bool(_SHA_RE.fullmatch(original)))  # a sha means HEAD was detached
