"""The GitHub Action entry point: read the PR event and inputs, run the pipeline, report.

GitHub passes a Docker action its inputs as ``INPUT_<NAME>`` environment variables and the event as
a JSON file. Results go back through ``$GITHUB_OUTPUT`` (outputs), ``$GITHUB_STEP_SUMMARY`` (the run
page) and ``::workflow-command::`` lines on stdout (annotations).

Policy: doc-healer must not break a team's CI because of its own problems. A missing LLM key, a
read-only token on a fork, a failed push or an API hiccup all degrade to a comment or a step summary and
a warning. Only invalid inputs fail the step, and ``fail_on_error`` opts in to failing on anything else.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .driver import DEFAULT_INDEX_PATH, RunConfig, RunResult, run_check
from .github import GitHubClient, GitHubError
from .gitutil import GitError, has_commit, merge_base, rev_parse, switch
from .llm import LLMClient, LLMConfig, ProviderConfig, build_client
from .modes import DEFAULT_CONFIDENCE_THRESHOLD
from .workflow import PRContext, WorkflowResult, run_pr_workflow


class ActionError(Exception):
    """Invalid input or an environment the Action cannot work in. Fails the step."""


@dataclass(frozen=True)
class ActionInputs:
    llm_api_key: str = ""
    llm_provider: str = "groq"
    llm_model: str | None = None
    fallback_provider: str = ""
    fallback_api_key: str = ""
    fallback_model: str | None = None
    max_llm_calls: int | None = None
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    auto_merge: bool = False
    classifier_backend: str = "llm"
    comment_mode: str = "always"
    index_path: str = DEFAULT_INDEX_PATH
    embeddings: bool = True
    github_token: str = ""
    fail_on_error: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "ActionInputs":
        def get(name: str, default: str = "") -> str:
            return env.get(f"INPUT_{name.upper()}", "").strip() or default

        def boolean(name: str, default: bool) -> bool:
            raw = get(name).lower()
            if not raw:
                return default
            if raw not in ("true", "false"):
                raise ActionError(f"input '{name}' must be 'true' or 'false', got {raw!r}")
            return raw == "true"

        def number(name: str, kind, default):
            raw = get(name)
            if not raw:
                return default
            try:
                return kind(raw)
            except ValueError:
                raise ActionError(f"input '{name}' must be a {kind.__name__}, got {raw!r}") from None

        threshold = number("confidence_threshold", float, DEFAULT_CONFIDENCE_THRESHOLD)
        if not 0.0 <= threshold <= 1.0:
            raise ActionError(f"input 'confidence_threshold' must be between 0 and 1, got {threshold}")
        classifier = get("classifier_backend", "llm")
        if classifier not in ("llm", "none"):
            raise ActionError(f"input 'classifier_backend' must be 'llm' or 'none', got {classifier!r}")
        comment_mode = get("comment_mode", "always")
        if comment_mode not in ("always", "needed"):
            raise ActionError(f"input 'comment_mode' must be 'always' or 'needed', got {comment_mode!r}")
        max_calls = number("max_llm_calls", int, None)
        if max_calls is not None and max_calls < 0:
            raise ActionError("input 'max_llm_calls' must not be negative")
        fallback_provider, fallback_key = get("fallback_provider"), get("fallback_api_key")
        if fallback_provider and not fallback_key:
            raise ActionError("input 'fallback_provider' is set but 'fallback_api_key' is not")
        return cls(
            llm_api_key=get("llm_api_key"), llm_provider=get("llm_provider", "groq"), llm_model=get("llm_model") or None,
            fallback_provider=fallback_provider, fallback_api_key=fallback_key,
            fallback_model=get("fallback_model") or None, max_llm_calls=max_calls,
            confidence_threshold=threshold, auto_merge=boolean("auto_merge", False),
            classifier_backend=classifier, comment_mode=comment_mode, index_path=get("index_path", DEFAULT_INDEX_PATH),
            embeddings=boolean("embeddings", True), github_token=get("github_token"),
            fail_on_error=boolean("fail_on_error", False),
        )

    def llm_config(self) -> LLMConfig | None:
        """None when no key was given; the run then lists suspect sections without verifying them."""
        if not self.llm_api_key:
            return None
        fallback = (
            ProviderConfig(self.fallback_provider, self.fallback_api_key, self.fallback_model)
            if self.fallback_provider else None
        )
        return LLMConfig(ProviderConfig(self.llm_provider, self.llm_api_key, self.llm_model), fallback, self.max_llm_calls)


@dataclass(frozen=True)
class PullRequestEvent:
    repo: str  # owner/name of the repository the workflow runs in
    number: int
    base_sha: str
    head_sha: str
    head_ref: str
    is_fork: bool


def parse_event(path: Path) -> PullRequestEvent | None:
    """The PR described by a GitHub event payload, or None if it is not a pull request event."""
    event = json.loads(path.read_text(encoding="utf-8"))
    pr = event.get("pull_request")
    if not pr:
        return None
    base_repo = (pr.get("base") or {}).get("repo", {}).get("full_name") or event.get("repository", {}).get("full_name", "")
    head_repo_name = ((pr.get("head") or {}).get("repo") or {}).get("full_name")
    return PullRequestEvent(
        repo=base_repo, number=int(pr["number"]), base_sha=pr["base"]["sha"], head_sha=pr["head"]["sha"],
        head_ref=pr["head"]["ref"],
        is_fork=head_repo_name is None or head_repo_name != base_repo,  # a deleted fork has no head repo
    )


def main(
    env: Mapping[str, str] | None = None,
    *,
    llm_client: LLMClient | None = None,
    github: GitHubClient | None = None,
    stdout=None,
) -> int:
    """Run the Action. ``llm_client`` and ``github`` let tests inject fakes; in production they are built from inputs."""
    env = os.environ if env is None else env
    out = stdout or sys.stdout

    def say(line: str) -> None:
        print(line, file=out, flush=True)

    try:
        inputs = ActionInputs.from_env(env)
    except ActionError as exc:
        say(_annotation("error", str(exc)))
        return 1
    try:
        return _run(env, inputs, llm_client, github, say)
    except ActionError as exc:
        say(_annotation("error", str(exc)))
        return 1
    except Exception as exc:  # never let our own bug break the team's CI unless they asked for that
        say(_annotation("error" if inputs.fail_on_error else "warning", f"doc-healer failed: {type(exc).__name__}: {exc}"))
        say(traceback.format_exc())
        return 1 if inputs.fail_on_error else 0


def _run(env, inputs: ActionInputs, llm_client, github, say) -> int:
    event_path = env.get("GITHUB_EVENT_PATH")
    event = parse_event(Path(event_path)) if event_path and Path(event_path).is_file() else None
    if event is None:
        say(_annotation("notice", "doc-healer only runs on pull_request events; nothing to do."))
        return 0

    workspace = Path(env.get("GITHUB_WORKSPACE") or ".").resolve()
    base, head = _resolve_range(workspace, event)

    client = llm_client
    if client is None:
        cfg = inputs.llm_config()
        if cfg is None:
            say(_annotation("notice", "No llm_api_key was given: suspect sections will be listed but not verified or fixed."))
        else:
            client = build_client(cfg)

    config = RunConfig(
        base=base, head=head, index_path=inputs.index_path, classifier_backend=inputs.classifier_backend,
        confidence_threshold=inputs.confidence_threshold, use_embeddings=inputs.embeddings,
        store_path=Path(tempfile.gettempdir()) / "doc-healer-chroma",  # a cache that must not land in the repo
    )
    result = run_check(workspace, config, client)

    if github is None and inputs.github_token and event.repo:
        github = GitHubClient(inputs.github_token, event.repo, env.get("GITHUB_API_URL", "https://api.github.com"))
    dry_run = github is None
    if dry_run:
        say(_annotation("warning", "No github_token: nothing will be pushed or commented."))
    ctx = PRContext(event.repo, event.number, event.head_ref, event.head_sha)
    workflow = run_pr_workflow(
        workspace, result, ctx, github, dry_run=dry_run, comment_mode=inputs.comment_mode, auto_merge=inputs.auto_merge,
        no_fix_reason="pull requests from forks cannot receive fix pull requests (the token is read-only)" if event.is_fork else None,
    )
    _report(env, inputs, result, workflow, say)
    return 0


def _resolve_range(workspace: Path, event: PullRequestEvent) -> tuple[str, str]:
    """The PR's own changes: from the merge base of its base and head, to its head. Leaves the checkout on the head."""
    for sha in (event.base_sha, event.head_sha):
        if not has_commit(workspace, sha):
            raise ActionError(
                f"commit {sha[:8]} is not in this checkout. Use actions/checkout with `fetch-depth: 0` "
                "and `ref: ${{ github.event.pull_request.head.sha }}`."
            )
    try:
        base = merge_base(workspace, event.base_sha, event.head_sha)
    except GitError as exc:
        raise ActionError(f"could not find where the PR branched off ({exc}); use `fetch-depth: 0` in actions/checkout") from exc
    if rev_parse(workspace) != event.head_sha:
        switch(workspace, event.head_sha, detach=True)  # the default checkout is a merge commit, not the PR head
    return base, event.head_sha


# -- reporting ---------------------------------------------------------------------------

def _report(env, inputs, result: RunResult, workflow: WorkflowResult, say) -> None:
    s = result.summary
    say(
        f"doc-healer: {s.suspects} suspect sections: {s.accurate} accurate, {s.stale} stale "
        f"({s.auto_fix} auto-fix, {s.draft} draft, {s.flag} flagged), {s.unverified} unverified."
    )
    if workflow.fix_pr:
        say(f"Fix pull request: {workflow.fix_pr.url}" + (" (merged)" if workflow.merged else ""))
    if workflow.comment_url:
        say(f"Summary comment: {workflow.comment_url}")
    for error in workflow.errors:
        say(_annotation("warning", f"doc-healer: {error}"))
    if s.needs_review:
        say(_annotation("notice", f"{s.needs_review} documentation section(s) need review."))

    if path := env.get("GITHUB_STEP_SUMMARY"):  # the same report on the run page, even when posting failed
        with open(path, "a", encoding="utf-8") as f:
            f.write((workflow.comment or "No documentation sections were affected by this PR.") + "\n")
    if path := env.get("GITHUB_OUTPUT"):
        write_outputs(Path(path), action_outputs(result, workflow))


def action_outputs(result: RunResult, workflow: WorkflowResult) -> dict[str, str]:
    """The Action's outputs, each a single-line string (JSON for lists)."""
    data = result.to_dict()
    applied = {a.section_id for a in workflow.apply.applied}
    corrections = [
        {
            "section": d.section_id, "file": d.verification.suspect.section.file_path, "mode": d.mode,
            "confidence": d.confidence, "applied": d.section_id in applied,
        }
        for d in result.decisions if d.mode != "flag"
    ]
    return {
        "stale_sections": json.dumps(data["stale_sections"]),
        "corrections": json.dumps(corrections),
        "summary": json.dumps(data["summary"]),
        "fix_pr_url": workflow.fix_pr.url if workflow.fix_pr else "",
    }


def write_outputs(path: Path, outputs: Mapping[str, str]) -> None:
    """Append outputs in the ``$GITHUB_OUTPUT`` format. A random delimiter makes any value safe."""
    with path.open("a", encoding="utf-8") as f:
        for name, value in outputs.items():
            delimiter = f"ghadelimiter_{uuid.uuid4().hex}"
            f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def _annotation(level: str, message: str) -> str:
    """A workflow command; GitHub requires %, CR and LF to be escaped in the message."""
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::{level}::{escaped}"


def cli() -> None:
    sys.exit(main())
