import io
import json
import re

import pytest
from conftest import ScriptedProvider
from test_driver import full_script, llm
from test_github import FakeGitHub
from test_workflow import ADDING, FIX_BRANCH, make_pr, remote_branches, remote_file, sh

from doc_healer.action import (
    ActionError, ActionInputs, _annotation, action_outputs, main, parse_event, write_outputs,
)
from doc_healer.llm import LLMClient


@pytest.fixture
def pr(taskr_repo, tmp_path):
    return make_pr(taskr_repo, tmp_path)


def event(base, head, number=5, head_repo="o/r", head_ref="feature"):
    return {
        "repository": {"full_name": "o/r"},
        "pull_request": {
            "number": number,
            "base": {"sha": base, "ref": "main", "repo": {"full_name": "o/r"}},
            "head": {"sha": head, "ref": head_ref, "repo": {"full_name": head_repo} if head_repo else None},
        },
    }


class Run:
    """One invocation of the Action with its files, env and captured output."""

    def __init__(self, tmp_path, repo, payload, **inputs):
        self.event_path = tmp_path / "event.json"
        self.event_path.write_text(json.dumps(payload))
        self.output_path, self.summary_path = tmp_path / "output", tmp_path / "summary.md"
        self.env = {
            "GITHUB_EVENT_PATH": str(self.event_path), "GITHUB_WORKSPACE": str(repo.path),
            "GITHUB_OUTPUT": str(self.output_path), "GITHUB_STEP_SUMMARY": str(self.summary_path),
            "INPUT_LLM_API_KEY": "key", "INPUT_EMBEDDINGS": "false", "INPUT_GITHUB_TOKEN": "tok",
        }
        self.env.update({f"INPUT_{k.upper()}": str(v) for k, v in inputs.items()})
        self.out = io.StringIO()

    def go(self, llm_client=None, github=None):
        self.code = main(self.env, llm_client=llm_client, github=github, stdout=self.out)
        return self

    @property
    def log(self):
        return self.out.getvalue()

    @property
    def outputs(self):
        text = self.output_path.read_text() if self.output_path.exists() else ""
        return {m.group(1): m.group(3) for m in re.finditer(r"^(\w+)<<(\w+)\n(.*)\n\2$", text, re.M)}

    @property
    def summary(self):
        return self.summary_path.read_text() if self.summary_path.exists() else ""


def scripted(*outcomes, **kw):
    return LLMClient(ScriptedProvider("groq", *outcomes), sleep=lambda s: None, **kw)


def run_pr(tmp_path, pr, *, llm_client="default", github="default", **inputs):
    repo, base, head, origin, ctx = pr
    run = Run(tmp_path, repo, event(base, head), **inputs)
    return run.go(
        llm_client=scripted(*full_script()) if llm_client == "default" else llm_client,
        github=FakeGitHub().client() if github == "default" else github,
    )


# -- the happy path ----------------------------------------------------------------------

def test_a_pr_gets_a_fix_branch_a_fix_pr_a_comment_outputs_and_a_step_summary(pr, tmp_path):
    repo, base, head, origin, _ = pr
    fake = FakeGitHub()
    run = Run(tmp_path, repo, event(base, head)).go(llm_client=scripted(*full_script()), github=fake.client())

    assert run.code == 0
    assert "2 suspect sections: 1 accurate, 1 stale (1 auto-fix, 0 draft, 0 flagged), 0 unverified." in run.log
    assert "Fix pull request: https://github.com/o/r/pull/7" in run.log
    assert "add_task(name, priority=1)" in remote_file(origin, FIX_BRANCH, "docs/usage.md")

    out = run.outputs
    assert json.loads(out["stale_sections"]) == [ADDING]
    (correction,) = json.loads(out["corrections"])
    assert correction == {"section": ADDING, "file": "docs/usage.md", "mode": "auto_fix",
                          "confidence": correction["confidence"], "applied": True}
    assert json.loads(out["summary"])["auto_fix"] == 1 and json.loads(out["summary"])["needs_review"] == 0
    assert out["fix_pr_url"] == "https://github.com/o/r/pull/7"
    assert "**Doc Check Results:** 1 section verified accurate, 1 auto-fixed" in run.summary


def test_the_checkout_is_left_on_the_pr_head(pr, tmp_path):
    repo, base, head, _, _ = pr
    run_pr(tmp_path, pr)
    assert sh("git", "rev-parse", "HEAD", cwd=repo.path) == head
    assert sh("git", "status", "--porcelain", "--untracked-files=no", cwd=repo.path) == ""


def test_a_merge_commit_checkout_is_switched_to_the_pr_head(pr, tmp_path):
    repo, base, head, origin, _ = pr
    repo.git("switch", "-q", "main")  # as if actions/checkout left us somewhere else
    run = run_pr(tmp_path, pr)
    assert run.code == 0 and FIX_BRANCH in remote_branches(origin)
    assert sh("git", "rev-parse", "HEAD", cwd=repo.path) == head


def test_only_the_prs_own_changes_are_analyzed_even_when_the_base_branch_moved_on(pr, tmp_path):
    repo, base, head, origin, _ = pr
    repo.git("switch", "-q", "main")
    repo.edit("taskr/config.py", "max_tasks: int = 100", "max_tasks: int = 500")  # unrelated, on main only
    moved = repo.commit("main moves on")
    repo.git("switch", "-q", "feature")
    run = Run(tmp_path, repo, event(moved, head)).go(llm_client=scripted(*full_script()), github=FakeGitHub().client())
    assert run.code == 0
    assert "2 suspect sections" in run.log  # the add_task rename only; main's config change is not in the PR
    assert json.loads(run.outputs["stale_sections"]) == [ADDING]


def test_a_summary_is_posted_even_when_nothing_was_affected(pr, tmp_path):
    repo, base, head, _, _ = pr
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    new_head = repo.commit("comment")
    fake = FakeGitHub()
    run = Run(tmp_path, repo, event(head, new_head)).go(llm_client=scripted("unused"), github=fake.client())
    assert run.code == 0 and "no documentation sections were affected" in run.summary
    assert any(m == "POST" and p.endswith("/comments") for m, p, *_ in fake.requests)
    assert json.loads(run.outputs["stale_sections"]) == [] and run.outputs["fix_pr_url"] == ""


def test_comment_mode_needed_stays_quiet(pr, tmp_path):
    repo, base, head, _, _ = pr
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    new_head = repo.commit("comment")
    fake = FakeGitHub()
    Run(tmp_path, repo, event(head, new_head), comment_mode="needed").go(llm_client=scripted("x"), github=fake.client())
    assert fake.requests == []


# -- degrading instead of breaking CI ----------------------------------------------------

def test_without_an_llm_key_suspects_are_listed_unverified_and_the_step_succeeds(pr, tmp_path):
    repo, base, head, origin, _ = pr
    fake = FakeGitHub()
    run = Run(tmp_path, repo, event(base, head), llm_api_key="").go(github=fake.client())
    assert run.code == 0
    assert "::notice::No llm_api_key was given" in run.log
    summary = json.loads(run.outputs["summary"])
    assert (summary["suspects"], summary["unverified"], summary["needs_review"]) == (2, 2, 2)
    assert "No LLM was configured" in run.summary and FIX_BRANCH not in remote_branches(origin)


def test_without_a_github_token_it_warns_and_changes_nothing(pr, tmp_path):
    repo, base, head, origin, _ = pr
    run = Run(tmp_path, repo, event(base, head), github_token="").go(llm_client=scripted(*full_script()))
    assert run.code == 0 and "::warning::No github_token" in run.log
    assert FIX_BRANCH not in remote_branches(origin) and run.outputs["fix_pr_url"] == ""
    assert "not opened in a dry run" in run.summary


def test_a_fork_pr_gets_no_fix_pr_but_the_fixes_are_listed(pr, tmp_path):
    repo, base, head, origin, _ = pr
    run = Run(tmp_path, repo, event(base, head, head_repo="someone/fork")).go(
        llm_client=scripted(*full_script()), github=FakeGitHub().client())
    assert run.code == 0 and FIX_BRANCH not in remote_branches(origin)
    assert "cannot receive fix pull requests" in run.summary and "Suggested edit" in run.summary
    (correction,) = json.loads(run.outputs["corrections"])
    assert correction["applied"] is False


def test_github_errors_become_warnings_not_failures(pr, tmp_path):
    fake = FakeGitHub(fail={("POST", "/repos/o/r/pulls"): 403, ("POST", "/repos/o/r/issues/5/comments"): 403})
    run = run_pr(tmp_path, pr, github=fake.client())
    assert run.code == 0
    assert "::warning::doc-healer: the fix pull request could not be opened" in run.log
    assert "::warning::doc-healer: the summary comment could not be posted" in run.log
    assert "Doc Check Results" in run.summary  # the report still reaches the run page


def test_an_unexpected_crash_warns_by_default_and_fails_only_when_asked(pr, tmp_path):
    boom = lambda: scripted(RuntimeError("provider exploded"))
    soft = run_pr(tmp_path, pr, llm_client=boom())
    assert soft.code == 0 and "::warning::doc-healer failed: RuntimeError: provider exploded" in soft.log
    hard_dir = tmp_path / "hard"
    hard_dir.mkdir()
    hard = run_pr(hard_dir, pr, llm_client=boom(), fail_on_error="true")
    assert hard.code == 1 and "::error::doc-healer failed" in hard.log


# -- things that should stop the step ----------------------------------------------------

@pytest.mark.parametrize("inputs, message", [
    (dict(confidence_threshold="1.5"), "between 0 and 1"),
    (dict(confidence_threshold="high"), "must be a float"),
    (dict(auto_merge="maybe"), "'true' or 'false'"),
    (dict(max_llm_calls="-1"), "must not be negative"),
    (dict(max_llm_calls="lots"), "must be a int"),
    (dict(classifier_backend="laya"), "'llm' or 'none'"),
    (dict(comment_mode="sometimes"), "'always' or 'needed'"),
    (dict(fallback_provider="gemini"), "fallback_api_key"),
])
def test_invalid_inputs_fail_the_step_with_an_error_annotation(pr, tmp_path, inputs, message):
    run = run_pr(tmp_path, pr, **inputs)
    assert run.code == 1 and "::error::" in run.log and message in run.log


def test_a_checkout_without_history_explains_how_to_fix_it(pr, tmp_path):
    repo, base, head, _, _ = pr
    run = Run(tmp_path, repo, event(base, "f" * 40)).go(llm_client=scripted("x"), github=FakeGitHub().client())
    assert run.code == 1 and "fetch-depth: 0" in run.log and "::error::" in run.log


def test_events_that_are_not_pull_requests_are_skipped(pr, tmp_path):
    repo, *_ = pr
    run = Run(tmp_path, repo, {"ref": "refs/heads/main", "repository": {"full_name": "o/r"}}).go()
    assert run.code == 0 and "only runs on pull_request events" in run.log
    missing = Run(tmp_path, repo, {}).go()
    (tmp_path / "event.json").unlink()
    assert main({"GITHUB_WORKSPACE": str(repo.path)}, stdout=io.StringIO()) == 0
    assert missing.code == 0


# -- units -------------------------------------------------------------------------------

def test_defaults():
    inputs = ActionInputs.from_env({})
    assert (inputs.llm_provider, inputs.confidence_threshold, inputs.auto_merge, inputs.comment_mode) == ("groq", 0.8, False, "always")
    assert (inputs.index_path, inputs.embeddings, inputs.fail_on_error, inputs.llm_config()) == (".doc-healer/index.json", True, False, None)


def test_llm_config_is_built_from_inputs():
    cfg = ActionInputs.from_env({
        "INPUT_LLM_API_KEY": "k", "INPUT_LLM_PROVIDER": "openai", "INPUT_LLM_MODEL": "m", "INPUT_MAX_LLM_CALLS": "9",
        "INPUT_FALLBACK_PROVIDER": "gemini", "INPUT_FALLBACK_API_KEY": "k2",
    }).llm_config()
    assert (cfg.primary.provider, cfg.primary.model, cfg.fallback.provider, cfg.max_calls) == ("openai", "m", "gemini", 9)
    assert "k2" not in repr(cfg.fallback)


@pytest.mark.parametrize("head_repo, fork", [("o/r", False), ("someone/fork", True), (None, True)])
def test_parse_event_detects_forks(tmp_path, head_repo, fork):
    path = tmp_path / "e.json"
    path.write_text(json.dumps(event("a" * 40, "b" * 40, head_repo=head_repo)))
    parsed = parse_event(path)
    assert (parsed.repo, parsed.number, parsed.head_ref, parsed.is_fork) == ("o/r", 5, "feature", fork)
    path.write_text(json.dumps({"ref": "refs/heads/main"}))
    assert parse_event(path) is None


def test_annotations_escape_newlines_and_percent_signs():
    assert _annotation("warning", "50% done\nnext\r") == "::warning::50%25 done%0Anext%0D"


def test_outputs_use_a_random_delimiter_so_any_value_is_safe(tmp_path):
    path = tmp_path / "out"
    value = "line one\nline two\nEOF\nghadelimiter_fake"
    write_outputs(path, {"a": value, "b": "plain"})
    text = path.read_text()
    first = re.match(r"a<<(ghadelimiter_[0-9a-f]{32})\n", text)
    assert first and f"\n{first.group(1)}\n" in text and text.count(first.group(1)) == 2
    assert "b<<" in text and value in text


def test_action_outputs_have_the_documented_names(pr, tmp_path):
    repo, base, head, _, _ = pr
    from doc_healer.driver import RunConfig, run_check
    from doc_healer.workflow import run_pr_workflow
    result = run_check(repo.path, RunConfig(base=base, head=head, use_embeddings=False), llm(*full_script())[0])
    wf = run_pr_workflow(repo.path, result, pr[4], None, dry_run=True)
    outs = action_outputs(result, wf)
    assert set(outs) == {"stale_sections", "corrections", "summary", "fix_pr_url"}
    assert all("\n" not in v for v in outs.values()) and outs["fix_pr_url"] == ""
    assert json.loads(outs["corrections"])[0]["applied"] is True  # in a dry run the fix applies cleanly
