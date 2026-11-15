import subprocess

import pytest
from test_driver import commit_index, config, full_script, llm
from test_github import FakeGitHub
from test_verification import reply as verify_reply

from doc_healer.driver import run_check
from doc_healer.render import FIX_MARKER, REVIEW_MARKER
from doc_healer.workflow import PRContext, WorkflowError, fix_branch_name, run_pr_workflow

ADDING = "docs/usage.md#Usage > Adding tasks"
FIX_BRANCH = "doc-healer/fix-5"


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def pr(taskr_repo, tmp_path):
    """A PR: `main` has a committed index, `feature` renames a parameter, `origin` is a real bare repo."""
    repo, _ = taskr_repo
    base = commit_index(repo)
    origin = tmp_path / "origin.git"
    sh("git", "init", "--bare", "-q", str(origin), cwd=tmp_path)
    repo.git("remote", "add", "origin", str(origin))
    repo.git("push", "-q", "origin", "main")
    repo.git("switch", "-q", "-c", "feature")
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')
    head = repo.commit("rename")
    repo.git("push", "-q", "origin", "feature")
    ctx = PRContext("o/r", 5, "feature", head)
    return repo, base, head, origin, ctx


def analyze(repo, base, head, script=None, **kw):
    return run_check(repo.path, config(base, head=head, **kw), llm(*(script or full_script()))[0])


def remote_file(origin, branch, path):
    return sh("git", "--git-dir", str(origin), "show", f"{branch}:{path}", cwd=origin)


def remote_branches(origin):
    return sh("git", "--git-dir", str(origin), "branch", "--format=%(refname:short)", cwd=origin).split()


def paths(fake):
    return [(m, p) for m, p, *_ in fake.requests]


def test_auto_fix_is_committed_pushed_and_opened_as_a_pr_into_the_original_branch(pr):
    repo, base, head, origin, ctx = pr
    fake = FakeGitHub()
    out = run_pr_workflow(repo.path, analyze(repo, base, head), ctx, fake.client())

    assert out.fix_branch == FIX_BRANCH and not out.dry_run
    assert "add_task(name, priority=1)" in remote_file(origin, FIX_BRANCH, "docs/usage.md")
    assert "add_task(title, priority=1)" in remote_file(origin, "feature", "docs/usage.md")  # the PR branch is untouched
    assert sh("git", "--git-dir", str(origin), "log", "-1", "--format=%an", FIX_BRANCH, cwd=origin) == "doc-healer[bot]"
    assert sh("git", "--git-dir", str(origin), "log", "-1", "--format=%s", FIX_BRANCH, cwd=origin) == "docs: fix 1 stale section"

    pr_body = next(b for m, p, _, b, _h in fake.requests if (m, p) == ("POST", "/repos/o/r/pulls"))
    assert (pr_body["head"], pr_body["base"], pr_body["draft"]) == (FIX_BRANCH, "feature", False)
    assert pr_body["title"] == "docs: update 1 section for changes in #5" and FIX_MARKER in pr_body["body"]
    assert out.fix_pr.number == 7 and out.fix_pr.created and out.fix_pr_title == pr_body["title"]
    assert not out.review_items and out.comment_url  # a short comment points at the fix PR
    assert "https://github.com/o/r/pull/7" in out.review_comment and REVIEW_MARKER in out.review_comment


def test_the_checkout_is_left_exactly_as_it_was(pr):
    repo, base, head, _, ctx = pr
    run_pr_workflow(repo.path, analyze(repo, base, head), ctx, FakeGitHub().client())
    assert sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=repo.path) == "feature"
    assert sh("git", "rev-parse", "HEAD", cwd=repo.path) == head
    assert sh("git", "status", "--porcelain", "--untracked-files=no", cwd=repo.path) == ""
    assert "add_task(title, priority=1)" in (repo.path / "docs/usage.md").read_text()


def test_a_detached_checkout_is_restored_too(pr):
    repo, base, head, _, ctx = pr
    repo.git("switch", "-q", "--detach", head)
    run_pr_workflow(repo.path, analyze(repo, base, head), ctx, FakeGitHub().client())
    assert sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=repo.path) == "HEAD"  # still detached
    assert sh("git", "rev-parse", "HEAD", cwd=repo.path) == head


def test_rerunning_refreshes_the_branch_pr_and_comment_instead_of_duplicating(pr):
    repo, base, head, origin, ctx = pr
    first = FakeGitHub()
    run_pr_workflow(repo.path, analyze(repo, base, head), ctx, first.client())

    second = FakeGitHub(
        open_prs=[{"number": 7, "html_url": "https://github.com/o/r/pull/7"}],
        comments=[{"id": 99, "body": f"{REVIEW_MARKER}\nold"}],
    )
    out = run_pr_workflow(repo.path, analyze(repo, base, head), ctx, second.client())
    assert out.fix_pr.number == 7 and not out.fix_pr.created
    assert ("PATCH", "/repos/o/r/pulls/7") in paths(second) and ("POST", "/repos/o/r/pulls") not in paths(second)
    assert ("PATCH", "/repos/o/r/issues/comments/99") in paths(second)
    assert remote_branches(origin).count(FIX_BRANCH) == 1


def test_drafts_are_listed_in_a_comment_and_nothing_is_committed(pr):
    repo, base, head, origin, ctx = pr
    result = analyze(repo, base, head, script=full_script()[:4], classifier_backend="none")
    assert result.summary.draft == 1
    fake = FakeGitHub()
    out = run_pr_workflow(repo.path, result, ctx, fake.client())
    assert out.fix_branch is None and out.fix_pr is None and out.apply.applied == []
    assert FIX_BRANCH not in remote_branches(origin)
    assert [m for m, p in paths(fake) if p.endswith("/pulls")] == []  # no PR calls at all
    comment = out.review_comment
    assert "### Needs human review (1)" in comment and "draft for review" in comment
    assert f"https://github.com/o/r/blob/{head}/docs/usage.md#L" in comment
    assert "TODO" not in comment and "Suggested edit" in comment and "add_task(name, priority=1)" in comment
    assert out.comment_url and out.review_items[0].kind == "draft"


def test_unverified_and_flagged_sections_are_listed(pr):
    repo, base, head, _, ctx = pr
    no_llm = run_check(repo.path, config(base, head=head), client=None)
    fake = FakeGitHub()
    out = run_pr_workflow(repo.path, no_llm, ctx, fake.client())
    assert [i.kind for i in out.review_items] == ["unverified", "unverified"]
    assert "could not be checked" in out.review_comment and "no LLM is configured" in out.review_comment

    script = full_script()
    script[3] = '{"accurate": true, "preserved": false, "style_consistent": true, "confidence": 0.9, "issues": ["dropped the example"]}'
    flagged = run_pr_workflow(repo.path, analyze(repo, base, head, script), ctx, FakeGitHub().client())
    assert [i.kind for i in flagged.review_items] == ["flag"] and "dropped the example" in flagged.review_comment


def test_a_fix_that_no_longer_applies_is_reported_and_nothing_is_pushed(pr):
    repo, base, head, origin, ctx = pr
    result = analyze(repo, base, head)
    path = repo.path / "docs/usage.md"
    path.write_text(path.read_text().replace("Use `add_task(title, priority=1)`", "Call `add_task(title, priority=1)` now"))
    fake = FakeGitHub()
    out = run_pr_workflow(repo.path, result, ctx, fake.client())
    assert out.fix_branch is None and out.fix_pr is None and FIX_BRANCH not in remote_branches(origin)
    assert [i.kind for i in out.review_items] == ["not_applied"]
    assert "no longer matches" in out.review_comment
    assert sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=repo.path) == "feature"


def test_dry_run_touches_nothing_but_still_renders(pr):
    repo, base, head, origin, ctx = pr
    fake = FakeGitHub()
    out = run_pr_workflow(repo.path, analyze(repo, base, head), ctx, None, dry_run=True)
    assert out.dry_run and out.fix_branch == FIX_BRANCH and len(out.apply.applied) == 1
    assert out.fix_pr is None and out.comment_url is None and fake.requests == []
    assert "add_task(name, priority=1)" in out.fix_pr_body
    assert "not opened in a dry run" in out.review_comment and REVIEW_MARKER in out.review_comment
    assert FIX_BRANCH not in remote_branches(origin)
    assert sh("git", "branch", "--list", FIX_BRANCH, cwd=repo.path) == ""
    assert "add_task(title, priority=1)" in (repo.path / "docs/usage.md").read_text()


def test_nothing_to_do_posts_nothing(pr):
    repo, base, head, _, ctx = pr
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    new_head = repo.commit("comment")
    fake = FakeGitHub()
    result = run_check(repo.path, config(head, head=new_head), llm("unused")[0])  # only the comment-only commit
    out = run_pr_workflow(repo.path, result, PRContext("o/r", 5, "feature", new_head), fake.client())
    assert out.fix_branch is None and out.review_comment is None and fake.requests == []


def test_without_a_pr_number_the_comment_is_rendered_but_not_posted(pr):
    repo, base, head, _, _ = pr
    fake = FakeGitHub()
    out = run_pr_workflow(repo.path, analyze(repo, base, head), PRContext("o/r", None, "feature", head), fake.client())
    assert out.fix_branch == f"doc-healer/fix-{head[:8]}" and out.review_comment and out.comment_url is None
    assert not any("/comments" in p for _, p in paths(fake))


def test_guards(pr):
    repo, base, head, _, ctx = pr
    result = analyze(repo, base, head)
    with pytest.raises(WorkflowError, match="GitHubClient is required"):
        run_pr_workflow(repo.path, result, ctx, None)
    repo.edit("README.md", "A tiny task manager.", "A tiny, tidy task manager.")
    repo.commit("someone pushed more")
    with pytest.raises(WorkflowError, match="not at the analyzed commit"):
        run_pr_workflow(repo.path, result, ctx, FakeGitHub().client())


def test_branch_names_are_safe():
    assert fix_branch_name(PRContext("o/r", 12, "f", "a" * 40)) == "doc-healer/fix-12"
    assert fix_branch_name(PRContext("o/r", None, "f", "abcdef1234567890")) == "doc-healer/fix-abcdef12"
