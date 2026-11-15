import json
import logging
import subprocess

import pytest
from conftest import ScriptedProvider
from test_repair import edits
from test_validation import verdict
from test_verification import reply as verify_reply

from doc_healer import cli
from doc_healer.cli import main
from doc_healer.driver import RunConfig, resolve_index, run_check
from doc_healer.llm import LLMClient, RateLimitError
from doc_healer.pipeline import build_index

ADDING = "docs/usage.md#Usage > Adding tasks"
QUICK = "README.md#taskr > Quick start"
QUOTE = "Use `add_task(title, priority=1)`"
SIMPLE = json.dumps({"label": "simple", "probability": 0.95, "reason": "a rename"})


def llm(*outcomes, **kw):
    provider = ScriptedProvider("groq", *outcomes)
    return LLMClient(provider, sleep=lambda s: None, **kw), provider


def full_script():
    """Replies in pipeline order: verify x2, repair, validate, complexity of the one changed chunk."""
    return [
        verify_reply(False, 0.8, "the README only names the method"),
        verify_reply(True, 0.9, "the parameter is now called name", [QUOTE]),
        edits(("`add_task(title, priority=1)`", "`add_task(name, priority=1)`")),
        verdict(confidence=0.9),
        SIMPLE,
    ]


def commit_index(repo):
    build_index(repo.path, index_path=repo.path / ".doc-healer" / "index.json", use_embeddings=False)
    return repo.commit("add index")


def rename_param(repo):
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')
    return repo.commit("rename")


def config(base, **kw):
    kw.setdefault("use_embeddings", False)
    return RunConfig(base=base, **kw)


def test_full_run_on_a_pr_with_a_committed_index(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    head = rename_param(repo)
    client, provider = llm(*full_script())

    result = run_check(repo.path, config(base, head=head), client)

    assert result.index_source == "committed"
    s = result.summary
    assert (s.suspects, s.accurate, s.stale, s.auto_fix, s.draft, s.flag, s.unverified) == (2, 1, 1, 1, 0, 0, 0)
    assert s.needs_review == 0
    (fix,) = result.decisions
    assert (fix.section_id, fix.mode) == (ADDING, "auto_fix")
    assert "add_task(name, priority=1)" in fix.new_content and "The default priority is 1." in fix.new_content
    assert result.decision_for(ADDING) is fix and result.decision_for(QUICK) is None
    assert result.llm_calls == {"calls": 5, "retries": 0, "fallback_calls": 0, "by_provider": {"groq": 5}}
    assert provider.calls == 5


def test_without_a_committed_index_one_is_built_from_a_temporary_checkout_of_base(taskr_repo):
    repo, base = taskr_repo  # no index committed
    head = rename_param(repo)
    index, source = resolve_index(repo.path, base, use_embeddings=False)
    assert source == "built" and index.docs_for_chunk("taskr/tasks.py::TaskStore.add_task")
    assert repo.git("worktree", "list").count("\n") == 0  # only the main worktree remains

    result = run_check(repo.path, config(base, head=head), llm(*full_script())[0])
    assert (result.index_source, result.summary.auto_fix) == ("built", 1)
    assert not (repo.path / ".doc-healer").exists()  # the main checkout was never touched


def test_the_built_index_reflects_base_not_the_pr(taskr_repo):
    repo, base = taskr_repo
    repo.edit("docs/usage.md", "`complete_task(store, title)`", "`finish_task(store, title)`")
    repo.edit("taskr/tasks.py", "def complete_task(", "def finish_task(")
    repo.commit("rename the function and its docs")
    index, _ = resolve_index(repo.path, base, use_embeddings=False)
    assert index.docs_for_chunk("taskr/tasks.py::complete_task")  # linked as at base
    assert not index.docs_for_chunk("taskr/tasks.py::finish_task")


def test_an_unusable_committed_index_is_rebuilt(taskr_repo, caplog):
    repo, _ = taskr_repo
    (repo.path / ".doc-healer").mkdir()
    (repo.path / ".doc-healer" / "index.json").write_text(json.dumps({"version": 999, "links": []}))
    base = repo.commit("stale index")
    with caplog.at_level(logging.WARNING):
        index, source = resolve_index(repo.path, base, use_embeddings=False)
    assert source == "built" and index.links and "unusable" in caplog.text


def test_with_embeddings_the_fallback_build_uses_the_supplied_embedder(taskr_repo, embedder):
    repo, base = taskr_repo
    index, source = resolve_index(repo.path, base, use_embeddings=True, embedder=embedder)
    assert source == "built" and embedder.calls  # the fake embedder was used, no model download


def test_no_llm_lists_suspects_as_unverified(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    result = run_check(repo.path, config(base, head=rename_param(repo)), client=None)
    s = result.summary
    assert (s.suspects, s.unverified, s.accurate, s.stale, s.needs_review) == (2, 2, 0, 0, 2)
    assert result.decisions == [] and result.corrections == [] and result.llm_calls == {}
    assert "no LLM is configured" in result.verification.results[0].explanation


def test_an_llm_that_is_out_of_capacity_degrades_to_unverified(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    head = rename_param(repo)
    client = LLMClient(ScriptedProvider("groq", RateLimitError()), max_retries=0, sleep=lambda s: None)
    s = run_check(repo.path, config(base, head=head), client).summary
    assert (s.suspects, s.unverified, s.needs_review, s.auto_fix) == (2, 2, 2, 0)


def test_classifier_none_never_auto_fixes_a_modified_chunk(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    head = rename_param(repo)
    client, _ = llm(*full_script()[:4])  # no complexity call is made without a classifier
    result = run_check(repo.path, config(base, head=head, classifier_backend="none"), client)
    assert (result.summary.auto_fix, result.summary.draft) == (0, 1)
    assert "TODO(doc-healer)" in result.decisions[0].new_content


def test_unknown_and_unimplemented_backends_fail_even_without_an_llm(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    with pytest.raises(NotImplementedError):
        run_check(repo.path, config(base, classifier_backend="laya"), None)
    with pytest.raises(ValueError):
        run_check(repo.path, config(base, classifier_backend="bogus"), None)


def test_a_pr_with_no_meaningful_change_makes_no_llm_calls(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    client, provider = llm("should never be asked")
    result = run_check(repo.path, config(base, head=repo.commit("comment")), client)
    assert result.summary.suspects == 0 and result.decisions == []
    assert client.stats.calls == 0 and provider.calls == 0


def test_sections_the_pr_already_edited_are_counted_not_checked(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    repo.edit("docs/usage.md", "Use `add_task(title, priority=1)`", "Use `add_task(name, priority=1)`")
    head = rename_param(repo)
    client, _ = llm(verify_reply(False, 0.8, "fine"))
    result = run_check(repo.path, config(base, head=head), client)
    assert (result.summary.suspects, result.summary.already_updated) == (1, 1)
    assert result.to_dict()["already_updated"] == [ADDING]


def test_to_dict_is_json_ready_and_complete(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    result = run_check(repo.path, config(base, head=rename_param(repo)), llm(*full_script())[0])
    data = json.loads(result.to_json())
    assert data["base"] == base and data["index_source"] == "committed"
    assert data["summary"]["auto_fix"] == 1 and data["summary"]["needs_review"] == 0
    assert data["stale_sections"] == [ADDING] and data["llm_calls"]["calls"] == 5
    by_id = {s["section"]: s for s in data["sections"]}
    assert by_id[QUICK]["status"] == "accurate" and by_id[QUICK]["mode"] is None and by_id[QUICK]["diff"] is None
    fix = by_id[ADDING]
    assert (fix["status"], fix["mode"], fix["file"]) == ("stale", "auto_fix", "docs/usage.md")
    assert "+Use `add_task(name, priority=1)`" in fix["diff"] or "add_task(name, priority=1)" in fix["diff"]
    assert fix["stale_statements"] == [QUOTE]
    assert fix["triggers"] == [{"chunk": "taskr/tasks.py::TaskStore.add_task", "change": "api_changed", "link": "heuristic"}]


def test_flagged_sections_expose_no_diff(taskr_repo):
    repo, _ = taskr_repo
    base = commit_index(repo)
    head = rename_param(repo)
    script = full_script()
    script[3] = verdict(preserved=False, issues=["dropped the example"])  # validation fails
    data = json.loads(run_check(repo.path, config(base, head=head), llm(*script)[0]).to_json())
    fix = next(s for s in data["sections"] if s["section"] == ADDING)
    assert (fix["mode"], fix["diff"]) == ("flag", None) and "dropped the example" in fix["reasons"]


# -- CLI ---------------------------------------------------------------------------------

def test_cli_run_prints_a_summary_and_writes_json(taskr_repo, capsys, monkeypatch, tmp_path):
    repo, _ = taskr_repo
    base = commit_index(repo)
    head = rename_param(repo)
    client, _ = llm(*full_script())
    monkeypatch.setattr(cli, "_make_client", lambda: client)
    out = tmp_path / "result.json"
    capsys.readouterr()
    assert main(["run", str(repo.path), "--base", base, "--head", head, "--no-embeddings", "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "index: committed" in text
    assert "2 suspect sections: 1 accurate, 1 stale (1 auto-fix, 0 draft, 0 flagged), 0 unverified" in text
    assert f"[auto_fix] {ADDING}" in text and f"[accurate] {QUICK}" in text
    assert json.loads(out.read_text())["summary"]["auto_fix"] == 1


def test_cli_run_without_a_key_still_lists_suspects(taskr_repo, capsys, monkeypatch):
    repo, _ = taskr_repo
    base = commit_index(repo)
    head = rename_param(repo)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    capsys.readouterr()
    assert main(["run", str(repo.path), "--base", base, "--head", head, "--no-embeddings"]) == 0
    text = capsys.readouterr().out
    assert "LLM_API_KEY is not set" in text and "2 unverified" in text
