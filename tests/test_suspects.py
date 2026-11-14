from doc_healer.analysis import analyze_changes, docs_at, updated_section_ids
from doc_healer.cli import main
from doc_healer.index import LinkIndex
from doc_healer.meaningful import Verdict
from doc_healer.changes import ChunkChange
from doc_healer.models import DocSection, Link
from doc_healer.pipeline import build_index
from doc_healer.suspects import find_suspects

ADDING = "docs/usage.md#Usage > Adding tasks"
QUICK = "README.md#taskr > Quick start"
COMPLETING = "docs/usage.md#Usage > Completing tasks"
SETTINGS = "docs/configuration.md#Configuration > Settings"
ADD_TASK = "taskr/tasks.py::TaskStore.add_task"


def run(repo, base, index):
    head = repo.commit("change")
    return analyze_changes(repo.path, base, head, index)


def ids(suspects):
    return [s.section.id for s in suspects]


def rename_param(repo):
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')


def test_api_change_flags_every_doc_section_linked_to_the_chunk(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    rename_param(repo)
    report = run(repo, base, index).report
    assert ids(report.suspects) == [QUICK, ADDING]  # qualified-name link (1.0) outranks short-name (0.9)
    assert [s.priority for s in report.suspects] == [1.0, 0.9]
    (trigger,) = report.suspects[1].triggers
    assert (trigger.verdict.reason, trigger.link.code_chunk_id) == ("api_changed", ADD_TASK)
    assert report.already_updated == [] and report.unresolved == []


def test_removed_function_flags_its_docs(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    repo.edit("taskr/tasks.py", "def complete_task(", "def finish_task(")
    report = run(repo, base, index).report
    assert ids(report.suspects) == [COMPLETING]
    assert report.suspects[0].triggers[0].verdict.reason == "removed"


def test_config_change_flags_the_settings_docs_only(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    repo.edit("taskr/config.py", "max_tasks: int = 100", "max_tasks: int = 500")
    assert ids(run(repo, base, index).report.suspects) == [SETTINGS]


def test_cosmetic_and_unlinked_changes_flag_nothing(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    repo.write("taskr/extra.py", "def brand_new():\n    return 1\n")  # added; no docs link to it yet
    analysis = run(repo, base, index)
    assert analysis.report.suspects == []
    assert len(analysis.changes) == 2


def test_sections_the_pr_also_edited_are_reported_separately(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    rename_param(repo)
    repo.edit("docs/usage.md", "Use `add_task(title, priority=1)`", "Use `add_task(name, priority=1)`")
    report = run(repo, base, index).report
    assert ids(report.suspects) == [QUICK]
    assert ids(report.already_updated) == [ADDING]


def test_links_to_deleted_sections_are_unresolved_not_suspects(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    (repo.path / "docs/usage.md").unlink()
    rename_param(repo)
    report = run(repo, base, index).report
    assert ids(report.suspects) == [QUICK]
    assert {l.doc_section_id for l in report.unresolved} == {ADDING}


def test_a_section_hit_by_several_changes_is_listed_once(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    rename_param(repo)
    repo.edit("taskr/tasks.py", "def list_tasks(self, status: str", "def list_tasks(self, state: str")
    repo.edit("taskr/tasks.py", 't["status"] == status', 't["status"] == state')
    report = run(repo, base, index).report
    quick = next(s for s in report.suspects if s.section.id == QUICK)
    assert sorted(t.link.code_chunk_id for t in quick.triggers) == [ADD_TASK, "taskr/tasks.py::TaskStore.list_tasks"]
    assert ids(report.suspects).count(QUICK) == 1


def test_docs_at_and_updated_ids(taskr_repo):
    repo, base = taskr_repo
    before = docs_at(repo.path, base)
    assert ADDING in before and len(before) == 8
    repo.edit("docs/usage.md", "The default priority is 1.", "The default priority is 2.")
    repo.write("docs/new.md", "# New\n\nbody\n")
    after = docs_at(repo.path, repo.commit("c"))
    assert updated_section_ids(before, after) == {ADDING, "docs/new.md#New"}


def _verdict(cid, reason, meaningful=True):
    return Verdict(ChunkChange(cid, "modified", None, None), meaningful, reason, 1.0)


def _section(sid):
    return DocSection(sid, "d.md", ("S",), "text", (), 1, 2)


def test_find_suspects_skips_non_meaningful_and_applies_min_score():
    index = LinkIndex([Link("c1", "d.md#A", "embedding", 0.72), Link("c2", "d.md#B", "heuristic", 1.0)])
    sections = {"d.md#A": _section("d.md#A"), "d.md#B": _section("d.md#B")}
    verdicts = [_verdict("c1", "body_changed"), _verdict("c2", "cosmetic", meaningful=False)]
    assert ids(find_suspects(verdicts, index, sections).suspects) == ["d.md#A"]
    assert find_suspects(verdicts, index, sections, min_score=0.8).suspects == []


def test_priority_orders_by_link_score_and_change_weight():
    index = LinkIndex([Link("c1", "d.md#A", "heuristic", 0.9), Link("c2", "d.md#B", "heuristic", 0.9)])
    sections = {"d.md#A": _section("d.md#A"), "d.md#B": _section("d.md#B")}
    verdicts = [_verdict("c1", "body_changed"), _verdict("c2", "removed")]
    report = find_suspects(verdicts, index, sections)
    assert [(s.section.id, s.priority) for s in report.suspects] == [("d.md#B", 0.9), ("d.md#A", 0.54)]


def test_cli_check_lists_suspects(taskr_repo, capsys):
    repo, base = taskr_repo
    assert main(["build", str(repo.path), "--no-embeddings"]) == 0
    capsys.readouterr()
    rename_param(repo)
    head = repo.commit("c")
    assert main(["check", str(repo.path), "--base", base, "--head", head]) == 0
    out = capsys.readouterr().out
    assert "2 suspect doc sections" in out
    assert f"api_changed: {ADD_TASK}" in out and ADDING in out and QUICK in out


def test_cli_check_without_index(taskr_repo, capsys):
    repo, base = taskr_repo
    assert main(["check", str(repo.path), "--base", base]) == 2
    assert "run 'doc-healer build' first" in capsys.readouterr().out
