import textwrap

import pytest

from doc_healer.classifier import Classification
from doc_healer.changes import ChunkChange, detect_chunk_changes, diff_chunks
from doc_healer.chunker import chunk_source
from doc_healer.index import LinkIndex
from doc_healer.meaningful import filter_changes, is_private, judge, meaningful_only
from doc_healer.models import Link

BASE = textwrap.dedent('''
    from dataclasses import dataclass

    @dataclass
    class Settings:
        """Config."""
        limit: int = 10

    class Store:
        """A store."""

        def add(self, title: str, priority: int = 1) -> dict:
            """Add an item."""
            item = {"title": title, "priority": priority}
            return item

        def _internal(self):
            return 1

    def helper(x):
        """Help."""
        return x + 1

    @app.get("/items")
    def list_items():
        return []
''')


def change(new_src: str, chunk: str, old_src: str = BASE) -> ChunkChange:
    changes = [c for c in diff_chunks(chunk_source(old_src, "m.py"), chunk_source(new_src, "m.py"))
               if c.chunk_id == f"m.py::{chunk}"]
    assert len(changes) == 1, changes
    return changes[0]


def verdict(new_src, chunk, **kw):
    v = judge(change(new_src, chunk), **kw)
    return v.meaningful, v.reason


class FakeClassifier:
    def __init__(self, label, probability=0.9):
        self.label, self.probability, self.calls = label, probability, []

    def classify_meaningful(self, change):
        self.calls.append(change.chunk_id)
        return Classification(self.label, self.probability)

    def classify_complexity(self, change):
        raise AssertionError("not used by the filter")


def test_comment_only_edit_is_cosmetic():
    new = BASE.replace("        item = {", "        # build it\n        item = {")
    assert verdict(new, "Store.add") == (False, "cosmetic")


def test_whitespace_only_edit_is_cosmetic():
    new = BASE.replace('{"title": title, "priority": priority}', '{ "title":title,"priority":priority }')
    assert verdict(new, "Store.add") == (False, "cosmetic")


def test_docstring_only_edit():
    assert verdict(BASE.replace('"""Add an item."""', '"""Add a new item."""'), "Store.add") == (False, "docstring_only")


def test_renamed_parameter_is_an_api_change():
    new = BASE.replace("title: str, priority", "name: str, priority").replace('"title": title', '"title": name')
    assert verdict(new, "Store.add") == (True, "api_changed")


def test_changed_default_in_signature_is_an_api_change():
    assert verdict(BASE.replace("priority: int = 1)", "priority: int = 2)"), "Store.add") == (True, "api_changed")


def test_return_annotation_change_is_an_api_change():
    assert verdict(BASE.replace(") -> dict:", ") -> list:"), "Store.add") == (True, "api_changed")


def test_decorator_route_change_is_an_api_change():
    assert verdict(BASE.replace('"/items"', '"/things"'), "list_items") == (True, "api_changed")


def test_config_default_change_is_meaningful():
    assert verdict(BASE.replace("limit: int = 10", "limit: int = 99"), "Settings") == (True, "config_changed")


def test_added_and_removed_public_chunks_are_meaningful():
    added = BASE + "\ndef new_feature():\n    return 1\n"
    assert verdict(added, "new_feature") == (True, "added")
    removed = BASE.replace("def helper(x):\n    \"\"\"Help.\"\"\"\n    return x + 1\n", "")
    assert verdict(removed, "helper", ) == (True, "removed")


def test_private_chunks_are_skipped_unless_docs_link_to_them():
    new = BASE.replace("return 1\n\ndef helper", "return 2\n\ndef helper")
    assert verdict(new, "Store._internal") == (False, "private")
    assert verdict(new, "Store._internal", linked_ids={"m.py::Store._internal"}) == (True, "body_changed")


def test_is_private():
    assert is_private("m.py::_f") and is_private("m.py::C._m") and is_private("m.py::_C.run")
    assert not is_private("m.py::C.__init__") and not is_private("m.py::C.run")


def test_body_change_without_classifier_is_kept_with_low_confidence():
    v = judge(change(BASE.replace("x + 1", "x + 2"), "helper"))
    assert (v.meaningful, v.reason, v.confidence) == (True, "body_changed", 0.5)


@pytest.mark.parametrize("label, kept, reason", [("meaningful", True, "body_changed"), ("internal", False, "internal_refactor")])
def test_classifier_decides_ambiguous_body_changes(label, kept, reason):
    clf = FakeClassifier(label, 0.8)
    v = judge(change(BASE.replace("x + 1", "x + 2"), "helper"), classifier=clf)
    assert (v.meaningful, v.reason) == (kept, reason)
    assert v.confidence == 0.8
    assert clf.calls == ["m.py::helper"]


def test_classifier_is_only_asked_about_ambiguous_changes():
    clf = FakeClassifier("meaningful")
    changes = [
        change(BASE.replace("        item = {", "        # c\n        item = {"), "Store.add"),   # cosmetic
        change(BASE.replace("limit: int = 10", "limit: int = 3"), "Settings"),        # config
        change(BASE.replace("x + 1", "x + 2"), "helper"),                              # ambiguous
    ]
    filter_changes(changes, classifier=clf)
    assert clf.calls == ["m.py::helper"]


def test_meaningful_only_drops_skipped_changes():
    changes = [
        change(BASE.replace("        item = {", "        # c\n        item = {"), "Store.add"),
        change(BASE.replace("limit: int = 10", "limit: int = 3"), "Settings"),
    ]
    kept = meaningful_only(filter_changes(changes))
    assert [v.change.chunk_id for v in kept] == ["m.py::Settings"]


def test_end_to_end_on_a_git_diff(taskr_repo):
    repo, base = taskr_repo
    repo.edit("taskr/tasks.py", "        self._tasks.append(task)", "        # keep it\n        self._tasks.append(task)")
    repo.edit("taskr/config.py", "max_tasks: int = 100", "max_tasks: int = 500")
    repo.edit("taskr/tasks.py", "def complete_task(", "def finish_task(")
    head = repo.commit("c")
    verdicts = filter_changes(detect_chunk_changes(repo.path, base, head))
    assert {(v.change.chunk_id, v.reason) for v in verdicts} == {
        ("taskr/tasks.py::TaskStore.add_task", "cosmetic"),
        ("taskr/config.py::Settings", "config_changed"),
        ("taskr/tasks.py::finish_task", "added"),
        ("taskr/tasks.py::complete_task", "removed"),
    }
    assert {v.change.chunk_id for v in meaningful_only(verdicts)} == {
        "taskr/config.py::Settings", "taskr/tasks.py::finish_task", "taskr/tasks.py::complete_task",
    }


def test_linked_chunk_ids():
    index = LinkIndex([Link("c1", "d1", "heuristic", 1.0), Link("c1", "d2", "embedding", 0.8), Link("c2", "d1", "heuristic", 0.9)])
    assert index.linked_chunk_ids() == {"c1", "c2"}
