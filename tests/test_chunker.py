import logging
from pathlib import Path

import pytest

from doc_healer.chunker import chunk_project

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture(scope="module")
def chunks():
    return {c.id: c for c in chunk_project(FIXTURE)}


def test_expected_chunk_ids(chunks):
    assert set(chunks) == {
        "taskr/config.py::Settings",
        "taskr/config.py::load_settings",
        "taskr/tasks.py::TaskStore",
        "taskr/tasks.py::TaskStore.__init__",
        "taskr/tasks.py::TaskStore.add_task",
        "taskr/tasks.py::TaskStore.list_tasks",
        "taskr/tasks.py::complete_task",
        "taskr/api.py::get_tasks",
        "taskr/api.py::create_task",
        "taskr/cli.py::cli",
        "taskr/cli.py::add",
    }


def test_function_signature_and_docstring(chunks):
    c = chunks["taskr/tasks.py::TaskStore.add_task"]
    assert c.kind == "function"
    assert c.signature == "def add_task(self, title: str, priority: int=1) -> dict"
    assert c.docstring == "Add a task with the given title and priority."
    assert c.name == "TaskStore.add_task"


def test_kinds(chunks):
    assert chunks["taskr/config.py::Settings"].kind == "config"
    assert chunks["taskr/tasks.py::TaskStore"].kind == "class"
    assert chunks["taskr/cli.py::add"].kind == "cli"
    assert chunks["taskr/cli.py::cli"].kind == "cli"


def test_endpoints(chunks):
    get = chunks["taskr/api.py::get_tasks"]
    assert get.kind == "endpoint"
    assert get.extra == {"http_method": "GET", "route": "/tasks"}
    assert get.signature.startswith("async def get_tasks(")
    assert chunks["taskr/api.py::create_task"].extra["http_method"] == "POST"


def test_nested_functions_are_not_chunked(chunks):
    assert not any("_match" in cid for cid in chunks)


def test_test_files_and_syntax_errors_skipped(chunks, caplog):
    assert not any("test_tasks" in cid for cid in chunks)
    assert not any("broken" in cid for cid in chunks)
    with caplog.at_level(logging.WARNING):
        chunk_project(FIXTURE)
    assert "broken.py" in caplog.text


def test_ids_are_stable_when_code_moves(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    first = {c.id for c in chunk_project(tmp_path)}
    (tmp_path / "m.py").write_text("# header\n\n\ndef f():\n    return 1\n")
    assert {c.id for c in chunk_project(tmp_path)} == first


def test_line_range_includes_decorators(chunks):
    c = chunks["taskr/api.py::get_tasks"]
    lines = (FIXTURE / "taskr/api.py").read_text().splitlines()
    assert lines[c.start_line - 1].startswith("@app.get")
