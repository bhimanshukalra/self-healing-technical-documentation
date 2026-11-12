from pathlib import Path

import pytest

from doc_healer.docparser import extract_code_refs, parse_doc, parse_docs

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture(scope="module")
def sections():
    return {s.id: s for s in parse_docs(FIXTURE)}


def test_expected_section_ids(sections):
    assert set(sections) == {
        "README.md#taskr",
        "README.md#taskr > Quick start",
        "docs/usage.md#Usage > Adding tasks",
        "docs/usage.md#Usage > Listing tasks",
        "docs/usage.md#Usage > Completing tasks",
        "docs/usage.md#Usage > HTTP API",
        "docs/configuration.md#Configuration > Environment Variables",
        "docs/configuration.md#Configuration > Settings",
    }


def test_heading_path_and_content(sections):
    s = sections["docs/configuration.md#Configuration > Environment Variables"]
    assert s.heading_path == ("Configuration", "Environment Variables")
    assert s.heading == "Configuration > Environment Variables"
    assert s.content.startswith("taskr reads `TASKR_DB_PATH`")
    assert "Settings" not in s.content  # next section's text is not included


def test_code_refs_for_fixture_sections(sections):
    refs = {sid: s.code_refs for sid, s in sections.items()}
    assert refs["docs/usage.md#Usage > Adding tasks"] == (
        "add_task", "TaskStore", "taskr", "add",
    )
    assert refs["docs/usage.md#Usage > Listing tasks"] == ("list_tasks",)
    assert refs["docs/usage.md#Usage > Completing tasks"] == ("complete_task",)
    assert refs["docs/usage.md#Usage > HTTP API"] == ("GET /tasks", "POST /tasks")
    assert refs["docs/configuration.md#Configuration > Environment Variables"] == (
        "TASKR_DB_PATH", "TASKR_MAX_TASKS", "load_settings",
    )
    assert refs["README.md#taskr > Quick start"] == (
        "TaskStore.add_task", "TaskStore.list_tasks",
    )
    assert refs["README.md#taskr"] == ()


def test_headings_inside_code_fences_are_ignored(tmp_path):
    (tmp_path / "a.md").write_text(
        "# Title\n\nintro\n\n```bash\n# not a heading\nrun_it()\n```\n\n## Next\n\ntext\n"
    )
    result = parse_doc(tmp_path / "a.md", tmp_path)
    assert [s.heading_path for s in result] == [("Title",), ("Title", "Next")]
    assert "# not a heading" in result[0].content
    assert result[0].code_refs == ("run_it",)


def test_sections_without_body_are_skipped(tmp_path):
    (tmp_path / "a.md").write_text("# Parent\n\n## Child\n\nbody\n")
    result = parse_doc(tmp_path / "a.md", tmp_path)
    assert [s.heading_path for s in result] == [("Parent", "Child")]


def test_heading_levels_reset_path(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nx\n\n## B\n\ny\n\n# C\n\nz\n")
    result = parse_doc(tmp_path / "a.md", tmp_path)
    assert [s.heading_path for s in result] == [("A",), ("A", "B"), ("C",)]


def test_line_ranges(sections):
    s = sections["docs/usage.md#Usage > Listing tasks"]
    lines = (FIXTURE / "docs/usage.md").read_text().splitlines()
    assert lines[s.start_line - 1] == "## Listing tasks"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Use `Foo.bar` here", ("Foo.bar",)),
        ("Call `run(x, y=2)` now", ("run",)),
        ("Run `tool sync --force \"a b\"`", ("tool", "sync")),
        ("`GET /items/{id}`", ("GET /items/{id}",)),
        ("`a` and `a` again", ("a",)),
        ("no code here", ()),
    ],
)
def test_extract_code_refs(text, expected):
    assert extract_code_refs(text) == expected
