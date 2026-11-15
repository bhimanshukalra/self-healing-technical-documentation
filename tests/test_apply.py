import pytest

from doc_healer.apply import apply_decisions
from doc_healer.docparser import parse_markdown
from doc_healer.modes import Decision
from doc_healer.suspects import Suspect
from doc_healer.verification import Verification

DOC = """# Guide

Intro text.

## Alpha

Alpha is 1.

## Beta

Beta is 2.

## Gamma

Gamma is 3.
"""


def decision(section, new_content, mode="auto_fix"):
    v = Verification(Suspect(section, (), 1.0), "stale", "wrong", (), 0.9)
    return Decision(v, mode, 0.9, "simple", (), new_content)


def sections(text, path="d.md"):
    return {s.heading_path[-1]: s for s in parse_markdown(text, path)}


def write(tmp_path, text, name="d.md", newline="\n"):
    with (tmp_path / name).open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", newline))
    return tmp_path / name


def test_a_section_is_replaced_and_everything_else_is_untouched(tmp_path):
    path = write(tmp_path, DOC)
    s = sections(DOC)
    result = apply_decisions(tmp_path, [decision(s["Beta"], "Beta is 20.")])
    assert path.read_text() == DOC.replace("Beta is 2.", "Beta is 20.")
    assert [(a.section_id, a.file_path) for a in result.applied] == [("d.md#Guide > Beta", "d.md")]
    assert result.skipped == [] and result.changed_files == ["d.md"]


def test_several_sections_in_one_file_even_when_replacements_change_line_counts(tmp_path):
    path = write(tmp_path, DOC)
    s = sections(DOC)
    apply_decisions(tmp_path, [
        decision(s["Alpha"], "Alpha is 10.\nIt grew a second line.\nAnd a third."),
        decision(s["Gamma"], "Gamma is 30."),
        decision(s["Beta"], "Beta is 20."),
    ])
    assert path.read_text() == DOC.replace("Alpha is 1.", "Alpha is 10.\nIt grew a second line.\nAnd a third.").replace(
        "Beta is 2.", "Beta is 20.").replace("Gamma is 3.", "Gamma is 30.")


def test_identical_text_in_another_section_is_not_touched(tmp_path):
    text = "# T\n\n## One\n\nSame words.\n\n## Two\n\nSame words.\n"
    path = write(tmp_path, text)
    apply_decisions(tmp_path, [decision(sections(text)["Two"], "Changed words.")])
    assert path.read_text() == "# T\n\n## One\n\nSame words.\n\n## Two\n\nChanged words.\n"


def test_crlf_files_keep_their_line_endings(tmp_path):
    path = write(tmp_path, DOC, newline="\r\n")
    apply_decisions(tmp_path, [decision(sections(DOC)["Alpha"], "Alpha is 10.\nTwo lines.")])
    raw = path.read_bytes().decode()
    assert "\n" not in raw.replace("\r\n", "")  # every newline is still CRLF
    assert "Alpha is 10.\r\nTwo lines." in raw and "Gamma is 3." in raw


def test_the_trailing_newline_is_preserved(tmp_path):
    for text in (DOC, DOC.rstrip("\n")):
        path = write(tmp_path, text)
        apply_decisions(tmp_path, [decision(sections(text)["Gamma"], "Gamma is 30.")])
        assert path.read_text() == text.replace("Gamma is 3.", "Gamma is 30.")


def test_a_section_that_changed_since_analysis_is_skipped_not_corrupted(tmp_path):
    s = sections(DOC)
    edited = DOC.replace("Beta is 2.", "Beta is 2 (edited by someone).")
    path = write(tmp_path, edited)
    result = apply_decisions(tmp_path, [decision(s["Beta"], "Beta is 20.")])
    assert path.read_text() == edited
    assert result.applied == [] and result.changed_files == []
    assert "no longer matches" in result.skipped[0].reason


def test_missing_files_and_paths_outside_the_repo_are_skipped(tmp_path):
    inner = tmp_path / "repo"
    inner.mkdir()
    (tmp_path / "outside.md").write_text(DOC)
    for rel in ("absent.md", "../outside.md"):
        section = parse_markdown(DOC, rel)[1]
        result = apply_decisions(inner, [decision(section, "x")])
        assert result.applied == [] and "not a file inside the repository" in result.skipped[0].reason
    assert (tmp_path / "outside.md").read_text() == DOC


def test_only_auto_fix_decisions_apply_by_default(tmp_path):
    path = write(tmp_path, DOC)
    s = sections(DOC)
    result = apply_decisions(tmp_path, [decision(s["Alpha"], "x", mode="draft"), decision(s["Beta"], None)])
    assert result.applied == [] and path.read_text() == DOC
    assert apply_decisions(tmp_path, [decision(s["Alpha"], "A!", mode="draft")], modes=("draft",)).applied


def test_dry_run_reports_without_writing(tmp_path):
    path = write(tmp_path, DOC)
    result = apply_decisions(tmp_path, [decision(sections(DOC)["Beta"], "Beta is 20.")], dry_run=True)
    assert len(result.applied) == 1 and result.changed_files == ["d.md"]
    assert path.read_text() == DOC


@pytest.mark.parametrize("nested", ["docs/guide.md", "a/b/c.md"])
def test_nested_paths(tmp_path, nested):
    (tmp_path / nested).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / nested).write_text(DOC)
    section = sections(DOC, nested)["Alpha"]
    result = apply_decisions(tmp_path, [decision(section, "Alpha is 10.")])
    assert result.changed_files == [nested] and "Alpha is 10." in (tmp_path / nested).read_text()
