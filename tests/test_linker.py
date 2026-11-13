import json
from pathlib import Path

import pytest

from doc_healer.chunker import chunk_project
from doc_healer.docparser import parse_docs
from doc_healer.index import SCHEMA_VERSION, IndexVersionError, LinkIndex
from doc_healer.linker import (
    SCORE_AMBIGUOUS_SHORT,
    SCORE_QUALIFIED,
    SCORE_UNIQUE_SHORT,
    link_heuristic,
)
from doc_healer.models import CodeChunk, DocSection, Link

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture(scope="module")
def links():
    return link_heuristic(chunk_project(FIXTURE), parse_docs(FIXTURE))


def test_heuristic_links_match_hand_labeled_ground_truth(links):
    expected = json.loads((FIXTURE / "expected_links.json").read_text())
    actual: dict[str, set[str]] = {sid: set() for sid in expected}
    for link in links:
        actual[link.doc_section_id].add(link.code_chunk_id)
    assert actual == {sid: set(ids) for sid, ids in expected.items()}


def test_scores(links):
    score = {(l.doc_section_id, l.code_chunk_id): l.score for l in links}
    usage = "docs/usage.md#Usage > "
    assert score[(usage + "Adding tasks", "taskr/tasks.py::TaskStore")] == SCORE_QUALIFIED
    assert score[(usage + "Adding tasks", "taskr/tasks.py::TaskStore.add_task")] == SCORE_UNIQUE_SHORT
    assert score[(usage + "HTTP API", "taskr/api.py::get_tasks")] == SCORE_QUALIFIED
    assert all(l.method == "heuristic" for l in links)


def _chunk(cid: str, name: str, kind="function", source="", extra=None) -> CodeChunk:
    return CodeChunk(cid, kind, "x.py", name, f"def {name}()", None, source, 1, 2, extra or {})


def _section(refs: tuple[str, ...]) -> DocSection:
    return DocSection("d.md#S", "d.md", ("S",), "text", refs, 1, 2)


def test_ambiguous_short_name_links_all_candidates_with_lower_score():
    chunks = [_chunk("a.py::A.run", "A.run"), _chunk("b.py::B.run", "B.run")]
    result = link_heuristic(chunks, [_section(("run",))])
    assert {(l.code_chunk_id, l.score) for l in result} == {
        ("a.py::A.run", SCORE_AMBIGUOUS_SHORT),
        ("b.py::B.run", SCORE_AMBIGUOUS_SHORT),
    }


def test_qualified_name_beats_ambiguity():
    chunks = [_chunk("a.py::A.run", "A.run"), _chunk("b.py::B.run", "B.run")]
    result = link_heuristic(chunks, [_section(("A.run",))])
    assert [(l.code_chunk_id, l.score) for l in result] == [("a.py::A.run", SCORE_QUALIFIED)]


def test_config_fields_link_by_field_name_not_by_prose():
    cfg = _chunk(
        "c.py::Cfg", "Cfg", kind="config", source="class Cfg:\n    '''Settings for taskr.'''\n    port: int = 1",
        extra={"fields": "port"},
    )
    assert [l.code_chunk_id for l in link_heuristic([cfg], [_section(("port",))])] == ["c.py::Cfg"]
    assert link_heuristic([cfg], [_section(("taskr",))]) == []


def test_unknown_refs_produce_no_links():
    assert link_heuristic([_chunk("a.py::f", "f")], [_section(("nothing", "GET /x"))]) == []


def test_pair_keeps_highest_score():
    chunks = [_chunk("a.py::A.run", "A.run")]
    result = link_heuristic(chunks, [_section(("run", "A.run"))])
    assert [l.score for l in result] == [SCORE_QUALIFIED]


def test_index_round_trip_and_queries(links, tmp_path):
    index = LinkIndex(links)
    path = tmp_path / "nested" / "index.json"
    index.save(path)
    loaded = LinkIndex.load(path)
    assert loaded.links == index.links
    suspects = loaded.docs_for_chunk("taskr/tasks.py::TaskStore.add_task")
    assert {l.doc_section_id for l in suspects} == {
        "README.md#taskr > Quick start",
        "docs/usage.md#Usage > Adding tasks",
    }
    assert [l.code_chunk_id for l in loaded.chunks_for_doc("docs/usage.md#Usage > Listing tasks")] == [
        "taskr/tasks.py::TaskStore.list_tasks"
    ]
    assert loaded.docs_for_chunk("nope") == []


def test_index_output_is_deterministic():
    a = Link("c1", "d1", "heuristic", 1.0)
    b = Link("c2", "d1", "heuristic", 0.9)
    assert LinkIndex([a, b]).to_json() == LinkIndex([b, a]).to_json()


def test_index_rejects_other_schema_version(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": SCHEMA_VERSION + 1, "links": []}))
    with pytest.raises(IndexVersionError):
        LinkIndex.load(path)
