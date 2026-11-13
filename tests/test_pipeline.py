import json
from pathlib import Path

from doc_healer.cli import main
from doc_healer.index import LinkIndex
from doc_healer.linker import link_heuristic
from doc_healer.chunker import chunk_project
from doc_healer.docparser import parse_docs
from doc_healer.pipeline import build_index

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"


def _heuristic():
    return link_heuristic(chunk_project(FIXTURE), parse_docs(FIXTURE))


def test_build_without_embeddings_is_heuristic_only_and_loads_no_model():
    result = build_index(FIXTURE, use_embeddings=False)
    assert result.index.links == LinkIndex(_heuristic()).links
    assert (result.embedding_links, result.embedded_chunks) == (0, 0)
    assert result.heuristic_links == len(result.index.links)
    assert len(result.chunks) == 11 and len(result.sections) == 8


def test_build_matches_hand_labeled_links_and_adds_embedding_extras(tmp_path, embedder):
    result = build_index(FIXTURE, embedder=embedder, threshold=0.3)
    expected = json.loads((FIXTURE / "expected_links.json").read_text())
    for sid, ids in expected.items():
        found = {l.code_chunk_id for l in result.index.chunks_for_doc(sid) if l.method == "heuristic"}
        assert found == set(ids)
    assert result.embedding_links > 0
    assert result.heuristic_links + result.embedding_links == len(result.index.links)
    # a pair linked by name is never duplicated as an embedding link
    pairs = [(l.code_chunk_id, l.doc_section_id) for l in result.index.links]
    assert len(pairs) == len(set(pairs))


def test_build_saves_loadable_index(tmp_path, embedder):
    path = tmp_path / "out" / "index.json"
    result = build_index(FIXTURE, index_path=path, embedder=embedder)
    assert LinkIndex.load(path).links == result.index.links


def test_build_writes_nothing_by_default(tmp_path, embedder):
    build_index(FIXTURE, embedder=embedder)
    assert not (FIXTURE / ".doc-healer").exists()


def test_second_build_reuses_the_vector_cache(tmp_path, embedder):
    store = tmp_path / "chroma"
    first = build_index(FIXTURE, store_path=store, embedder=embedder)
    second = build_index(FIXTURE, store_path=store, embedder=embedder)
    assert first.embedded_chunks == 11
    assert second.embedded_chunks == 0
    assert second.index.links == first.index.links


def test_empty_project(tmp_path, embedder):
    result = build_index(tmp_path, embedder=embedder)
    assert result.index.links == [] and result.chunks == [] and result.sections == []


def test_cli_build_no_embeddings(tmp_path, capsys):
    index = tmp_path / "index.json"
    code = main(["build", str(FIXTURE), "--index", str(index), "--no-embeddings"])
    assert code == 0
    out = capsys.readouterr().out
    assert "11 code chunks, 8 doc sections" in out and "0 embedding links" in out
    assert LinkIndex.load(index).links == LinkIndex(_heuristic()).links
