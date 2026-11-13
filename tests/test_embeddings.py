from pathlib import Path

from doc_healer.chunker import chunk_project
from doc_healer.docparser import parse_docs
from doc_healer.embeddings import VectorStore, chunk_text, section_text
from doc_healer.linker import link_embeddings, link_heuristic, merge_links
from doc_healer.models import CodeChunk, DocSection, Link

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"


def _chunk(cid, name, doc, source="") -> CodeChunk:
    return CodeChunk(cid, "function", "x.py", name, f"def {name}()", doc, source, 1, 2)


def _section(sid, heading, content) -> DocSection:
    return DocSection(sid, "d.md", (heading,), content, (), 1, 2)


def test_text_builders():
    c = _chunk("x.py::f", "f", "Does a thing.", "def f(): pass")
    assert chunk_text(c).splitlines() == ["function f", "def f()", "Does a thing.", "def f(): pass"]
    s = _section("d.md#H", "H", "body text")
    assert section_text(s) == "H\nbody text"


def test_links_semantically_similar_pairs_only(tmp_path, embedder):
    chunks = [
        _chunk("a.py::send_email", "send_email", "Send an email message to a recipient."),
        _chunk("a.py::resize_image", "resize_image", "Resize an image to a given width."),
    ]
    store = VectorStore(embedder, tmp_path / "chroma")
    store.sync(chunks)
    section = _section("d.md#Mail", "Mail", "How to send an email message to a recipient.")
    links = link_embeddings([section], store, embedder, threshold=0.5)
    assert [(l.code_chunk_id, l.method) for l in links] == [("a.py::send_email", "embedding")]
    assert 0.5 <= links[0].score <= 1.0


def test_threshold_filters_everything_when_high(tmp_path, embedder):
    chunks = [_chunk("a.py::send_email", "send_email", "Send an email message.")]
    store = VectorStore(embedder, tmp_path / "chroma")
    store.sync(chunks)
    section = _section("d.md#Mail", "Mail", "Completely unrelated gardening advice.")
    assert link_embeddings([section], store, embedder, threshold=0.65) == []


def test_sync_only_reembeds_changed_chunks(tmp_path, embedder):
    path = tmp_path / "chroma"
    a = _chunk("a.py::a", "a", "alpha")
    b = _chunk("a.py::b", "b", "beta")
    assert VectorStore(embedder, path).sync([a, b]) == 2
    # new process-like reopen: same persisted store, nothing changed
    assert VectorStore(embedder, path).sync([a, b]) == 0
    b2 = _chunk("a.py::b", "b", "beta changed")
    assert VectorStore(embedder, path).sync([a, b2]) == 1


def test_sync_removes_deleted_chunks(tmp_path, embedder):
    store = VectorStore(embedder, tmp_path / "chroma")
    a, b = _chunk("a.py::a", "a", "alpha apple"), _chunk("a.py::b", "b", "beta banana")
    store.sync([a, b])
    store.sync([a])
    ids = [cid for cid, _ in store.query(embedder.embed(["alpha apple beta banana"])[0], 10)]
    assert ids == ["a.py::a"]


def test_empty_store_and_no_sections(tmp_path, embedder):
    store = VectorStore(embedder, tmp_path / "chroma")
    assert store.query([0.0] * embedder.dim, 5) == []
    assert link_embeddings([], store, embedder) == []


def test_merge_prefers_heuristic_and_keeps_extras():
    heuristic = [Link("c1", "d1", "heuristic", 0.9)]
    embedding = [Link("c1", "d1", "embedding", 0.99), Link("c2", "d1", "embedding", 0.7)]
    assert merge_links(heuristic, embedding) == [
        Link("c1", "d1", "heuristic", 0.9),
        Link("c2", "d1", "embedding", 0.7),
    ]


def test_end_to_end_on_fixture_keeps_all_heuristic_links(tmp_path, embedder):
    chunks, sections = chunk_project(FIXTURE), parse_docs(FIXTURE)
    store = VectorStore(embedder, tmp_path / "chroma")
    store.sync(chunks)
    heuristic = link_heuristic(chunks, sections)
    merged = merge_links(heuristic, link_embeddings(sections, store, embedder, threshold=0.3))
    assert set(heuristic) <= set(merged)
