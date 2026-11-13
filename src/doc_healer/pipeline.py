"""Phase 1 pipeline: code + docs in, link graph out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_project
from .docparser import parse_docs
from .embeddings import Embedder, SentenceTransformerEmbedder, VectorStore
from .index import LinkIndex
from .linker import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    link_embeddings,
    link_heuristic,
    merge_links,
)
from .models import CodeChunk, DocSection


@dataclass(frozen=True)
class BuildResult:
    index: LinkIndex
    chunks: list[CodeChunk]
    sections: list[DocSection]
    heuristic_links: int
    embedding_links: int  # embedding links kept after merging (pairs not already linked by name)
    embedded_chunks: int  # chunks (re-)embedded this run; 0 when the cache was fully warm


def build_index(
    root: Path,
    *,
    index_path: Path | None = None,
    store_path: Path | None = None,
    embedder: Embedder | None = None,
    use_embeddings: bool = True,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> BuildResult:
    """Chunk the code, parse the docs, link them, and optionally save the index.

    Nothing is written unless ``index_path`` (JSON index) or ``store_path`` (persistent
    vector cache) is given. With ``use_embeddings=False`` only name-based links are built
    and no model is loaded.
    """
    chunks = chunk_project(root)
    sections = parse_docs(root)

    heuristic = link_heuristic(chunks, sections)
    embedding: list = []
    embedded_chunks = 0
    if use_embeddings:
        embedder = embedder or SentenceTransformerEmbedder()
        store = VectorStore(embedder, store_path)
        embedded_chunks = store.sync(chunks)
        embedding = link_embeddings(sections, store, embedder, threshold, top_k)

    merged = merge_links(heuristic, embedding)  # heuristic first: it wins on shared pairs
    index = LinkIndex(merged)
    if index_path is not None:
        index.save(index_path)

    return BuildResult(
        index=index,
        chunks=chunks,
        sections=sections,
        heuristic_links=sum(l.method == "heuristic" for l in merged),
        embedding_links=sum(l.method == "embedding" for l in merged),
        embedded_chunks=embedded_chunks,
    )
