"""Embeddings and the Chroma-backed vector store used for embedding links."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .models import CodeChunk, DocSection

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SOURCE_CHARS = 1000  # keep chunk text within the model's input window


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalized vector per text."""
        ...


class SentenceTransformerEmbedder:
    """Local Hugging Face model; no API key. The model loads on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


def chunk_text(chunk: CodeChunk) -> str:
    parts = [f"{chunk.kind} {chunk.name}", chunk.signature]
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.source[:MAX_SOURCE_CHARS])
    return "\n".join(parts)


def section_text(section: DocSection) -> str:
    return f"{section.heading}\n{section.content}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class VectorStore:
    """Persistent cosine-similarity store of code chunk embeddings (ChromaDB).

    Vectors are cached by content hash, so unchanged chunks are never re-embedded.
    """

    def __init__(self, embedder: Embedder, path: Path | None = None, collection: str = "code_chunks"):
        import chromadb

        client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self._collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder

    def sync(self, chunks: Sequence[CodeChunk]) -> int:
        """Make the store match ``chunks``; returns how many were (re-)embedded."""
        texts = {c.id: chunk_text(c) for c in chunks}
        hashes = {cid: content_hash(t) for cid, t in texts.items()}

        existing = self._collection.get(include=["metadatas"])
        stored = {i: (m or {}).get("hash") for i, m in zip(existing["ids"], existing["metadatas"])}

        stale_ids = [cid for cid in stored if cid not in texts]
        if stale_ids:
            self._collection.delete(ids=stale_ids)

        todo = [cid for cid in texts if stored.get(cid) != hashes[cid]]
        if todo:
            vectors = self._embedder.embed([texts[cid] for cid in todo])
            self._collection.upsert(
                ids=todo,
                embeddings=vectors,
                metadatas=[{"hash": hashes[cid]} for cid in todo],
            )
        return len(todo)

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Nearest chunk ids with cosine similarity, best first."""
        n = min(k, self._collection.count())
        if n == 0:
            return []
        res = self._collection.query(query_embeddings=[vector], n_results=n)
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]
