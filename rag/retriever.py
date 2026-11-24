"""Vector store wrapper around ChromaDB."""
from __future__ import annotations

import uuid

import chromadb

from rag.embeddings import EmbedFn
from rag.ingest import Chunk


class Retriever:
    """One instance = one private, in-memory collection (safe for per-session use)."""

    def __init__(self, embed_fn: EmbedFn, collection_name: str | None = None):
        self.embed_fn = embed_fn
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name or f"documind_{uuid.uuid4().hex[:12]}",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embed_fn([c.text for c in batch]),
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
            )

    def search(self, query: str, k: int = 4) -> list[dict]:
        total = self.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=self.embed_fn([query]),
            n_results=min(k, total),
        )
        return [
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": 1.0 - dist,  # cosine similarity
            }
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
