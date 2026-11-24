"""Sentence-transformer embeddings. Imported lazily so tests can run without torch."""
from __future__ import annotations

from collections.abc import Callable

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EmbedFn = Callable[[list[str]], list[list[float]]]


def get_embedder(model_name: str = DEFAULT_MODEL) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    return embed
