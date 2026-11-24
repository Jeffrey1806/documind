import hashlib
import math

from rag.ingest import Chunk
from rag.retriever import Retriever


def fake_embed(texts, dim=256):
    """Deterministic bag-of-words embedding so tests need no ML model."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def make_chunks():
    texts = {
        "a": "kubernetes pods run containers inside a cluster",
        "b": "terraform provisions cloud infrastructure as code",
        "c": "prometheus collects metrics and grafana draws dashboards",
    }
    return [Chunk(t, "doc.txt", 0, f"doc::{k}") for k, t in texts.items()]


def test_search_returns_most_relevant_first():
    r = Retriever(fake_embed)
    r.add(make_chunks())
    results = r.search("which tool provisions infrastructure", k=2)
    assert len(results) == 2
    assert "terraform" in results[0]["text"]
    assert results[0]["source"] == "doc.txt"


def test_empty_store_returns_nothing():
    assert Retriever(fake_embed).search("anything") == []


def test_k_larger_than_store_is_clamped():
    r = Retriever(fake_embed)
    r.add(make_chunks())
    assert len(r.search("metrics", k=50)) == 3


def test_reset_clears_collection():
    r = Retriever(fake_embed)
    r.add(make_chunks())
    assert r.count() == 3
    r.reset()
    assert r.count() == 0


def test_instances_are_isolated():
    a, b = Retriever(fake_embed), Retriever(fake_embed)
    a.add(make_chunks())
    assert b.count() == 0
