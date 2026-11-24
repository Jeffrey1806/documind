"""Evaluate retrieval quality and tune chunk size / top-k.

Metric: hit@k = share of questions where at least one of the top-k retrieved
chunks contains the expected keyword (case-insensitive). MRR is also reported.

Usage:
    python -m eval.eval                       # default settings
    python -m eval.eval --sweep               # grid over chunk sizes and k
    python -m eval.eval --docs my_docs/ --questions my_questions.json
    python -m eval.eval --answers             # also check LLM answers (needs GROQ_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.embeddings import DEFAULT_MODEL, get_embedder
from rag.generator import answer, make_client
from rag.ingest import SUPPORTED_EXTENSIONS, build_chunks
from rag.retriever import Retriever


def load_chunks(docs_dir: Path, chunk_size: int, overlap: int):
    chunks = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            chunks += build_chunks(path.name, path.read_bytes(), chunk_size, overlap)
    return chunks


def evaluate(retriever: Retriever, questions: list[dict], k: int, check_answers: bool = False):
    hits, rr_sum, answer_hits = 0, 0.0, 0
    client = make_client() if check_answers else None
    for q in questions:
        expected = q["expected"].lower()
        results = retriever.search(q["question"], k=k)
        rank = next((i for i, r in enumerate(results, 1) if expected in r["text"].lower()), None)
        if rank:
            hits += 1
            rr_sum += 1 / rank
        if check_answers:
            reply = answer(q["question"], results, client=client)
            answer_hits += expected in reply.lower()
    n = len(questions)
    out = {"hit_at_k": hits / n, "mrr": rr_sum / n}
    if check_answers:
        out["answer_match"] = answer_hits / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=Path("data/sample_docs"))
    ap.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sweep", action="store_true", help="grid search chunk size x top-k")
    ap.add_argument("--answers", action="store_true", help="also grade LLM answers")
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text())
    embed = get_embedder(DEFAULT_MODEL)

    sizes = [300, 500, 800, 1200] if args.sweep else [args.chunk_size]
    ks = [1, 2, 4, 6] if args.sweep else [args.k]
    rows = []
    for size in sizes:
        overlap = min(args.overlap, size // 4)
        retriever = Retriever(embed)
        retriever.add(load_chunks(args.docs, size, overlap))
        for k in ks:
            m = evaluate(retriever, questions, k, args.answers)
            rows.append({"chunk_size": size, "overlap": overlap, "k": k, **m})

    print(f"{'chunk':>6} {'overlap':>8} {'k':>3} {'hit@k':>7} {'MRR':>6}" + ("  answer" if args.answers else ""))
    for r in rows:
        line = f"{r['chunk_size']:>6} {r['overlap']:>8} {r['k']:>3} {r['hit_at_k']:>7.0%} {r['mrr']:>6.2f}"
        if args.answers:
            line += f"  {r['answer_match']:.0%}"
        print(line)
    best = max(rows, key=lambda r: (r["hit_at_k"], r["mrr"], -r["chunk_size"]))
    print(
        f"\nBest: chunk_size={best['chunk_size']}, k={best['k']} "
        f"-> hit@k {best['hit_at_k']:.0%}, MRR {best['mrr']:.2f}"
    )
    Path("eval/results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
