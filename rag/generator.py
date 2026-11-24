"""Answer generation: build a grounded prompt and call the LLM (Groq by default)."""
from __future__ import annotations

import os

DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
NOT_FOUND = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = (
    "You are DocuMind, an assistant that answers questions using ONLY the provided "
    "context passages. Cite the passages you used like [1], [2]. "
    f"If the context does not contain the answer, reply exactly: \"{NOT_FOUND}\" "
    "Do not use outside knowledge and do not guess."
)


def build_user_prompt(question: str, passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        where = p["source"] + (f", page {p['page']}" if p.get("page") else "")
        blocks.append(f"[{i}] ({where})\n{p['text']}")
    context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def make_client(api_key: str | None = None):
    from groq import Groq

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key)


def answer(
    question: str,
    passages: list[dict],
    client=None,
    model: str | None = None,
) -> str:
    """Return an answer grounded in `passages`. `client` is anything exposing
    `.chat.completions.create(...)` (the Groq/OpenAI client interface)."""
    if not passages:
        return NOT_FOUND
    client = client or make_client()
    response = client.chat.completions.create(
        model=model or os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, passages)},
        ],
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()
