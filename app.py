"""DocuMind: chat with your documents (RAG) - Streamlit UI."""
from __future__ import annotations

import os

import streamlit as st

from rag.embeddings import DEFAULT_MODEL, get_embedder
from rag.generator import answer, make_client
from rag.ingest import SUPPORTED_EXTENSIONS, build_chunks
from rag.retriever import Retriever

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return get_embedder(DEFAULT_MODEL)


def init_state() -> None:
    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(load_embedder())
        st.session_state.messages = []
        st.session_state.indexed = set()


def index_files(files, chunk_size: int, overlap: int) -> None:
    retriever: Retriever = st.session_state.retriever
    added = 0
    for f in files:
        if f.name in st.session_state.indexed:
            continue
        chunks = build_chunks(f.name, f.getvalue(), chunk_size, overlap)
        if not chunks:
            st.warning(f"No extractable text in {f.name} (scanned PDF?).")
            continue
        retriever.add(chunks)
        st.session_state.indexed.add(f.name)
        added += len(chunks)
    if added:
        st.success(f"Indexed {added} chunks.")


def show_sources(sources: list[dict]) -> None:
    for i, p in enumerate(sources, start=1):
        page = f", page {p['page']}" if p["page"] else ""
        with st.expander(f"[{i}] {p['source']}{page}  (similarity {p['score']:.2f})"):
            st.write(p["text"])


def main() -> None:
    st.title("📄 DocuMind")
    st.caption("Ask questions about your documents. Answers are grounded in the uploaded text, with sources.")

    with st.sidebar:
        st.header("1. Upload documents")
        files = st.file_uploader(
            "PDF, TXT or MD",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 100)
        overlap = st.slider("Chunk overlap", 0, 300, 150, 50)
        top_k = st.slider("Passages retrieved (top-k)", 1, 8, 4)

        st.header("2. LLM access")
        env_key = os.environ.get("GROQ_API_KEY")
        api_key = env_key or st.text_input("Groq API key", type="password")

    init_state()

    with st.sidebar:
        if st.button("Index documents", type="primary", disabled=not files):
            with st.spinner("Reading and embedding..."):
                index_files(files, chunk_size, overlap)
        if st.button("Clear everything"):
            st.session_state.retriever.reset()
            st.session_state.messages = []
            st.session_state.indexed = set()
            st.rerun()
        st.metric("Chunks indexed", st.session_state.retriever.count())

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources", []))

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.retriever.count() == 0:
            reply, sources = "Please upload and index at least one document first.", []
        elif not api_key:
            reply, sources = "Please provide a Groq API key in the sidebar.", []
        else:
            try:
                sources = st.session_state.retriever.search(question, k=top_k)
                with st.spinner("Thinking..."):
                    reply = answer(question, sources, client=make_client(api_key))
            except Exception as exc:  # surface API/network errors to the user
                reply, sources = f"Something went wrong: {exc}", []
        st.markdown(reply)
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})


main()
