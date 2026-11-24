# 📄 DocuMind: RAG Document Q&A Assistant

Upload PDFs, text or Markdown files and ask questions in plain English. DocuMind retrieves the most relevant passages
from your documents and has an LLM answer **only from that context**, showing the source passages for every answer.

![CI](https://github.com/<your-username>/documind/actions/workflows/ci.yml/badge.svg)

**Live demo:** _add your Hugging Face Space link here_

## How it works

```
 PDF / TXT / MD ──► split into overlapping chunks ──► embed (all-MiniLM-L6-v2) ──► ChromaDB
                                                                                      │
 question ──► embed ──► top-k similar chunks ◄────────────────────────────────────────┘
                              │
                              ▼
              grounded prompt ("answer ONLY from this context") ──► LLM (Groq) ──► answer + sources
```

| Layer | Choice |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally on CPU) |
| Vector store | ChromaDB (in-memory, one private collection per browser session) |
| LLM | Groq API (`llama-3.1-8b-instant` by default, free tier) |
| Packaging / CI | Docker, GitHub Actions (ruff + pytest + image build) |

## Project structure

```
documind/
├── app.py                  # Streamlit UI
├── rag/
│   ├── ingest.py           # load PDF/TXT/MD, chunking with overlap
│   ├── embeddings.py       # sentence-transformer embedder
│   ├── retriever.py        # ChromaDB wrapper (add / search / reset)
│   └── generator.py        # grounded prompt + LLM call
├── eval/
│   ├── eval.py             # hit@k / MRR, sweep chunk size and top-k
│   └── questions.json      # labeled question set for the sample doc
├── data/sample_docs/       # sample document to try the app and eval
├── tests/                  # unit tests (no model or API key needed)
├── Dockerfile
├── requirements.txt
├── requirements-ci.txt
└── .github/workflows/ci.yml
```

## Run locally

```bash
git clone https://github.com/<your-username>/documind.git && cd documind
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here                       # free key: https://console.groq.com
streamlit run app.py
```

Open http://localhost:8501, upload `data/sample_docs/platform_handbook.md`, click **Index documents**, and ask e.g.
*"What share of production traffic does a canary release receive first?"*

If you skip the environment variable, the app asks for the key in the sidebar.

## Run with Docker

```bash
docker build -t documind .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key_here documind
```

## Evaluate retrieval

`eval/eval.py` measures how often the correct passage is among the top-k retrieved chunks (**hit@k**) and the mean
reciprocal rank (**MRR**), and can sweep chunk size and top-k to pick settings.

```bash
python -m eval.eval --sweep            # grid over chunk sizes x top-k
python -m eval.eval --answers          # also check LLM answers (needs GROQ_API_KEY)
python -m eval.eval --docs my_docs/ --questions my_questions.json   # your own documents
```

Use your own documents and 15-20 questions for numbers worth quoting, and record your results here:

| Chunk size | Top-k | hit@k | MRR |
|---|---|---|---|
| _fill in_ | | | |

## Tests and CI

```bash
pip install -r requirements-ci.txt
ruff check . && pytest -q
```

GitHub Actions runs lint, unit tests and a Docker build on every push and pull request.

## Deploy a live demo on Hugging Face Spaces

1. Create a **Space** at https://huggingface.co/new-space, choose the **Docker** SDK (blank template).
2. Add this block at the very top of the Space's `README.md`:
   ```yaml
   ---
   title: DocuMind
   emoji: 📄
   sdk: docker
   app_port: 8501
   ---
   ```
3. In the Space **Settings**:
   - Secret: `GROQ_API_KEY` = your key
   - Variables: `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false` and `STREAMLIT_SERVER_ENABLE_CORS=false`
     (needed for file uploads inside the Spaces iframe)
4. Push the repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/documind
   git push space main
   ```

## Limitations

- Scanned PDFs without a text layer are not supported (no OCR).
- The index lives in memory and is lost when the session or server restarts.
- Answer quality depends on retrieval quality; tune chunk size and top-k with the eval script.
