"""Document loading and chunking. Deliberately free of heavy ML imports."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int  # 1-based page for PDFs, 0 for plain text files
    chunk_id: str


def load_document(name: str, data: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Plain text files use page 0."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(0, data.decode("utf-8", errors="ignore"))]
    raise ValueError(
        f"Unsupported file type '{suffix}' for {name}. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text:
        return []

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > chunk_size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(
    name: str, data: bytes, chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    """Load a file and turn it into Chunk objects with source/page metadata."""
    chunks: list[Chunk] = []
    for page, text in load_document(name, data):
        for i, piece in enumerate(split_text(text, chunk_size, overlap)):
            chunks.append(Chunk(piece, name, page, f"{name}::{page}::{i}"))
    return chunks
