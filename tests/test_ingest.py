import io

import pytest
from pypdf import PdfWriter

from rag.ingest import build_chunks, split_text


def test_split_respects_chunk_size_and_overlap():
    text = ("Sentence number one is here. " * 100).strip()
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # overlap: end of one chunk reappears at the start of the next
    assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]


def test_split_empty_and_short():
    assert split_text("   \n  ") == []
    assert split_text("short text", 800, 100) == ["short text"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_split_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        split_text("hello", size, overlap)


def test_split_always_terminates_with_large_overlap():
    chunks = split_text("a" * 1000, chunk_size=100, overlap=99)
    assert chunks and all(len(c) <= 100 for c in chunks)


def test_build_chunks_text_file_metadata():
    chunks = build_chunks("notes.txt", b"Hello world. " * 200, 300, 50)
    assert len(chunks) > 1
    assert {c.source for c in chunks} == {"notes.txt"}
    assert {c.page for c in chunks} == {0}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_build_chunks_unsupported_type():
    with pytest.raises(ValueError):
        build_chunks("image.png", b"data")


def test_blank_pdf_yields_no_chunks():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    assert build_chunks("blank.pdf", buf.getvalue()) == []
