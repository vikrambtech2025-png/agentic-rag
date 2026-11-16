from app.config import settings
from app.ingest.chunker import chunk_document, count_tokens
from app.ingest.loaders import LoadedDoc


def test_chunk_count_and_budget():
    text = ("Hello world. " + "This is a sentence. " * 200)
    doc = LoadedDoc(pages=[(1, text)])
    chunks = chunk_document(doc, "src", "d1")
    assert len(chunks) > 1
    for c in chunks:
        assert c.tokens <= settings.chunk_tokens + 5  # small slack
        assert c.text.strip()


def test_heading_survives():
    text = "# Title\nSome body text here.\n## Sub\nMore body."
    doc = LoadedDoc(pages=[(1, text)])
    chunks = chunk_document(doc, "src", "d1")
    headings = {c.heading for c in chunks}
    assert any("Title" in h for h in headings)


def test_empty_doc():
    doc = LoadedDoc(pages=[(1, "")])
    assert chunk_document(doc, "src", "d1") == []
