"""Validation + retrieval smoke tests (no LLM required)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import fitz

from app.config import settings
from app.ingest.loaders import _check_file, _safe_url, ALLOWED_EXTS
from app.ingest.pipeline import ingest_path
from app.store.hybrid import search


def _make_pdf(path: Path, text: str) -> None:
    d = fitz.open()
    p = d.new_page()
    p.insert_text((72, 72), text)
    d.save(str(path))


def test_check_file_rejects_bad_ext(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"hi")
    try:
        _check_file(p)
        assert False, "should raise"
    except ValueError:
        pass


def test_check_file_rejects_too_large(tmp_path, monkeypatch):
    p = tmp_path / "big.pdf"
    p.write_bytes(b"x" * 100)
    monkeypatch.setattr("app.ingest.loaders.MAX_FILE_BYTES", 10)
    try:
        _check_file(p)
        assert False, "should raise"
    except ValueError:
        pass


def test_safe_url_blocks_private():
    for bad in ["http://127.0.0.1/x", "http://localhost/x", "http://10.0.0.1/x", "ftp://example.com/x"]:
        try:
            _safe_url(bad)
            assert False, f"should block {bad}"
        except ValueError:
            pass


def test_safe_url_allows_public():
    _safe_url("https://example.com/page")


def test_ingest_and_retrieve(tmp_path):
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, "The quick brown fox jumps over the lazy dog. " * 5)
    doc_id = ingest_path(str(pdf))
    assert doc_id
    res = search("quick brown fox", k=3)
    assert len(res) > 0
    assert any("fox" in r["text"].lower() for r in res)
