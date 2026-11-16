"""Ingestion orchestration: load -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from app.config import settings
from app.ingest.chunker import chunk_document
from app.ingest.loaders import (
    LoadedDoc,
    load_markdown,
    load_pdf,
    load_text,
    load_url,
    load_youtube,
    sha256_text,
)
from app.models import Chunk
from app.store.embeddings import embed_batch
from app.store.sqlite import connect, migrate
from app.store.vectors import upsert as vupsert
from app.store.keyword import upsert_fts


def _detect_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".md", ".markdown"}:
        return "markdown"
    return "text"


def _load_path(path: Path) -> LoadedDoc:
    kind = _detect_kind(path)
    if kind == "pdf":
        return load_pdf(path)
    if kind == "markdown":
        return load_markdown(path)
    return load_text(path)


def _persist(doc_id: str, source: str, kind: str, title: str, pages: int, sha: str, chunks: list[Chunk]) -> None:
    conn = migrate()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO documents(id,source,kind,title,pages,sha256,added_at,meta) VALUES (?,?,?,?,?,?,datetime('now'),?)",
            (doc_id, source, kind, title, pages, sha, None),
        )
        for c in chunks:
            conn.execute(
                "INSERT INTO chunks(id,doc_id,ordinal,page,heading,text,tokens) VALUES (?,?,?,?,?,?,?)",
                (c.id, doc_id, 0, c.page, c.heading, c.text, c.tokens),
            )
            conn.execute(
                "INSERT INTO chunks_fts(text, heading, chunk_id) VALUES (?,?,?)",
                (c.text, c.heading, c.id),
            )
        conn.commit()
    finally:
        conn.close()


def _already_ingested(sha: str) -> str | None:
    conn = migrate()
    try:
        row = conn.execute("SELECT id FROM documents WHERE sha256=?", (sha,)).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def ingest_document(doc: LoadedDoc, source: str) -> str:
    sha = sha256_text(doc.full_text)
    existing = _already_ingested(sha)
    if existing:
        return existing
    doc_id = str(__import__("uuid").uuid4())
    chunks = chunk_document(doc, source, doc_id)
    if not chunks:
        raise ValueError("No chunks produced from document")
    texts = [c.text for c in chunks]
    vecs = embed_batch(texts)
    rows = [
        {
            "id": c.id,
            "vector": v,
            "doc_id": doc_id,
            "page": c.page,
            "heading": c.heading,
            "text": c.text,
            "source": source,
        }
        for c, v in zip(chunks, vecs)
    ]
    vupsert(rows)
    _persist(doc_id, source, doc.kind, doc.title or source, len(doc.pages), sha, chunks)
    return doc_id


def ingest_path(path: str | Path) -> str:
    p = Path(path)
    doc = _load_path(p)
    return ingest_document(doc, str(p))


def ingest_url(url: str) -> str:
    doc = load_url(url)
    return ingest_document(doc, url)


def ingest_youtube(url: str) -> str:
    doc = load_youtube(url)
    return ingest_document(doc, url)


def ingest_text(text: str, title: str = "pasted") -> str:
    doc = LoadedDoc(pages=[(1, text)], kind="text", title=title)
    return ingest_document(doc, f"text:{title}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m app.ingest.pipeline <path>")
        raise SystemExit(1)
    print(ingest_path(sys.argv[1]))
