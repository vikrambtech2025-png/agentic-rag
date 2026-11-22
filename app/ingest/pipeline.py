"""Ingestion orchestration: load -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
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
from app.store.vectors import upsert as vupsert, delete_by_doc_id


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


def _ensure_user(conn, user_id: str | None) -> None:
    if not user_id or user_id == "anonymous":
        user_id = user_id or "anonymous"
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users(id, username, created_at) VALUES (?,?,datetime('now'))",
            (user_id, user_id),
        )
        conn.commit()
    except Exception:
        pass


def _source_key(source: str, user_id: str | None) -> str:
    return f"{user_id or ''}::{source}"


def _already_ingested(conn, sha: str, user_id: str | None) -> str | None:
    row = conn.execute(
        "SELECT id, version FROM documents WHERE sha256=? AND COALESCE(user_id,'')=COALESCE(?,'')",
        (sha, user_id),
    ).fetchone()
    return row["id"] if row else None


def _supersede(conn, old_doc_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (old_doc_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (old_doc_id,))
    delete_by_doc_id(old_doc_id)


def _persist(doc_id, user_id, source, kind, title, pages, sha, version, chunks):
    conn = migrate()
    try:
        conn.execute(
            """INSERT INTO documents(id,user_id,source,source_key,kind,title,pages,sha256,version,status,embed_model,added_at,updated_at,meta)
               VALUES (?,?,?,?,?,?,?,?,?,'ready',?,datetime('now'),datetime('now'),?)""",
            (doc_id, user_id, source, _source_key(source, user_id), kind, title, pages, sha, version, settings.embed_model, None),
        )
        for i, c in enumerate(chunks):
            c.ordinal = i
            conn.execute(
                "INSERT INTO chunks(id,doc_id,ordinal,page,heading,text,tokens,embedding_model) VALUES (?,?,?,?,?,?,?,?)",
                (c.id, doc_id, i, c.page, c.heading, c.text, c.tokens, settings.embed_model),
            )
        conn.commit()
    finally:
        conn.close()


def ingest_document(doc: LoadedDoc, source: str, user_id: str | None = None) -> str:
    sha = sha256_text(doc.full_text)
    conn = migrate()
    try:
        existing = _already_ingested(conn, sha, user_id)
        if existing:
            conn.close()
            return existing
        # supersede any prior version of the same source
        row = conn.execute(
            "SELECT id FROM documents WHERE source_key=?",
            (_source_key(source, user_id),),
        ).fetchone()
        prior_id = row["id"] if row else None
        conn.close()

        doc_id = str(uuid.uuid4())
        chunks = chunk_document(doc, source, doc_id)
        if not chunks:
            raise ValueError("No chunks produced from document")

        # ensure user row exists (FK)
        _cc = migrate()
        try:
            _ensure_user(_cc, user_id)
        finally:
            _cc.close()

        # pending row first (crash-safe)
        c2 = migrate()
        try:
            c2.execute(
                """INSERT INTO documents(id,user_id,source,source_key,kind,title,pages,sha256,version,status,embed_model,added_at,updated_at,meta)
                   VALUES (?,?,?,?,?,?,?,?,?,'pending',?,datetime('now'),datetime('now'),?)""",
                (doc_id, user_id, source, _source_key(source, user_id), doc.kind, doc.title or source, len(doc.pages), sha, 1, settings.embed_model, None),
            )
            c2.commit()
        finally:
            c2.close()

        texts = [ch.text for ch in chunks]
        vecs = embed_batch(texts)
        rows = [
            {
                "id": ch.id,
                "vector": v,
                "doc_id": doc_id,
                "user_id": user_id or "",
                "page": ch.page,
                "heading": ch.heading,
                "text": ch.text,
                "source": source,
                "embed_model": settings.embed_model,
            }
            for ch, v in zip(chunks, vecs)
        ]
        vupsert(rows)

        # remove prior version now that new one is ready
        if prior_id and prior_id != doc_id:
            _supersede_conn(prior_id)

        _persist_ready(doc_id, user_id, source, doc.kind, doc.title or source, len(doc.pages), sha, chunks)
        return doc_id
    except Exception:
        # mark failed
        try:
            cc = migrate()
            cc.execute("UPDATE documents SET status='failed', updated_at=datetime('now') WHERE id=?", (doc_id,))
            cc.commit()
            cc.close()
        except Exception:
            pass
        raise


def _persist_ready(doc_id, user_id, source, kind, title, pages, sha, chunks):
    conn = migrate()
    try:
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        for i, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO chunks(id,doc_id,ordinal,page,heading,text,tokens,embedding_model) VALUES (?,?,?,?,?,?,?,?)",
                (c.id, doc_id, i, c.page, c.heading, c.text, c.tokens, settings.embed_model),
            )
        conn.execute(
            """UPDATE documents SET status='ready', title=?, pages=?, sha256=?, updated_at=datetime('now') WHERE id=?""",
            (title, pages, sha, doc_id),
        )
        conn.commit()
    finally:
        conn.close()


def _supersede_conn(old_doc_id: str) -> None:
    conn = migrate()
    try:
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (old_doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (old_doc_id,))
        conn.commit()
    finally:
        conn.close()
    delete_by_doc_id(old_doc_id)


def ingest_path(path: str | Path, user_id: str | None = None) -> str:
    p = Path(path)
    doc = _load_path(p)
    return ingest_document(doc, str(p), user_id=user_id)


def ingest_url(url: str, user_id: str | None = None) -> str:
    doc = load_url(url)
    return ingest_document(doc, url, user_id=user_id)


def ingest_youtube(url: str, user_id: str | None = None) -> str:
    doc = load_youtube(url)
    return ingest_document(doc, url, user_id=user_id)


def ingest_text(text: str, title: str = "pasted", user_id: str | None = None) -> str:
    doc = LoadedDoc(pages=[(1, text)], kind="text", title=title)
    return ingest_document(doc, f"text:{title}", user_id=user_id)


def delete_document(doc_id: str, user_id: str | None = None) -> bool:
    conn = migrate()
    try:
        if user_id is None:
            row = conn.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)
            ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
    delete_by_doc_id(doc_id)
    return True


def list_documents(user_id: str | None = None) -> list[dict]:
    conn = migrate()
    try:
        if user_id:
            rows = conn.execute(
                "SELECT id, source, kind, title, pages, status, version, added_at, updated_at FROM documents WHERE user_id=? ORDER BY added_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, source, kind, title, pages, status, version, added_at, updated_at FROM documents ORDER BY added_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def reconcile() -> int:
    """Remove vector rows whose doc no longer exists (crash recovery)."""
    from app.store.vectors import table as vtable

    conn = migrate()
    try:
        ids = {r["id"] for r in conn.execute("SELECT id FROM documents").fetchall()}
    finally:
        conn.close()
    t = vtable()
    try:
        rows = t.to_pandas()["doc_id"].tolist() if hasattr(t, "to_pandas") else []
    except Exception:
        rows = []
    cleaned = 0
    for did in set(rows):
        if did not in ids:
            delete_by_doc_id(did)
            cleaned += 1
    return cleaned


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m app.ingest.pipeline <path>")
        sys.exit(1)
    print(ingest_path(sys.argv[1]))
