"""SQLite FTS5 keyword search."""
from __future__ import annotations

import sqlite3

from app.store.sqlite import connect


def fts_search(q: str, k: int = 20, user_id: str | None = None, filters: dict | None = None) -> list[dict]:
    conn = connect()
    try:
        where = ""
        params: list = [q]
        if user_id:
            where += " AND d.user_id = ?"
            params.append(user_id)
        f = filters or {}
        if f.get("source"):
            vals = f["source"]
            if isinstance(vals, str):
                vals = [vals]
            where += " AND d.source IN (" + ",".join("?" for _ in vals) + ")"
            params.extend(vals)
        if f.get("doc_id"):
            vals = f["doc_id"]
            if isinstance(vals, str):
                vals = [vals]
            where += " AND c.doc_id IN (" + ",".join("?" for _ in vals) + ")"
            params.extend(vals)
        if f.get("page_min") is not None:
            where += " AND c.page >= ?"
            params.append(int(f["page_min"]))
        if f.get("page_max") is not None:
            where += " AND c.page <= ?"
            params.append(int(f["page_max"]))
        params.append(k)
        rows = conn.execute(
            f"""
            SELECT c.id AS id, c.doc_id AS doc_id, c.page AS page,
                   c.heading AS heading, c.text AS text,
                   bm25(chunks_fts) AS score,
                   d.source AS source, d.user_id AS user_id
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            JOIN documents d ON d.id = c.doc_id
            WHERE chunks_fts MATCH ?{where}
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def upsert_fts(chunk_id: str, text: str, heading: str = "") -> None:
    # No-op when FTS sync triggers are installed; kept for backwards compat.
    return None


def delete_by_doc(doc_id: str) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    print("ok")
