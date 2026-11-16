"""SQLite FTS5 keyword search."""
from __future__ import annotations

import sqlite3

from app.store.sqlite import connect


def fts_search(q: str, k: int = 20) -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT c.id AS id, c.doc_id AS doc_id, c.page AS page,
                   c.heading AS heading, c.text AS text,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (q, k),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def upsert_fts(chunk_id: str, text: str, heading: str = "") -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO chunks_fts(text, heading, chunk_id) VALUES (?,?,?)",
            (text, heading, chunk_id),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    print("ok")
