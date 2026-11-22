"""LanceDB vector store."""
from __future__ import annotations

import threading
from typing import Any

import lancedb
import pyarrow as pa

from app.config import settings

_LOCK = threading.RLock()
_DB = None
_TABLE = None
TABLE_NAME = "chunks"


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("doc_id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("heading", pa.string()),
            pa.field("text", pa.string()),
            pa.field("parent_text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("embed_model", pa.string()),
        ]
    )


def _db():
    global _DB
    if _DB is None:
        with _LOCK:
            if _DB is None:
                _DB = lancedb.connect(str(settings.lancedb_path))
    return _DB


def _schema_ok(t) -> bool:
    try:
        names = {f.name for f in t.schema}
    except Exception:
        return False
    return {"id", "vector", "doc_id", "user_id", "text", "parent_text", "source", "embed_model"}.issubset(names)


def table():
    global _TABLE
    if _TABLE is None:
        with _LOCK:
            if _TABLE is None:
                db = _db()
                if TABLE_NAME not in db.list_tables().tables:
                    db.create_table(TABLE_NAME, schema=_schema(settings.embed_dim))
                else:
                    t = db.open_table(TABLE_NAME)
                    if not _schema_ok(t):
                        import logging

                        logging.getLogger("ragbuddy.vectors").warning(
                            "LanceDB schema outdated; recreating table (re-ingest required)"
                        )
                        db.drop_table(TABLE_NAME)
                        db.create_table(TABLE_NAME, schema=_schema(settings.embed_dim))
                _TABLE = db.open_table(TABLE_NAME)
    return _TABLE


def upsert(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # normalize rows
    norm = []
    for r in rows:
        rr = dict(r)
        rr.setdefault("user_id", rr.get("user_id") or "")
        rr.setdefault("embed_model", rr.get("embed_model") or settings.embed_model)
        norm.append(rr)
    t = table()
    try:
        t.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(norm)
    except Exception:
        # fallback for older lancedb
        t.add(norm)


def delete_by_doc_id(doc_id: str) -> int:
    t = table()
    try:
        t.delete(f"doc_id = '{doc_id}'")
        return 1
    except Exception:
        return 0


def delete_by_user(user_id: str) -> int:
    t = table()
    try:
        t.delete(f"user_id = '{user_id}'")
        return 1
    except Exception:
        return 0


def search(vec: list[float], k: int = 8, where: str | None = None) -> list[dict[str, Any]]:
    t = table()
    q = t.search(vec).limit(k)
    if where:
        q = q.where(where)
    return q.to_list()


def count() -> int:
    return table().count_rows()


def reset() -> None:
    global _TABLE, _DB
    with _LOCK:
        db = _db()
        if TABLE_NAME in db.list_tables().tables:
            db.drop_table(TABLE_NAME)
        _TABLE = None


def main() -> None:
    reset()
    from app.store.embeddings import embed_batch

    rows = [
        {"id": "a", "vector": embed_batch(["hello world"])[0], "doc_id": "d1", "page": 1, "heading": "h", "text": "hello world", "source": "t"},
        {"id": "b", "vector": embed_batch(["goodbye moon"])[0], "doc_id": "d1", "page": 2, "heading": "h", "text": "goodbye moon", "source": "t"},
    ]
    upsert(rows)
    res = search(embed_batch(["hi there"])[0], k=2)
    print("count", count())
    print("top", res[0]["id"], round(res[0]["_distance"], 4))
    reset()


if __name__ == "__main__":
    main()
