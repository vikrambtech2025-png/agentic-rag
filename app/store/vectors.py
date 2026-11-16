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
            pa.field("page", pa.int32()),
            pa.field("heading", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
        ]
    )


def _db():
    global _DB
    if _DB is None:
        with _LOCK:
            if _DB is None:
                _DB = lancedb.connect(str(settings.lancedb_path))
    return _DB


def table():
    global _TABLE
    if _TABLE is None:
        with _LOCK:
            if _TABLE is None:
                db = _db()
                if TABLE_NAME not in db.list_tables().tables:
                    db.create_table(TABLE_NAME, schema=_schema(settings.embed_dim))
                _TABLE = db.open_table(TABLE_NAME)
    return _TABLE


def upsert(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    t = table()
    t.add(rows)


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
