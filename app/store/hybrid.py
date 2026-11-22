"""Hybrid fusion: reciprocal rank fusion of vector + keyword results."""
from __future__ import annotations

from app.store.vectors import search as vsearch
from app.store.embeddings import embed_one
from app.store.keyword import fts_search


def _esc(v: str) -> str:
    return v.replace("'", "''")


def _rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores, key=lambda x: scores[x], reverse=True)
    out = []
    for cid in ordered:
        d = dict(by_id[cid])
        d["score"] = scores[cid]
        out.append(d)
    return out


def search(q: str, k: int = 8, where: str | None = None, user_id: str | None = None) -> list[dict]:
    vec = embed_one(q)
    vwhere = where
    if user_id:
        clause = f"user_id = '{_esc(user_id)}'"
        vwhere = f"{vwhere} AND {clause}" if vwhere else clause
    vres = vsearch(vec, k=k * 2, where=vwhere)
    fres = fts_search(q, k=k * 2, user_id=user_id)
    fused = _rrf([vres, fres])
    return fused[:k]


if __name__ == "__main__":
    print("ok")
