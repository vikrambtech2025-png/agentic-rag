"""Hybrid fusion: reciprocal rank fusion of vector + keyword results + rerank."""
from __future__ import annotations

from app.store.vectors import search as vsearch
from app.store.embeddings import embed_one
from app.store.keyword import fts_search
from app.store.rerank import rerank


def _esc(v: str) -> str:
    return v.replace("'", "''")


def _build_where(user_id: str | None, filters: dict | None) -> str | None:
    parts = []
    if user_id:
        parts.append(f"user_id = '{_esc(user_id)}'")
    f = filters or {}
    if f.get("source"):
        vals = f["source"]
        if isinstance(vals, str):
            vals = [vals]
        parts.append("source IN (" + ",".join("'" + _esc(v) + "'" for v in vals) + ")")
    if f.get("doc_id"):
        vals = f["doc_id"]
        if isinstance(vals, str):
            vals = [vals]
        parts.append("doc_id IN (" + ",".join("'" + _esc(v) + "'" for v in vals) + ")")
    if f.get("page_min") is not None:
        parts.append(f"page >= {int(f['page_min'])}")
    if f.get("page_max") is not None:
        parts.append(f"page <= {int(f['page_max'])}")
    return " AND ".join(parts) if parts else None


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


def search(q: str, k: int = 8, where: str | None = None, user_id: str | None = None, filters: dict | None = None, rerank_top: int = 30) -> list[dict]:
    vec = embed_one(q)
    w = where
    fb = _build_where(user_id, filters)
    if fb:
        w = f"{w} AND {fb}" if w else fb
    vres = vsearch(vec, k=rerank_top, where=w)
    fres = fts_search(q, k=rerank_top, user_id=user_id, filters=filters)
    fused = _rrf([vres, fres])
    cand = fused[: max(rerank_top, k)]
    # MMR-ish dedup by near-duplicate text
    seen_text = set()
    deduped = []
    for c in cand:
        key = (c.get("text") or "")[:80].strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        deduped.append(c)
    ranked = rerank(q, deduped, top_k=k)
    return ranked


if __name__ == "__main__":
    print("ok")
