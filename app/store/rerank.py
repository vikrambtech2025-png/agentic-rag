"""Optional cross-encoder reranker with graceful fallback."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("ragbuddy.rerank")

_MODEL = None
_AVAILABLE = None


def _load() -> bool:
    global _MODEL, _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder("BAAI/bge-reranker-base")
        _AVAILABLE = True
        log.info("cross-encoder reranker loaded")
    except Exception as e:
        _AVAILABLE = False
        log.info("reranker unavailable (%s); using heuristic fallback", e)
    return _AVAILABLE


def _heuristic(query: str, text: str) -> float:
    qtoks = {t for t in query.lower().split() if len(t) > 2}
    ttoks = {t for t in text.lower().split() if len(t) > 2}
    if not qtoks:
        return 0.0
    return len(qtoks & ttoks) / len(qtoks)


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if _load():
        try:
            pairs = [(query, (c.get("text") or "")[:1000]) for c in candidates]
            scores = _MODEL.predict(pairs)
            ranked = sorted(
                (dict(c, _rerank=float(s)) for c, s in zip(candidates, scores)),
                key=lambda x: x["_rerank"],
                reverse=True,
            )
            return ranked[:top_k]
        except Exception as e:
            log.warning("rerank failed: %s", e)
    scored = []
    for c in candidates:
        base = c.get("score", 0.0) or 0.0
        h = _heuristic(query, (c.get("heading", "") + " " + (c.get("text") or "")))
        scored.append((base + 0.5 * h, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


if __name__ == "__main__":
    print("reranker available:", _load())
