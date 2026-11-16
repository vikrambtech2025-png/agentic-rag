"""Web search via Tavily with SQLite cache."""
from __future__ import annotations

import json
import time
from typing import Any

from tavily import TavilyClient

from app.config import settings
from app.store.sqlite import connect, migrate


def _client():
    return TavilyClient(api_key=settings.tavily_api_key)


def _cache_get(q: str) -> list[dict[str, Any]] | None:
    migrate()
    conn = connect()
    try:
        row = conn.execute("SELECT payload, fetched_at FROM web_cache WHERE query=?", (q,)).fetchone()
        if not row:
            return None
        # cache for 1 hour
        if time.time() - float(row["fetched_at"]) > 3600:
            return None
        return json.loads(row["payload"])
    except Exception:
        return None
    finally:
        conn.close()


def _cache_set(q: str, payload: list[dict[str, Any]]) -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO web_cache(query,payload,fetched_at) VALUES (?,?,?)",
            (q, json.dumps(payload), str(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    cached = _cache_get(query)
    if cached is not None:
        return cached
    try:
        resp = _client().search(query=query, max_results=max_results, search_depth="basic")
    except Exception:
        return []
    results = []
    for r in resp.get("results", []):
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:600],
            }
        )
    _cache_set(query, results)
    return results


if __name__ == "__main__":
    print(web_search("latest python release"))

