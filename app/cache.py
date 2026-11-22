"""Optional Redis client + distributed rate limit."""
from __future__ import annotations

import os
import time
from typing import Any


def redis_client() -> Any | None:
    url = os.environ.get("REDIS_URL", "")
    if not url:
        return None
    try:
        import redis

        return redis.from_url(url, decode_responses=True)
    except Exception:
        return None


def rate_limit_ok(key: str, limit: int = 20, window: float = 60.0) -> bool:
    r = redis_client()
    if r is None:
        # in-memory fallback handled by caller; return None to signal fallback
        return None  # type: ignore
    now = time.time()
    bucket = f"rl:{key}"
    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(bucket, 0, now - window)
        pipe.zadd(bucket, {str(now): now})
        pipe.zcard(bucket)
        pipe.expire(bucket, int(window) + 1)
        _, _, count, _ = pipe.execute()
        return int(count) <= limit
    except Exception:
        return None  # type: ignore
