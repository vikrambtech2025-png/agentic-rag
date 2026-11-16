"""Embedding helpers: OpenAI-compatible (remote) or fastembed (local/offline)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.config import settings


def _is_remote() -> bool:
    m = settings.embed_model.lower()
    return ("text-embedding" in m) or m.startswith("openai/")


_REMOTE_CLIENT = None


def _remote_client():
    global _REMOTE_CLIENT
    if _REMOTE_CLIENT is None:
        from openai import OpenAI

        _REMOTE_CLIENT = OpenAI(api_key=settings.effective_api_key, base_url=settings.effective_base_url)
    return _REMOTE_CLIENT


def _remote_embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    batch = 100
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        for attempt in range(3):
            try:
                resp = _remote_client().embeddings.create(model=settings.embed_model, input=chunk)
                out.extend([list(d.embedding) for d in resp.data])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return out


_LOCAL_MODEL = None


def _local_embed(texts: list[str]) -> list[list[float]]:
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from fastembed import TextEmbedding

        _LOCAL_MODEL = TextEmbedding(model_name=settings.embed_model)
    vecs = list(_LOCAL_MODEL.embed(texts))
    return [ [float(x) for x in v] for v in vecs ]


def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if _is_remote():
        return _remote_embed(texts)
    return _local_embed(texts)


def embed_one(text: str) -> list[float]:
    return embed_batch([text])[0]


if __name__ == "__main__":
    v = embed_one("hello")
    print(len(v), v[:5])
