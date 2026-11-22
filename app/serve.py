"""Production entrypoint: mount Chainlit into FastAPI with /health, /readyz, /metrics.

Run with: uvicorn app.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Response

from app.metrics import prometheus_text
from app.observability import setup_tracing
from app.store.sqlite import connect

setup_tracing()

app = FastAPI(title="agentic-rag")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    try:
        c = connect()
        c.execute("SELECT 1").fetchone()
        c.close()
        return Response(content="ok", status_code=200, media_type="text/plain")
    except Exception as e:
        return Response(content=f"db-error: {e}", status_code=503, media_type="text/plain")


@app.get("/metrics")
async def metrics():
    return Response(content=prometheus_text(), media_type="text/plain")


# Mount Chainlit under / (root) so the UI stays at /
try:
    from chainlit.utils import mount_chainlit

    mount_chainlit(app=app, target="app/ui/app.py", path="/")
except Exception as e:
    import logging

    logging.getLogger("ragbuddy.serve").warning("mount_chainlit failed: %s", e)
