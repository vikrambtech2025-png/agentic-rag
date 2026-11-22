"""DB-backed ingestion jobs for async/non-blocking ingest."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime

from app.ingest.pipeline import (
    ingest_path,
    ingest_url,
    ingest_youtube,
    ingest_text,
)
from app.store.sqlite import connect, migrate


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def create_job(user_id: str | None, source: str, kind: str) -> str:
    jid = str(uuid.uuid4())
    conn = migrate()
    try:
        conn.execute(
            "INSERT INTO ingest_jobs(id,user_id,source,kind,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            (jid, user_id, source, kind, "queued", _now(), _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return jid


def update_job(jid: str, **fields) -> None:
    conn = migrate()
    try:
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [jid]
        conn.execute(f"UPDATE ingest_jobs SET {sets}, updated_at=? WHERE id=?", vals[:-1] + [_now(), jid])
        conn.commit()
    finally:
        conn.close()


def get_job(jid: str) -> dict | None:
    conn = migrate()
    try:
        row = conn.execute("SELECT * FROM ingest_jobs WHERE id=?", (jid,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def process_job(jid: str) -> None:
    job = get_job(jid)
    if not job:
        return
    update_job(jid, status="running")
    try:
        source = job["source"]
        kind = job["kind"]
        user_id = job["user_id"]
        if kind == "file":
            doc_id = ingest_path(source, user_id=user_id)
        elif kind == "url":
            if "youtube.com" in source or "youtu.be" in source:
                doc_id = ingest_youtube(source, user_id=user_id)
            else:
                doc_id = ingest_url(source, user_id=user_id)
        elif kind == "text":
            doc_id = ingest_text(source, user_id=user_id)
        else:
            raise ValueError(f"unknown kind {kind}")
        update_job(jid, status="done", doc_id=doc_id)
    except Exception as e:
        update_job(jid, status="failed", error=str(e)[:500])


def run_in_background(jid: str) -> None:
    t = threading.Thread(target=process_job, args=(jid,), daemon=True)
    t.start()


def worker_loop(poll: float = 2.0) -> None:
    import time

    while True:
        conn = migrate()
        try:
            rows = conn.execute("SELECT id FROM ingest_jobs WHERE status='queued'").fetchall()
            ids = [r["id"] for r in rows]
        finally:
            conn.close()
        for jid in ids:
            process_job(jid)
        time.sleep(poll)


if __name__ == "__main__":
    worker_loop()
