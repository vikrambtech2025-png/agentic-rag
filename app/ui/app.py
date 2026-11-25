"""Chainlit UI for the agentic RAG buddy."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import os
import shutil
from pathlib import Path

import chainlit as cl

from app.config import settings
from app.graph.build import get_app
from app.ingest.pipeline import ingest_path, ingest_url, ingest_youtube, ingest_text
from app.models import Chunk as _  # noqa

APP = get_app()


@cl.on_app_startup
async def _startup() -> None:
    from app.observability import setup_tracing

    setup_tracing()


def _thread_id() -> str:
    return cl.user_session.get("thread_id") or "default"


if settings.app_user and settings.app_password:
    @cl.password_auth_callback
    async def auth(username: str, password: str):
        import hmac

        if hmac.compare_digest(username, settings.app_user) and hmac.compare_digest(password, settings.app_password):
            return cl.User(identifier=username)
        return None


@cl.on_chat_start
async def on_start() -> None:
    cl.user_session.set("thread_id", os.urandom(8).hex())
    user = cl.user_session.get("user")
    user_id = getattr(user, "identifier", None) if user else None
    if not user_id:
        user_id = "anonymous"
    cl.user_session.set("user_id", user_id)
    await cl.Message(
        content=(
            "Hi! I'm your agentic RAG assistant. Upload a PDF/MD/TXT, paste a URL, "
            "or ask a question. I'll show my reasoning steps and cite sources."
        )
    ).send()


def _user_id() -> str:
    return cl.user_session.get("user_id") or "anonymous"


async def _ingest_file(file: cl.File) -> str:
    dest = settings.uploads_path / Path(file.name).name
    shutil.copyfile(file.path, dest)
    from app.jobs import create_job, run_in_background, get_job

    jid = create_job(_user_id(), str(dest), "file")
    run_in_background(jid)
    # poll
    while True:
        job = await asyncio.to_thread(get_job, jid)
        if job and job["status"] in ("done", "failed"):
            break
        await asyncio.sleep(1)
    if job["status"] == "failed":
        raise RuntimeError(job.get("error") or "ingest failed")
    return job["doc_id"]


_RATE: dict[str, list[float]] = {}


def _rate_ok(key: str, limit: int = 20, window: float = 60.0) -> bool:
    from app.cache import rate_limit_ok as _rl

    r = _rl(key, limit, window)
    if r is not None:
        return bool(r)
    import time

    now = time.time()
    bucket = [t for t in _RATE.get(key, []) if now - t < window]
    if len(bucket) >= limit:
        _RATE[key] = bucket
        return False
    bucket.append(now)
    _RATE[key] = bucket
    return True


@cl.on_message
async def on_message(message: cl.Message) -> None:
    from app.logging_config import REQUEST_ID

    REQUEST_ID.set(os.urandom(6).hex())
    uid = _user_id()
    if not _rate_ok(uid):
        await cl.Message(content="Rate limit exceeded. Please wait a minute.").send()
        return
    import time as _t

    _req_t0 = _t.perf_counter()
    from app.metrics import REQUESTS_TOTAL

    REQUESTS_TOTAL.inc("chat")
    from app.budget import CURRENT_USER, check_budget

    CURRENT_USER.set(uid)
    try:
        check_budget(uid)
    except Exception as e:
        await cl.Message(content=f"Budget exceeded: {e}").send()
        return    # Handle uploads
    for el in message.elements or []:
        if isinstance(el, cl.File):
            await cl.Message(content=f"Ingesting {el.name}...").send()
            try:
                did = await _ingest_file(el)
                await cl.Message(content=f"Ingested {el.name} (id {did[:8]}). Ask away!").send()
            except Exception as e:
                await cl.Message(content=f"Failed to ingest {el.name}: {e}").send()
        elif isinstance(el, cl.Image):
            pass

    text = (message.content or "").strip()
    from app.security import audit

    if text:
        audit("query", uid, text[:200])
    # Memory shortcut: "remember: ..."
    if text.lower().startswith("remember:"):
        fact = text.split(":", 1)[1].strip()
        from app.memory import remember

        remember(fact)
        await cl.Message(content=f"Remembered: {fact}").send()
        return
    # URL ingest shortcut
    if text.lower().startswith(("http://", "https://")) and " " not in text:
        await cl.Message(content=f"Ingesting URL {text}...").send()
        try:
            from app.jobs import create_job, run_in_background, get_job

            kind = "url"
            jid = create_job(_user_id(), text, kind)
            run_in_background(jid)
            while True:
                job = await asyncio.to_thread(get_job, jid)
                if job and job["status"] in ("done", "failed"):
                    break
                await asyncio.sleep(1)
            if job["status"] == "failed":
                raise RuntimeError(job.get("error") or "ingest failed")
            did = job["doc_id"]
            await cl.Message(content=f"Ingested (id {did[:8]}). Ask away!").send()
        except Exception as e:
            await cl.Message(content=f"Failed: {e}").send()
        return

    if not text:
        return
    # query cache
    from app.cache import get_cached, set_cached

    cached = get_cached(uid, text)
    if cached:
        msg = cl.Message(content=cached.get("answer", ""))
        await msg.send()
        return

    thread_id = _thread_id()
    state = {"question": text, "messages": [], "user_id": _user_id()}
    token_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def sink(delta: str) -> None:
        loop.call_soon_threadsafe(token_queue.put_nowait, delta)

    cfg = {"configurable": {"thread_id": thread_id, "stream_sink": sink}}

    msg = cl.Message(content="")
    await msg.send()

    step = cl.Step(name="agent", type="tool")
    await step.send()

    task = asyncio.create_task(asyncio.to_thread(APP.invoke, state, config=cfg))

    # consume tokens while running
    while not task.done():
        try:
            tok = await asyncio.wait_for(token_queue.get(), timeout=0.1)
            await msg.stream_token(tok)
        except asyncio.TimeoutError:
            continue
    # drain
    while not token_queue.empty():
        tok = token_queue.get_nowait()
        await msg.stream_token(tok)

    try:
        result = task.result()
    except Exception as e:
        from app.metrics import REQUEST_DURATION

        REQUEST_DURATION.observe("chat", _t.perf_counter() - _req_t0)
        await step.remove()
        await msg.stream_token(f"\n[error] {e}")
        await msg.update()
        return
    from app.metrics import REQUEST_DURATION

    REQUEST_DURATION.observe("chat", _t.perf_counter() - _req_t0)

    await step.remove()

    answer = result.get("answer", "") if isinstance(result, dict) else ""
    citations = result.get("citations", []) if isinstance(result, dict) else []
    # Ensure the answer is shown even when no tokens were streamed (e.g. chat route)
    if not answer:
        answer = "I couldn't generate an answer. Please try again."
    msg.content = answer
    if answer != "I couldn't generate an answer. Please try again.":
        try:
            set_cached(uid, text, answer, citations)
        except Exception:
            pass
    # if streaming already pushed tokens, answer may duplicate; clear content then set
    # We streamed tokens already, so don't overwrite; just attach sources.
    elements = []
    for c in citations:
        elements.append(
            cl.Text(name=f"[{c['n']}] {c.get('source','')} p.{c.get('page')}", content=c.get("snippet", ""), display="side")
        )
    msg.elements = elements
    unsupported = result.get("unsupported", []) if isinstance(result, dict) else []
    if unsupported:
        await msg.stream_token("\n\n**Unsupported claims:** " + "; ".join(unsupported))
    await msg.update()


@cl.on_feedback
async def on_feedback(feedback: cl.Feedback) -> None:
    from app.store.sqlite import connect, migrate

    migrate()
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO feedback(user_id,thread_id,message_id,value,comment,created_at) VALUES (?,?,?,?,?,datetime('now'))",
            (_user_id(), None, feedback.for_id, 1 if feedback.value else 0, feedback.comment),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


if __name__ == "__main__":
    pass

