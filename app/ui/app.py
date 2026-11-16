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


def _thread_id() -> str:
    return cl.user_session.get("thread_id") or "default"


@cl.on_chat_start
async def on_start() -> None:
    cl.user_session.set("thread_id", os.urandom(8).hex())
    await cl.Message(
        content=(
            "Hi! I'm your agentic RAG buddy. Upload a PDF/MD/TXT, paste a URL, "
            "or ask a question. I'll show my reasoning steps and cite sources."
        )
    ).send()


async def _ingest_file(file: cl.File) -> str:
    dest = settings.uploads_path / Path(file.name).name
    shutil.copyfile(file.path, dest)
    doc_id = await asyncio.to_thread(ingest_path, str(dest))
    return doc_id


_RATE: dict[str, list[float]] = {}


def _rate_ok(thread_id: str, limit: int = 20, window: float = 60.0) -> bool:
    import time

    now = time.time()
    bucket = [t for t in _RATE.get(thread_id, []) if now - t < window]
    if len(bucket) >= limit:
        _RATE[thread_id] = bucket
        return False
    bucket.append(now)
    _RATE[thread_id] = bucket
    return True


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thread_id = _thread_id()
    if not _rate_ok(thread_id):
        await cl.Message(content="Rate limit exceeded. Please wait a minute.").send()
        return
    # Handle uploads
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
            if "youtube.com" in text or "youtu.be" in text:
                did = await asyncio.to_thread(ingest_youtube, text)
            else:
                did = await asyncio.to_thread(ingest_url, text)
            await cl.Message(content=f"Ingested (id {did[:8]}). Ask away!").send()
        except Exception as e:
            await cl.Message(content=f"Failed: {e}").send()
        return

    if not text:
        return

    thread_id = _thread_id()
    state = {"question": text, "messages": []}
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
        await step.remove()
        await msg.stream_token(f"\n[error] {e}")
        await msg.update()
        return

    await step.remove()

    answer = result.get("answer", "") if isinstance(result, dict) else ""
    # if streaming already pushed tokens, answer may duplicate; clear content then set
    # We streamed tokens already, so don't overwrite; just attach sources.
    citations = result.get("citations", []) if isinstance(result, dict) else []
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
            "INSERT INTO feedback(thread_id,message_id,value,comment,created_at) VALUES (?,?,?,?,datetime('now'))",
            (None, feedback.for_id, 1 if feedback.value else 0, feedback.comment),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


if __name__ == "__main__":
    pass

