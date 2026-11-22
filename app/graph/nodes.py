"""Graph nodes."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("ragbuddy.graph")

from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.graph.prompts import (
    GENERATE_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    VERIFY_PROMPT,
)
from app.ingest.loaders import sha256_text as _  # noqa
from app.llm import chat, parse_json, stream
from app.memory import get_profile
from app.models import GraphState
from app.store.hybrid import search as hybrid_search
from app.tools.web import web_search as _web_search


def _cfg(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable", {}) or {}


def route_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    q = state.get("question") or ""
    if not q and state.get("messages"):
        for m in reversed(state["messages"]):
            c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if c:
                q = c
                break
    try:
        messages = [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": q},
        ]
        raw = chat(messages, model=settings.effective_router_model, temperature=0.0, response_format={"type": "json_object"})
        data = parse_json(raw)
        route = data.get("route", "docs")
        if route not in {"chat", "docs", "web", "code", "mixed"}:
            route = "docs"
        subs = data.get("sub_questions") or [q]
        subs = [s for s in subs if isinstance(s, str) and s.strip()][:3]
        if not subs:
            subs = [q]
    except Exception as e:
        log.warning("route failed: %s", e)
        route = "docs"
        subs = [q]
    return {"question": q, "route": route, "sub_questions": subs, "hops": state.get("hops", 0)}


def retrieve_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    qs = state.get("sub_questions") or [state.get("question", "")]
    seen: set[str] = set()
    chunks: list[dict[str, Any]] = list(state.get("chunks") or [])
    seen.update(c.get("id") for c in chunks)
    for q in qs:
        try:
            res = hybrid_search(q, k=settings.top_k, user_id=state.get("user_id"))
        except Exception:
            res = []
        for r in res:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            chunks.append(r)
    # cap
    chunks = chunks[: settings.top_k]
    out: dict[str, Any] = {"chunks": chunks, "hops": (state.get("hops", 0) + 1)}
    if state.get("route") == "mixed":
        # also pull web results
        web = list(state.get("web") or [])
        seen = {w.get("url") for w in web}
        for q in (state.get("sub_questions") or [state.get("question", "")]):
            try:
                res = _web_search(q)
            except Exception:
                res = []
            for r in res:
                if r.get("url") in seen:
                    continue
                seen.add(r.get("url"))
                web.append(r)
        out["web"] = web[:8]
    return out


def grade_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    q = state.get("question", "")
    chunks = state.get("chunks") or []
    if not chunks:
        return {"sufficient": state.get("route") == "chat"}
    ctx = "\n\n".join(f"[{i+1}] {c.get('text','')[:500]}" for i, c in enumerate(chunks[:8]))
    try:
        messages = [
            {"role": "system", "content": GRADE_PROMPT},
            {"role": "user", "content": f"Question: {q}\n\nPassages:\n{ctx}"},
        ]
        raw = chat(messages, model=settings.effective_router_model, temperature=0.0, response_format={"type": "json_object"})
        data = parse_json(raw)
        sufficient = bool(data.get("sufficient", False))
    except Exception as e:
        log.warning("grade failed: %s", e)
        sufficient = True  # fail open: try to generate
    return {"sufficient": sufficient}


def rewrite_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    q = state.get("question", "")
    chunks = state.get("chunks") or []
    ctx = "\n".join(c.get("text", "")[:200] for c in chunks[:5])
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Original: {q}\n\nTried passages:\n{ctx}"},
    ]
    raw = chat(messages, model=settings.effective_router_model, temperature=0.3, response_format={"type": "json_object"})
    data = parse_json(raw)
    newq = data.get("query") or q
    return {"question": newq, "sub_questions": [newq]}


def web_search_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    qs = state.get("sub_questions") or [state.get("question", "")]
    web: list[dict[str, Any]] = list(state.get("web") or [])
    seen = {w.get("url") for w in web}
    for q in qs:
        try:
            res = _web_search(q)
        except Exception:
            res = []
        for r in res:
            if r.get("url") in seen:
                continue
            seen.add(r.get("url"))
            web.append(r)
    return {"web": web[:8]}


def _build_context(state: GraphState) -> tuple[str, list[dict[str, Any]]]:
    chunks = (state.get("chunks") or [])[:4]
    web = (state.get("web") or [])[:3]
    items: list[tuple[str, str, str]] = []  # (label, text, source)
    n = 1
    cit_map: list[dict[str, Any]] = []
    for c in chunks:
        items.append((str(n), c.get("text", ""), f"{c.get('source','')} p.{c.get('page')}"))
        cit_map.append({"n": n, "chunk_id": c.get("id"), "source": c.get("source"), "page": c.get("page"), "snippet": c.get("text", "")[:300]})
        n += 1
    for w in web:
        items.append((str(n), w.get("snippet", ""), w.get("url", "")))
        cit_map.append({"n": n, "chunk_id": None, "source": w.get("url"), "page": 0, "snippet": w.get("snippet", "")[:300]})
        n += 1
    lines = [f"[{lab}] ({src})\n{text}" for lab, text, src in items]
    return "\n\n".join(lines), cit_map


def generate_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    q = state.get("question", "")
    cfg = _cfg(config)
    sink = cfg.get("stream_sink")
    context, cit_map = _build_context(state)
    if not context.strip():
        context = "(no retrieved context)"
    prof = get_profile()
    sys_prompt = GENERATE_PROMPT.format(context=context)
    if prof:
        sys_prompt = "User profile / preferences:\n" + prof + "\n\n" + sys_prompt
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": q},
    ]
    try:
        answer = stream(messages, sink=sink, temperature=0.2)
    except Exception as e:
        log.warning("generate failed: %s", e)
        answer = "I'm sorry, I couldn't generate an answer right now. Please try again."
    # keep only citations actually referenced in the answer
    import re

    used = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    if used:
        cit_map = [c for c in cit_map if c["n"] in used]
    else:
        cit_map = []
    return {"answer": answer, "citations": cit_map}


def verify_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    answer = state.get("answer", "")
    chunks = state.get("chunks") or []
    context = "\n\n".join(f"[{i+1}] {c.get('text','')[:400]}" for i, c in enumerate(chunks))
    web = state.get("web") or []
    for i, w in enumerate(web, start=len(chunks) + 1):
        context += f"\n\n[{i}] {w.get('snippet','')[:400]}"
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": f"Answer:\n{answer}\n\nSources:\n{context}"},
    ]
    raw = chat(messages, model=settings.effective_router_model, temperature=0.0, response_format={"type": "json_object"})
    data = parse_json(raw)
    unsupported = data.get("unsupported") or []
    if not isinstance(unsupported, list):
        unsupported = []
    return {"unsupported": unsupported}


def chat_node(state: GraphState, config: RunnableConfig = None) -> dict[str, Any]:
    q = state.get("question", "")
    messages = [{"role": "user", "content": q}]
    answer = chat(messages, temperature=0.4)
    return {"answer": answer, "citations": [], "unsupported": []}
