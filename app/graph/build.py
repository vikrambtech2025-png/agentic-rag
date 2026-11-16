"""Build the LangGraph StateGraph."""
from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from app.config import settings
from app.graph.nodes import (
    chat_node,
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
    route_node,
    verify_node,
    web_search_node,
)
from app.models import GraphState


def _route_decision(state: GraphState) -> str:
    r = state.get("route", "docs")
    if r == "chat":
        return "chat"
    if r in {"web"}:
        return "web"
    # docs, code, mixed all start with retrieve
    return "retrieve"


def _grade_decision(state: GraphState) -> str:
    if state.get("sufficient"):
        return "generate"
    if (state.get("hops", 0) or 0) >= settings.max_hops:
        return "generate"
    return "rewrite"


def _hop_guard(state: GraphState) -> str:
    if (state.get("hops", 0) or 0) >= settings.max_hops:
        return "generate"
    return "retrieve"


def build_graph(checkpointer=None):
    g = StateGraph(GraphState)
    g.add_node("route", route_node)
    g.add_node("chat", chat_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("web", web_search_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)

    g.set_entry_point("route")
    g.add_conditional_edges("route", _route_decision, {"chat": "chat", "web": "web", "retrieve": "retrieve"})
    g.add_edge("chat", END)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _grade_decision, {"generate": "generate", "rewrite": "rewrite"})
    g.add_conditional_edges("rewrite", _hop_guard, {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("web", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)

    return g.compile(checkpointer=checkpointer)


def make_checkpointer():
    conn = sqlite3.connect(settings.graph_db, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return SqliteSaver(conn)


def get_app():
    return build_graph(make_checkpointer())


if __name__ == "__main__":
    app = get_app()
    print("graph built")
    print(app.get_graph().draw_mermaid())
