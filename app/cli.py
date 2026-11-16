"""CLI harness: ask a question, see the agentic trace."""
from __future__ import annotations

import argparse
import json

from app.logging_config import setup as setup_logging

setup_logging()

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from app.graph.build import get_app

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--thread", default="cli")
    parser.add_argument("--trace", action="store_true", default=True)
    args = parser.parse_args()
    question = " ".join(args.question)

    app = get_app()
    config = {"configurable": {"thread_id": args.thread}}
    state = {"question": question, "messages": []}

    console.rule(f"[bold]Question[/bold]: {question}")
    final = None
    for chunk in app.stream(state, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if args.trace:
                _print_node(node, update)
            final = update if final is None else {**final, **update}
    # final state is the last update only; re-fetch via invoke? We already have merged partials (not complete). Use get_state.
    try:
        snap = app.get_state(config)
        final = snap.values
    except Exception:
        pass

    console.rule("[bold]Answer[/bold]")
    console.print(final.get("answer", "") if final else "")
    cites = final.get("citations", []) if final else []
    if cites:
        console.print("\n[bold]Sources[/bold]")
        for c in cites:
            console.print(f"  [{c['n']}] {c.get('source')} p.{c.get('page')}")
    unsupported = final.get("unsupported", []) if final else []
    if unsupported:
        console.print("\n[red]Unsupported claims:[/red]")
        for u in unsupported:
            console.print(f"  - {u}")
    console.print(f"\n[dim]route={final.get('route')} hops={final.get('hops')}[/dim]")


def _print_node(node: str, update: dict) -> None:
    tree = Tree(f"[cyan]{node}[/cyan]")
    if node == "route":
        tree.add(f"route={update.get('route')} subs={update.get('sub_questions')}")
    elif node == "retrieve":
        ch = update.get("chunks", [])
        tree.add(f"+{len(ch)} chunks (total seen)")
        for c in ch[-3:]:
            tree.add(f"  p.{c.get('page')} {c.get('text','')[:60]}")
    elif node == "grade":
        tree.add(f"sufficient={update.get('sufficient')}")
    elif node == "rewrite":
        tree.add(f"new query={update.get('question')}")
    elif node == "web":
        tree.add(f"+{len(update.get('web', []))} web results")
    elif node == "generate":
        tree.add(f"answer len={len(update.get('answer',''))}")
    elif node == "verify":
        tree.add(f"unsupported={update.get('unsupported')}")
    console.print(tree)


if __name__ == "__main__":
    main()
