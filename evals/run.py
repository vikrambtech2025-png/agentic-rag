"""Eval runner: recall@k, citation precision, refusal correctness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.build import get_app
from app.ingest.pipeline import ingest_path
from app.ingest.code import ingest_repo

APP = get_app()
GOLDEN = Path(__file__).parent / "golden.jsonl"

# Ensure a reproducible corpus exists.
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample.pdf"
if FIXTURE.exists():
    ingest_path(str(FIXTURE))
ingest_repo(str(Path(__file__).resolve().parents[1]))


def _is_refusal(result: dict) -> bool:
    cites = result.get("citations") or []
    local = [c for c in cites if c.get("chunk_id")]
    return len(local) == 0


def run() -> None:
    lines = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    recalls = []
    precisions = []
    refusals = []
    for item in lines:
        q = item["q"]
        expected = set(item["expected_chunk_ids"])
        must_refuse = item["must_refuse"]
        cfg = {"configurable": {"thread_id": f"eval-{hash(q)}"}}
        state = {"question": q, "messages": []}
        res = APP.invoke(state, config=cfg)
        retrieved = {c.get("id") for c in (res.get("chunks") or [])}
        cited = [c.get("chunk_id") for c in (res.get("citations") or []) if c.get("chunk_id")]
        # recall@k
        if expected:
            recall = len(expected & retrieved) / len(expected)
        else:
            recall = 1.0
        recalls.append(recall)
        # citation precision
        if cited:
            prec = sum(1 for cid in cited if cid in expected) / len(cited)
        else:
            prec = 1.0 if must_refuse else 0.0
        precisions.append(prec)
        # refusal
        refused = _is_refusal(res)
        refusals.append(refused == must_refuse)
        print(f"Q: {q[:60]}")
        print(f"   route={res.get('route')} hops={res.get('hops')} recall={recall:.2f} cite_prec={prec:.2f} refused_ok={refused == must_refuse}")

    print("\n=== SUMMARY ===")
    print(f"recall@k avg     : {sum(recalls)/len(recalls):.3f}")
    print(f"cite precision   : {sum(precisions)/len(precisions):.3f}")
    print(f"refusal accuracy : {sum(refusals)/len(refusals):.3f}")


if __name__ == "__main__":
    run()
