"""In-memory metrics with Prometheus text export."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class _Counter:
    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._v = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, label: str = "", amount: float = 1.0) -> None:
        with self._lock:
            self._v[label] += amount

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._v)


class _Histogram:
    def __init__(self, name: str, help_text: str, buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)):
        self.name = name
        self.help = help_text
        self.buckets = buckets
        self._sum = defaultdict(float)
        self._count = defaultdict(int)
        self._bucket = defaultdict(lambda: [0] * len(buckets))
        self._lock = threading.Lock()

    def observe(self, label: str, value: float) -> None:
        with self._lock:
            self._sum[label] += value
            self._count[label] += 1
            for i, b in enumerate(self.buckets):
                if value <= b:
                    self._bucket[label][i] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sum": dict(self._sum),
                "count": dict(self._count),
                "bucket": {k: list(v) for k, v in self._bucket.items()},
            }


REQUESTS_TOTAL = _Counter("agentic_rag_requests_total", "Total chat requests")
REQUEST_DURATION = _Histogram("agentic_rag_request_duration_seconds", "Chat request duration")
NODE_DURATION = _Histogram("agentic_rag_node_duration_seconds", "Per-node duration")
NODE_ERRORS = _Counter("agentic_rag_node_errors_total", "Per-node errors")
LLM_CALLS = _Counter("agentic_rag_llm_calls_total", "LLM calls")
LLM_ERRORS = _Counter("agentic_rag_llm_errors_total", "LLM errors")
INGEST_JOBS = _Counter("agentic_rag_ingest_jobs_total", "Ingest jobs by status")


def prometheus_text() -> str:
    lines = []
    # counters
    for c in (REQUESTS_TOTAL, NODE_ERRORS, LLM_CALLS, LLM_ERRORS, INGEST_JOBS):
        lines.append(f"# HELP {c.name} {c.help}")
        lines.append(f"# TYPE {c.name} counter")
        for label, val in c.snapshot().items():
            lbl = f'{{label="{label}"}}' if label else ""
            lines.append(f"{c.name}{lbl} {val}")
    # histograms (simplified: just sum/count)
    for h in (REQUEST_DURATION, NODE_DURATION):
        lines.append(f"# HELP {h.name} {h.help}")
        lines.append(f"# TYPE {h.name} summary")
        snap = h.snapshot()
        for label in set(list(snap["sum"].keys()) + list(snap["count"].keys())):
            lines.append(f'{h.name}_sum{{label="{label}"}} {snap["sum"].get(label,0)}')
            lines.append(f'{h.name}_count{{label="{label}"}} {snap["count"].get(label,0)}')
    return "\n".join(lines) + "\n"


def json_snapshot() -> dict[str, Any]:
    return {
        "requests_total": REQUESTS_TOTAL.snapshot(),
        "request_duration": REQUEST_DURATION.snapshot(),
        "node_duration": NODE_DURATION.snapshot(),
        "node_errors": NODE_ERRORS.snapshot(),
        "llm_calls": LLM_CALLS.snapshot(),
        "llm_errors": LLM_ERRORS.snapshot(),
        "ingest_jobs": INGEST_JOBS.snapshot(),
    }


if __name__ == "__main__":
    REQUESTS_TOTAL.inc("chat")
    REQUEST_DURATION.observe("chat", 1.2)
    print(prometheus_text()[:300])
