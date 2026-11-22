"""Tracing (OpenTelemetry) + structured spans."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable

log = logging.getLogger("ragbuddy.obs")

_TRACER = None


def setup_tracing() -> None:
    global _TRACER
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": os.environ.get("SERVICE_NAME", "agentic-rag")})
        provider = TracerProvider(resource=resource)
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
                log.info("OTLP tracing enabled -> %s", endpoint)
            except Exception as e:
                log.warning("OTLP exporter unavailable: %s", e)
        elif os.environ.get("LOG_TRACES", "").lower() in {"1", "true", "yes"}:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("agentic-rag")
    except Exception as e:
        log.warning("tracing setup failed: %s", e)
        _TRACER = None


def get_tracer():
    global _TRACER
    if _TRACER is None:
        try:
            from opentelemetry import trace

            _TRACER = trace.get_tracer("agentic-rag")
        except Exception:
            _TRACER = _NoopTracer()
    return _TRACER


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def set_attribute(self, *a, **k):
        pass

    def set_status(self, *a, **k):
        pass

    def record_exception(self, *a, **k):
        pass


class _NoopTracer:
    def start_as_current_span(self, name, **kw):
        return _NoopSpan()


@contextmanager
def span(name: str, **attrs: Any):
    tracer = get_tracer()
    try:
        with tracer.start_as_current_span(name) as s:
            for k, v in attrs.items():
                try:
                    s.set_attribute(k, v)
                except Exception:
                    pass
            yield s
    except Exception:
        # noop tracer path
        yield _NoopSpan()


def trace_node(name: str, fn: Callable):
    def wrapped(state, config=None):
        import time

        from app.metrics import NODE_DURATION, NODE_ERRORS

        t0 = time.perf_counter()
        try:
            with span(f"node.{name}"):
                out = fn(state, config)
            NODE_DURATION.observe(name, time.perf_counter() - t0)
            return out
        except Exception as e:
            NODE_ERRORS.inc(name)
            NODE_DURATION.observe(name, time.perf_counter() - t0)
            raise

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


if __name__ == "__main__":
    setup_tracing()
    print("tracing ready")
