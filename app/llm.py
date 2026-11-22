"""LLM client with retry, jitter, and fallback models."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Callable, Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import settings

log = logging.getLogger("ragbuddy.llm")

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _retryable_exc(e: Exception) -> bool:
    if isinstance(e, _RETRYABLE):
        return True
    if isinstance(e, APIStatusError) and getattr(e, "status_code", 0) >= 500:
        return True
    return False


def _fallback_models() -> list[str]:
    raw = os.environ.get("LLM_FALLBACK_MODELS", "") or getattr(settings, "llm_fallback_models", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _client_kwargs(model: str) -> dict[str, Any]:
    return {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8, jitter=2),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _chat_once(kwargs: dict[str, Any]) -> str:
    resp = _openai_client().chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def _openai_client():
    from openai import OpenAI

    return OpenAI(base_url=settings.effective_base_url, api_key=settings.effective_api_key)


def chat(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    response_format: dict | None = None,
    max_tokens: int | None = None,
) -> str:
    m = model or settings.effective_model
    kwargs: dict[str, Any] = dict(model=m, messages=messages, temperature=temperature)
    if response_format is not None:
        kwargs["response_format"] = response_format
    kwargs["max_tokens"] = max_tokens if max_tokens is not None else 512

    candidates = [m] + [fm for fm in _fallback_models() if fm != m]
    last_err: Exception | None = None
    for cand in candidates:
        kwargs["model"] = cand
        try:
            return _chat_once(kwargs)
        except Exception as e:
            last_err = e
            if not _retryable_exc(e):
                raise
            log.warning("chat failed on %s: %s", cand, e)
            # try next fallback
    if last_err is not None:
        raise last_err
    raise RuntimeError("no model available")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8, jitter=2),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _stream_once(kwargs: dict[str, Any], sink):
    acc = ""
    for chunk in _openai_client().chat.completions.create(**kwargs):
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            acc += delta
            if sink is not None:
                sink(delta)
    return acc


def stream(
    messages: list[dict[str, Any]],
    sink: Callable[[str], None] | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    m = model or settings.effective_model
    kwargs: dict[str, Any] = dict(model=m, messages=messages, temperature=temperature, stream=True)
    kwargs["max_tokens"] = max_tokens if max_tokens is not None else 1000

    candidates = [m] + [fm for fm in _fallback_models() if fm != m]
    last_err: Exception | None = None
    for cand in candidates:
        kwargs["model"] = cand
        try:
            return _stream_once(kwargs, sink)
        except Exception as e:
            last_err = e
            if not _retryable_exc(e):
                raise
            log.warning("stream failed on %s: %s", cand, e)
    if last_err is not None:
        raise last_err
    raise RuntimeError("no model available")


def parse_json(raw: str) -> dict[str, Any]:
    import re

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
