"""Shared LLM helpers built on the OpenAI-compatible client."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Iterator

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger("ragbuddy.llm")

from app.config import settings


def client() -> OpenAI:
    return OpenAI(api_key=settings.effective_api_key, base_url=settings.effective_base_url)


_RETRYABLE = (Exception,)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _chat_call(kwargs: dict[str, Any]) -> str:
    resp = client().chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


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
    try:
        return _chat_call(kwargs)
    except Exception as e:
        log.warning("chat failed: %s", e)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _stream_call(kwargs: dict[str, Any], sink):
    acc = ""
    for chunk in client().chat.completions.create(**kwargs):
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
    try:
        return _stream_call(kwargs, sink)
    except Exception as e:
        log.warning("stream failed: %s", e)
        raise


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        return {}
