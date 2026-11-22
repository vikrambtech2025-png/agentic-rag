from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    text: str = ""
    page: int = 0
    heading: str = ""
    source: str = ""
    score: float = 0.0
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page": self.page,
            "heading": self.heading,
            "source": self.source,
            "score": self.score,
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Citation:
    n: int
    chunk_id: str
    source: str
    page: int
    snippet: str


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


Route = Literal["chat", "docs", "web", "code", "mixed"]


class GraphState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    question: str
    sub_questions: list[str]
    route: Route
    chunks: list[dict[str, Any]]
    web: list[dict[str, Any]]
    hops: int
    sufficient: bool
    answer: str
    citations: list[dict[str, Any]]
    unsupported: list[str]
    thread_id: str
    user_id: str
