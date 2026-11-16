"""Heading-aware, token-bounded chunker."""
from __future__ import annotations

import hashlib
import re
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingest.loaders import LoadedDoc
from app.models import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(s: str) -> int:
    return len(_ENCODER.encode(s or ""))


def _split_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown/text into (heading_path, body) sections."""
    lines = (text or "").splitlines()
    stack: list[str] = []
    sections: list[tuple[str, str]] = []
    cur: list[str] = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if cur and any(c.strip() for c in cur):
                sections.append((" > ".join(stack), "\n".join(cur).strip()))
            cur = []
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1] + [title]
        else:
            cur.append(line)
    if cur and any(c.strip() for c in cur):
        sections.append((" > ".join(stack), "\n".join(cur).strip()))
    if not sections:
        sections.append(("", (text or "").strip()))
    return sections


def chunk_document(doc: LoadedDoc, source: str, doc_id: str) -> list[Chunk]:
    chunk_size = settings.chunk_tokens
    overlap = int(chunk_size * settings.chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[Chunk] = []
    ordinal = 0
    for page_no, text in doc.pages:
        if not (text or "").strip():
            continue
        for heading, body in _split_headings(text):
            if not body.strip():
                continue
            pieces = splitter.split_text(body)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                embed_text = f"{heading}\n{piece}" if heading else piece
                cid = hashlib.sha256(f"{source}|{page_no}|{ordinal}|{piece}".encode()).hexdigest()[:32]
                out.append(
                    Chunk(
                        id=cid,
                        doc_id=doc_id,
                        text=piece,
                        page=page_no,
                        heading=heading,
                        source=source,
                        tokens=count_tokens(piece),
                    )
                )
                ordinal += 1
    # re-index ordinals not needed; Chunk.id already unique
    return out


if __name__ == "__main__":
    # self-test
    d = LoadedDoc(pages=[(1, "# H1\npara one.\n## H2\npara two with more words " * 5)])
    cs = chunk_document(d, "t", "d1")
    print("chunks:", len(cs))
    for c in cs:
        print(c.page, c.heading, c.tokens, c.text[:60])
