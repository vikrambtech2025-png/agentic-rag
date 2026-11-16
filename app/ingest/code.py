"""Code indexer using tree-sitter."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from tree_sitter_language_pack import get_parser

from app.config import settings
from app.models import Chunk
from app.store.embeddings import embed_batch
from app.store.sqlite import connect, migrate
from app.store.vectors import upsert as vupsert
from app.store.keyword import upsert_fts as _fts  # noqa

EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

DEFAULT_EXCLUDE = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target"}


def _node_name(node, src: bytes) -> str:
    for child in node.children:
        if child.type == "identifier" or child.type == "name":
            return src[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return node.type


def _extract_symbols(tree, src: bytes, prefix: str = "") -> list[tuple[str, Any]]:
    """Return list of (symbol_name, node) for function/class definitions (top-level + methods)."""
    out: list[tuple[str, Any]] = []
    root = tree.root_node

    def walk(node, scope: str):
        for child in node.children:
            if child.type in ("function_definition", "function_declaration", "method_definition", "class_definition", "class_declaration", "impl_item", "fn_item"):
                name = _node_name(child, src)
                qual = f"{scope}.{name}" if scope else name
                out.append((qual, child))
                # recurse into class for methods
                if "class" in child.type:
                    walk(child, qual)
            else:
                # recurse into module-level (top-level only handled by root walk)
                pass
        # top-level recursion: only walk root's direct children handled above; but classes need method walk
    # Walk root children for top-level defs
    for child in root.children:
        if child.type in ("function_definition", "function_declaration", "class_definition", "class_declaration", "impl_item"):
            name = _node_name(child, src)
            out.append((name, child))
            if "class" in child.type:
                # find method defs inside
                for m in child.children:
                    if m.type in ("block", "class_body", "declaration_list"):
                        for mm in m.children:
                            if mm.type in ("function_definition", "function_declaration", "method_definition", "fn_item"):
                                mname = _node_name(mm, src)
                                out.append((f"{name}.{mname}", mm))
        elif child.type == "export_statement":
            for c in child.children:
                if c.type in ("function_declaration", "class_declaration"):
                    name = _node_name(c, src)
                    out.append((name, c))
    return out


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    lang = EXT_LANG.get(path.suffix.lower())
    if not lang:
        return []
    try:
        parser = get_parser(lang)
    except Exception:
        return []
    try:
        src = path.read_bytes()
    except Exception:
        return []
    if not src.strip():
        return []
    tree = parser.parse(src)
    rel = path.relative_to(root).as_posix()
    source = f"repo://{rel}"
    doc_id = str(uuid.uuid4())
    chunks: list[Chunk] = []
    symbols = _extract_symbols(tree, src)
    if not symbols:
        # fall back: whole file as one chunk (truncated)
        text = src.decode("utf-8", errors="replace")
        if len(text) > 4000:
            text = text[:4000]
        cid = hashlib.sha256(f"{source}|1|{rel}|{text[:80]}".encode()).hexdigest()[:32]
        chunks.append(Chunk(id=cid, doc_id=doc_id, text=text, page=1, heading=rel, source=source, tokens=len(text.split())))
        return chunks
    seen_lines: set[int] = set()
    for sym, node in symbols:
        start = node.start_point[0] + 1
        end = node.end_point[0] + 1
        if start in seen_lines:
            continue
        seen_lines.add(start)
        text = src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            continue
        # cap chunk size
        if len(text) > 3000:
            text = text[:3000]
        cid = hashlib.sha256(f"{source}|{start}|{sym}|{text[:80]}".encode()).hexdigest()[:32]
        chunks.append(
            Chunk(
                id=cid,
                doc_id=doc_id,
                text=text,
                page=start,
                heading=f"{rel}::{sym}",
                source=source,
                tokens=len(text.split()),
            )
        )
    return chunks


def ingest_repo(root: str | Path, include: list[str] | None = None) -> list[str]:
    root = Path(root).resolve()
    if include is None:
        include = list(EXT_LANG.keys())
    include_exts = {e if e.startswith(".") else "." + e for e in include}
    # allow glob-ish: if include contains "*.py" style
    exts = set()
    for pat in include:
        if "*" in pat:
            exts.add(Path(pat).suffix.lower())
        else:
            exts.add(pat.lower() if pat.startswith(".") else "." + pat.lower())
    if not exts:
        exts = set(EXT_LANG.keys())

    all_chunks: list[Chunk] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in exts:
                continue
            if p.suffix.lower() not in EXT_LANG:
                continue
            all_chunks.extend(chunk_file(p, root))

    if not all_chunks:
        return []

    # group by doc_id (file) not needed; store all chunks under a single repo doc
    repo_doc_id = str(uuid.uuid4())
    for c in all_chunks:
        c.doc_id = repo_doc_id

    # embed
    texts = [c.text for c in all_chunks]
    vecs = embed_batch(texts)
    rows = [
        {"id": c.id, "vector": v, "doc_id": repo_doc_id, "page": c.page, "heading": c.heading, "text": c.text, "source": c.source}
        for c, v in zip(all_chunks, vecs)
    ]
    vupsert(rows)

    # persist metadata
    migrate()
    conn = connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO documents(id,source,kind,title,pages,sha256,added_at,meta) VALUES (?,?,?,?,?,?,datetime('now'),?)",
            (repo_doc_id, str(root), "code", root.name, len(all_chunks), hashlib.sha256(str(root).encode()).hexdigest(), None),
        )
        for c in all_chunks:
            conn.execute(
                "INSERT INTO chunks(id,doc_id,ordinal,page,heading,text,tokens) VALUES (?,?,?,?,?,?,?)",
                (c.id, repo_doc_id, 0, c.page, c.heading, c.text, c.tokens),
            )
            conn.execute("INSERT INTO chunks_fts(text, heading, chunk_id) VALUES (?,?,?)", (c.text, c.heading, c.id))
        conn.commit()
    finally:
        conn.close()

    return [repo_doc_id]


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ids = ingest_repo(root)
    print("ingested repo", root, "doc_ids", ids)
