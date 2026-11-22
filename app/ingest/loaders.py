"""Document loaders -> LoadedDoc (pages: list[(page_no, text)])."""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pymupdf4llm
from youtube_transcript_api import YouTubeTranscriptApi

import ipaddress
import socket

from app.config import settings
from app.security import (
    check_pdf_pages,
    redact_enabled,
    redact_pii,
    safe_fetch,
    sniff_mime,
)

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTS = {".pdf", ".md", ".markdown", ".txt", ".text"}


@dataclass
class LoadedDoc:
    pages: list[tuple[int, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    kind: str = "text"
    title: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(t for _, t in self.pages)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File too large (> {MAX_FILE_BYTES // (1024*1024)} MB)")


def load_pdf(path: str | Path) -> LoadedDoc:
    p = Path(path)
    _check_file(p)
    import fitz

    with fitz.open(str(p)) as d:
        check_pdf_pages(d.page_count)
    pages_raw = pymupdf4llm.to_markdown(str(p), page_chunks=True)
    # fallback to rag path if layout parser yields no text
    try:
        text = "\n".join((c.get("text") or "") for c in pages_raw)
    except Exception:
        text = ""
    if len(text.strip()) < 50:
        try:
            import pymupdf4llm.helpers.pymupdf_rag as _rag

            pages_raw = _rag.to_markdown(str(p), page_chunks=True)
        except Exception:
            pass
    pages = []
    for c in pages_raw:
        pn = c.get("metadata", {}).get("page_number", 0)
        pages.append((int(pn) if pn else 0, c.get("text", "") or ""))
    title = Path(path).stem
    text = "\n".join(t for _, t in pages)
    if len(text.strip()) < 50:
        ocr_text = _ocr_pdf(p)
        if ocr_text and len(ocr_text.strip()) >= 50:
            return LoadedDoc(pages=[(1, ocr_text)], meta={"pages": 1, "ocr": True}, kind="pdf", title=title)
        raise ValueError("PDF has no extractable text (scanned?). Install rapidocr-onnxruntime or pytesseract for OCR.")
    return LoadedDoc(pages=pages, meta={"pages": len(pages)}, kind="pdf", title=title)


def load_markdown(path: str | Path) -> LoadedDoc:
    p = Path(path)
    _check_file(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    if redact_enabled():
        text = redact_pii(text)
    return LoadedDoc(pages=[(1, text)], meta={}, kind="markdown", title=Path(path).stem)


def load_text(path: str | Path) -> LoadedDoc:
    p = Path(path)
    _check_file(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    if redact_enabled():
        text = redact_pii(text)
    return LoadedDoc(pages=[(1, text)], meta={}, kind="text", title=Path(path).stem)


def _safe_url(url: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) URLs are allowed")
    host = parsed.hostname or ""
    if not host:
        raise ValueError("Missing host")
    # block private/loopback/link-local
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError("Cannot resolve host")
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Refusing to fetch private/internal host (SSRF protection)")


def load_url(url: str) -> LoadedDoc:
    resp = safe_fetch(url)
    html = resp.text
    # naive html -> text
    text = _html_to_text(html)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = title_match.group(1).strip() if title_match else urlparse(url).netloc
    return LoadedDoc(pages=[(1, text)], meta={"url": url}, kind="url", title=title)


def _html_to_text(html: str) -> str:
    # strip scripts/styles
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.I | re.S)
    # convert common block tags to newlines
    html = re.sub(r"<(br|p|div|li|h[1-6]|tr)[^>]*>", "\n", html, flags=re.I)
    # strip remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    import html as _html

    text = _html.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_youtube(url: str) -> LoadedDoc:
    # extract video id
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError("Could not parse YouTube video id")
    vid = m.group(1)
    api = YouTubeTranscriptApi()
    transcript = api.fetch(vid)
    # group into ~60s windows
    pages: list[tuple[int, str]] = []
    bucket: list[str] = []
    bucket_start = 0
    cur_start = 0
    for snip in transcript:
        t = getattr(snip, "text", None)
        if t is None:
            t = snip.get("text", "") if isinstance(snip, dict) else str(snip)
        start = getattr(snip, "start", None)
        if start is None:
            start = snip.get("start", 0) if isinstance(snip, dict) else 0
        if not bucket:
            bucket_start = int(start)
            cur_start = bucket_start
        if start - bucket_start >= 60:
            pages.append((bucket_start // 60 + 1, " ".join(bucket)))
            bucket = []
            bucket_start = int(start)
        bucket.append(t)
    if bucket:
        pages.append((bucket_start // 60 + 1, " ".join(bucket)))
    if not pages:
        raise ValueError("Empty transcript")
    return LoadedDoc(pages=pages, meta={"video_id": vid}, kind="youtube", title=f"youtube:{vid}")


def sha256_text(text: str) -> str:
    return _sha256(text)


def new_id() -> str:
    return str(uuid.uuid4())


def _ocr_pdf(path: Path) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        result, _ = ocr(str(path))
        if result:
            return "\n".join(r[1] for r in result)
    except Exception:
        pass
    try:
        import fitz
        import pytesseract
        from PIL import Image

        doc = fitz.open(str(path))
        out = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img))
        return "\n".join(out)
    except Exception:
        return ""

