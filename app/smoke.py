"""Health check / entrypoint."""
from __future__ import annotations

from app.config import settings
from app.logging_config import setup as setup_logging

setup_logging()
from app.store.sqlite import migrate


def ensure_dirs() -> None:
    settings.data_path
    settings.lancedb_path
    settings.uploads_path


def main() -> None:
    ensure_dirs()
    conn = migrate()
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    print("rag-buddy smoke")
    print("  data_dir   :", settings.data_dir)
    print("  model      :", settings.effective_model, "@", settings.effective_base_url)
    print("  router     :", settings.effective_router_model)
    print("  embed      :", settings.embed_model, settings.embed_dim)
    print("  documents  :", docs)
    print("  chunks     :", chunks)
    print("  tavily key :", "set" if settings.tavily_api_key else "MISSING")
    print("  openai key :", "set" if settings.openai_api_key else "MISSING")


if __name__ == "__main__":
    main()
