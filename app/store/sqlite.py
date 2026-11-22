"""SQLite schema + connection helpers for ragbuddy.db (our data, not Chainlit's)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from app.config import settings

SCHEMA_VERSION = 2

DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    source TEXT NOT NULL,
    source_key TEXT,
    kind TEXT NOT NULL,
    title TEXT,
    pages INTEGER,
    sha256 TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'ready',
    embed_model TEXT,
    added_at TEXT NOT NULL,
    updated_at TEXT,
    meta TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL DEFAULT 0,
    page INTEGER,
    heading TEXT,
    text TEXT NOT NULL,
    tokens INTEGER,
    embedding_model TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, chunk_id UNINDEXED);

CREATE TABLE IF NOT EXISTS web_cache (
    query TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    thread_id TEXT,
    message_id TEXT,
    value INTEGER,
    comment TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    doc_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_source_key ON documents(user_id, source_key);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingest_jobs(status);
"""

# FTS sync triggers (idempotent: drop then recreate)
TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(text, heading, chunk_id) VALUES (NEW.text, NEW.heading, NEW.id);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.id;
    INSERT INTO chunks_fts(text, heading, chunk_id) VALUES (NEW.text, NEW.heading, NEW.id);
END;
"""

# column additions for upgrading older DBs (SQLite has no ADD COLUMN IF NOT EXISTS)
_ALTER = [
    ("documents", "user_id", "TEXT"),
    ("documents", "source_key", "TEXT"),
    ("documents", "version", "INTEGER NOT NULL DEFAULT 1"),
    ("documents", "status", "TEXT NOT NULL DEFAULT 'ready'"),
    ("documents", "embed_model", "TEXT"),
    ("documents", "updated_at", "TEXT"),
    ("chunks", "ordinal", "INTEGER NOT NULL DEFAULT 0"),
    ("chunks", "embedding_model", "TEXT"),
    ("feedback", "user_id", "TEXT"),
]


def _add_columns(conn: sqlite3.Connection) -> None:
    for table, col, typ in _ALTER:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            except sqlite3.OperationalError:
                pass


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.app_db, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def migrate() -> sqlite3.Connection:
    conn = connect()
    try:
        conn.executescript(DDL)
        _add_columns(conn)
        conn.executescript(INDEXES)
        conn.executescript(TRIGGERS)
        # backfill source_key for legacy rows
        conn.execute("UPDATE documents SET source_key = source WHERE source_key IS NULL")
        conn.execute("UPDATE documents SET status = 'ready' WHERE status IS NULL")
        conn.execute("UPDATE documents SET version = 1 WHERE version IS NULL")
        conn.execute("UPDATE chunks SET ordinal = 0 WHERE ordinal IS NULL")
        # record migration version
        row = conn.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1").fetchone()
        if not row or row["version"] < SCHEMA_VERSION:
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
                (SCHEMA_VERSION,),
            )
        conn.commit()
        # verify FTS5
        opts = {r[0] for r in conn.execute("PRAGMA compile_options")}
        if "ENABLE_FTS5" not in opts:
            raise RuntimeError("SQLite built without FTS5; install pysqlite3-binary")
        return conn
    except Exception:
        conn.close()
        raise


def main() -> None:
    c = migrate()
    print("migrated", settings.app_db, "v", SCHEMA_VERSION)
    c.close()


if __name__ == "__main__":
    main()
