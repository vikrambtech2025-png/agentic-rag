"""SQLite schema + connection helpers for ragbuddy.db (our data, not Chainlit's)."""
from __future__ import annotations

import sqlite3

from app.config import settings

DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    pages INTEGER,
    sha256 TEXT UNIQUE,
    added_at TEXT NOT NULL,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page INTEGER,
    heading TEXT,
    text TEXT NOT NULL,
    tokens INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, heading, chunk_id UNINDEXED);
CREATE TABLE IF NOT EXISTS web_cache (
    query TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.app_db, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def migrate() -> sqlite3.Connection:
    conn = connect()
    conn.executescript(DDL)
    conn.commit()
    # verify FTS5
    opts = {r[0] for r in conn.execute("PRAGMA compile_options")}
    if "ENABLE_FTS5" not in opts:
        raise RuntimeError("SQLite built without FTS5; install pysqlite3-binary")
    return conn


def main() -> None:
    c = migrate()
    print("migrated", settings.app_db)
    print("compile_options has FTS5:", "ENABLE_FTS5" in {r[0] for r in c.execute("PRAGMA compile_options")})
    c.close()


if __name__ == "__main__":
    main()

