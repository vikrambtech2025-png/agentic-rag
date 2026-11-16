"""Cross-thread memory: a single summarised profile row."""
from __future__ import annotations

from datetime import datetime

from app.store.sqlite import connect, migrate


def get_profile() -> str:
    migrate()
    conn = connect()
    try:
        row = conn.execute("SELECT value FROM profile WHERE key='summary'").fetchone()
        return row["value"] if row else ""
    finally:
        conn.close()


def set_profile(value: str) -> None:
    migrate()
    conn = connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO profile(key,value,updated_at) VALUES (?,?,?)",
            ("summary", value, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def remember(fact: str) -> None:
    cur = get_profile()
    new = (cur + "\n- " + fact).strip() if cur else "- " + fact
    set_profile(new)


if __name__ == "__main__":
    remember("prefers concise answers")
    print(get_profile())
