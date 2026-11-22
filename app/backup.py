"""Backup helper: snapshot data dir + sqlite dump."""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings


def main() -> None:
    src = Path(settings.data_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(settings.data_dir).parent / f"backup-{ts}"
    out.mkdir(parents=True, exist_ok=True)
    # copy data dir (excluding wal sidecars handled by sqlite backup)
    if src.exists():
        for p in src.iterdir():
            if p.name.endswith(("-wal", "-shm")):
                continue
            dest = out / p.name
            if p.is_dir():
                shutil.copytree(p, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(p, dest)
    # sqlite online backup
    db = Path(settings.app_db)
    if db.exists():
        conn = sqlite3.connect(str(db))
        bconn = sqlite3.connect(str(out / "ragbuddy.backup.db"))
        conn.backup(bconn)
        bconn.close()
        conn.close()
    print("backup written to", out)


if __name__ == "__main__":
    main()
