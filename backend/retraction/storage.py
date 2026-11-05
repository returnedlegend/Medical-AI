"""Small durable job store. A connection belongs to one operation/thread."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "runs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, status TEXT NOT NULL,
                progress REAL NOT NULL, config TEXT NOT NULL,
                error TEXT, summary TEXT, result TEXT
            )""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row: sqlite3.Row | None, include_result: bool = False) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for key in ("config", "summary", "result"):
            if key in item:
                item[key] = json.loads(item[key]) if item[key] is not None else None
        if not include_result:
            item.pop("result", None)
        return item

    def recover_interrupted(self) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? WHERE status IN ('queued','running')",
                ("Server restarted before this run finished. Re-run the saved configuration.", timestamp()),
            )

    def create(self, config: dict) -> dict:
        now = timestamp()
        run_id = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, now, now, "queued", 0.0, json.dumps(config, allow_nan=False), None, None, None),
            )
        return self.get(run_id)

    def get(self, run_id: str, include_result: bool = False) -> dict | None:
        columns = "*" if include_result else "id,created_at,updated_at,status,progress,config,error,summary"
        with self.connect() as db:
            row = db.execute(f"SELECT {columns} FROM runs WHERE id=?", (run_id,)).fetchone()
        return self.decode(row, include_result)

    def list(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,created_at,updated_at,status,progress,config,error,summary FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self.decode(row) for row in rows]

    def active_count(self) -> int:
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]

    def update(self, run_id: str, **fields: Any) -> None:
        allowed = {"status", "progress", "error", "summary", "result"}
        if not fields or not set(fields).issubset(allowed):
            raise ValueError("Invalid job update fields")
        for key in ("summary", "result"):
            if key in fields:
                fields[key] = json.dumps(fields[key], allow_nan=False)
        fields["updated_at"] = timestamp()
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                [*fields.values(), run_id],
            )
