"""Durable send attempts; uncertain acknowledgements are never retried blindly."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class DeliveryOutbox:
    """Use short, independent transactions so no database lock spans network I/O."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _execute(self, operation: str, key: str, **values):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS delivery_outbox (
                delivery_key TEXT PRIMARY KEY,
                group_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                medium TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                attempt_token TEXT NOT NULL,
                message_id TEXT,
                error_type TEXT,
                updated_at TEXT NOT NULL
            )""")
            if operation == "get":
                row = db.execute(
                    "SELECT status, medium FROM delivery_outbox WHERE delivery_key=?", (key,),
                ).fetchone()
                return {"status": row[0], "medium": row[1]} if row else None
            now = datetime.now().isoformat()
            if operation == "claim":
                cursor = db.execute("""INSERT INTO delivery_outbox
                    (delivery_key, group_id, status, medium, payload_sha256, attempt_token, updated_at)
                    VALUES (?, ?, 'sending', ?, ?, ?, ?)
                    ON CONFLICT(delivery_key) DO UPDATE SET
                        status='sending', medium=excluded.medium,
                        payload_sha256=excluded.payload_sha256, attempt_token=excluded.attempt_token,
                        message_id=NULL,
                        error_type=NULL, updated_at=excluded.updated_at
                    WHERE delivery_outbox.status='failed'""",
                    (key, values["group_id"], values["medium"],
                     values["payload_sha256"], values["attempt_token"], now),
                )
                return values["attempt_token"] if cursor.rowcount == 1 else None
            db.execute("""UPDATE delivery_outbox SET status=?, message_id=?, error_type=?, updated_at=?
                WHERE delivery_key=? AND status='sending' AND attempt_token=?""",
                (values["status"], values.get("message_id"), values.get("error_type"), now, key,
                 values["attempt_token"]),
            )

    async def blocked(self, key: str) -> bool:
        record = await asyncio.to_thread(self._execute, "get", key)
        return bool(record and record["status"] in {"sending", "sent", "unknown"})

    async def claim(self, key: str, *, group_id: int, medium: str, payload: dict) -> str | None:
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return await asyncio.to_thread(
            self._execute, "claim", key, group_id=group_id, medium=medium,
            payload_sha256=digest, attempt_token=uuid4().hex,
        )

    async def finish(self, key: str, status: str, *, attempt_token: str, message_id=None, error_type=None) -> None:
        await asyncio.to_thread(
            self._execute, "finish", key, status=status, attempt_token=attempt_token,
            message_id=str(message_id) if message_id is not None else None, error_type=error_type,
        )
