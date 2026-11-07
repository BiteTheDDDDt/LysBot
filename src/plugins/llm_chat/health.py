"""Runtime readiness with database integrity and OneBot connection state."""
import asyncio
import json
import os
import sqlite3
import time
from contextlib import closing

from nonebot import get_bots, logger

from .paths import data_path

_task = None


def check_database() -> bool:
    path = data_path("history.db").resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
        db.execute("PRAGMA query_only=ON")
        return db.execute("PRAGMA quick_check(1)").fetchall() == [("ok",)]


def write_health(database_ok: bool, error: str = "") -> None:
    from .conversation import task_snapshot
    path = data_path("health.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "pid": os.getpid(), "updated_at": time.time(), "plugin_ready": True,
        "database_ok": database_ok, "onebot_connected": bool(get_bots()),
        "error": error,
        "work": task_snapshot(),
    }), encoding="utf-8")
    temporary.replace(path)


async def _monitor() -> None:
    ok, error = False, "database check pending"
    next_check = 0.0
    try:
        while True:
            if time.monotonic() >= next_check:
                try:
                    ok = await asyncio.to_thread(check_database)
                    error = "" if ok else "database integrity check failed"
                except Exception as exc:
                    ok, error = False, f"{type(exc).__name__}: {exc}"
                if not ok:
                    logger.error(f"[health] {error}")
                next_check = time.monotonic() + 60
            write_health(ok, error)
            await asyncio.sleep(10)
    finally:
        data_path("health.json").unlink(missing_ok=True)


def start() -> None:
    global _task
    if _task is None or _task.done():
        write_health(False, "database check pending")
        _task = asyncio.create_task(_monitor(), name="bot-health")


async def close() -> None:
    global _task
    if _task:
        _task.cancel()
        await asyncio.gather(_task, return_exceptions=True)
        _task = None
