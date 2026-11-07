"""LLM 调用 token 使用统计的持久化存储。

复用 data/history.db。表结构：

    CREATE TABLE llm_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,                -- ISO 时间
        session_id TEXT NOT NULL,        -- group_xxx / user_xxx / _unknown_
        role TEXT NOT NULL,              -- chat / summary / search / ...
        spec TEXT NOT NULL,              -- 实际命中的 spec (provider:model_id)
        provider TEXT NOT NULL,          -- 平台名（dashscope / nvidia / ...）
        model_id TEXT NOT NULL,          -- 模型 id（stepfun-ai/step-3.5-flash 等）
        caller TEXT,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        latency_ms INTEGER,
        success INTEGER DEFAULT 1,
        fallback_attempt INTEGER DEFAULT 0
    )

提供 record_usage / query_session / query_global 三个异步函数。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import aiosqlite

from .logging_setup import general_logger
from .paths import data_path
from .sqlite_utils import measured_write_lock

DB_PATH = data_path("history.db")
_db: Optional[aiosqlite.Connection] = None
_db_lock = asyncio.Lock()
_write_lock = asyncio.Lock()
_initialized = False


async def _get_db() -> aiosqlite.Connection:
    global _db, _initialized
    if _db is not None and _initialized:
        return _db
    async with _db_lock:
        if _db is not None and _initialized:
            return _db
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(str(DB_PATH), timeout=30.0)
        # 与 memory.py 共用同一个 db 文件，开 WAL + busy_timeout 减少锁冲突
        try:
            await _db.execute("PRAGMA journal_mode=WAL")
            await _db.execute("PRAGMA busy_timeout=30000")
            await _db.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        await _db.execute("""
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                spec TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                caller TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_ms INTEGER,
                success INTEGER DEFAULT 1,
                fallback_attempt INTEGER DEFAULT 0,
                usage_known INTEGER NOT NULL DEFAULT 0
            )
        """)
        async with _db.execute("PRAGMA table_info(llm_usage)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if "usage_known" not in columns:
            await _db.execute("ALTER TABLE llm_usage ADD COLUMN usage_known INTEGER NOT NULL DEFAULT 0")
            # Historical zeroes also represented missing usage, so only nonzero
            # measurements can be identified as known during migration.
            await _db.execute(
                "UPDATE llm_usage SET usage_known=1 "
                "WHERE prompt_tokens>0 OR completion_tokens>0 OR total_tokens>0"
            )
        await _db.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON llm_usage(ts)")
        await _db.execute("CREATE INDEX IF NOT EXISTS idx_usage_session_ts ON llm_usage(session_id, ts)")
        await _db.execute("CREATE INDEX IF NOT EXISTS idx_usage_spec_ts ON llm_usage(spec, ts)")
        await _db.execute("CREATE INDEX IF NOT EXISTS idx_usage_provider_ts ON llm_usage(provider, ts)")
        await _db.commit()
        _initialized = True
        return _db


async def close_db() -> None:
    """供 on_shutdown 钩子使用。"""
    global _db, _initialized
    if _db is not None:
        try:
            await _db.close()
        except Exception:
            pass
        _db = None
        _initialized = False


async def record_usage(
    *,
    session_id: Optional[str],
    role: str,
    spec: str,
    caller: str,
    usage: Optional[Dict[str, int]],
    latency_ms: int,
    success: bool,
    fallback_attempt: int = 0,
) -> None:
    """写入一条调用记录。失败只 log，不抛。"""
    try:
        db = await _get_db()
        u = usage or {}
        provider, _, model_id = (spec or "").partition(":")
        if not provider:
            provider = "_unknown_"
        if not model_id:
            model_id = "_unknown_"
        async with measured_write_lock(_write_lock, "usage.record_usage"):
            try:
                await db.execute(
                    "INSERT INTO llm_usage "
                    "(ts, session_id, role, spec, provider, model_id, caller, "
                    " prompt_tokens, completion_tokens, total_tokens, "
                    " latency_ms, success, fallback_attempt, usage_known) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(),
                        session_id or "_unknown_",
                        role,
                        spec or "_unknown_",
                        provider,
                        model_id,
                        caller,
                        int(u.get("prompt_tokens", 0) or 0),
                        int(u.get("completion_tokens", 0) or 0),
                        int(u.get("total_tokens", 0) or 0),
                        int(latency_ms),
                        1 if success else 0,
                        int(fallback_attempt),
                        int(all(u.get(key) is not None for key in (
                            "prompt_tokens", "completion_tokens", "total_tokens",
                        ))),
                    ),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
    except Exception as e:
        general_logger.bind(category="llm").warning(f"[usage_db] 写入失败: {e}")


def _since_iso(window: str) -> Optional[str]:
    """window: '24h' / '7d' / 'all'"""
    if window == "all":
        return None
    now = datetime.now()
    if window == "24h":
        return (now - timedelta(hours=24)).isoformat()
    if window == "7d":
        return (now - timedelta(days=7)).isoformat()
    if window == "30d":
        return (now - timedelta(days=30)).isoformat()
    return None


async def _query(where_clauses: list[str], params: list, since_iso: Optional[str]) -> Dict[str, Any]:
    """聚合查询：返回 totals / by_provider / by_model / by_spec / by_role。"""
    db = await _get_db()
    if since_iso:
        where_clauses = where_clauses + ["ts >= ?"]
        params = params + [since_iso]
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # totals
    cur = await db.execute(
        f"SELECT COUNT(*), "
        f"       SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), "
        f"       SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), "
        f"       COALESCE(SUM(prompt_tokens), 0), "
        f"       COALESCE(SUM(completion_tokens), 0), "
        f"       COALESCE(SUM(total_tokens), 0), "
        f"       COALESCE(SUM(CASE WHEN fallback_attempt>0 THEN 1 ELSE 0 END), 0), "
        f"       COALESCE(AVG(latency_ms), 0), "
        f"       COALESCE(MAX(latency_ms), 0), "
        f"       COALESCE(SUM(CASE WHEN usage_known=0 THEN 1 ELSE 0 END), 0) "
        f"FROM llm_usage {where_sql}",
        params,
    )
    row = await cur.fetchone()
    await cur.close()
    totals = {
        "calls": int(row[0] or 0),
        "success": int(row[1] or 0),
        "failures": int(row[2] or 0),
        "prompt_tokens": int(row[3] or 0),
        "completion_tokens": int(row[4] or 0),
        "total_tokens": int(row[5] or 0),
        "fallback_calls": int(row[6] or 0),
        "avg_latency_ms": int(row[7] or 0),
        "max_latency_ms": int(row[8] or 0),
        "unknown_usage_calls": int(row[9] or 0),
    }

    async def _group(col: str, key_name: str) -> list:
        cur = await db.execute(
            f"SELECT {col}, COUNT(*) as n, "
            f"       COALESCE(SUM(prompt_tokens), 0), "
            f"       COALESCE(SUM(completion_tokens), 0), "
            f"       COALESCE(SUM(total_tokens), 0), "
            f"       COALESCE(AVG(latency_ms), 0), "
            f"       SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), "
            f"       SUM(CASE WHEN usage_known=0 THEN 1 ELSE 0 END) "
            f"FROM llm_usage {where_sql} "
            f"GROUP BY {col} ORDER BY COALESCE(SUM(total_tokens), 0) DESC, n DESC",
            params,
        )
        rows = await cur.fetchall()
        await cur.close()
        return [
            {
                key_name: r[0],
                "calls": int(r[1] or 0),
                "prompt_tokens": int(r[2] or 0),
                "completion_tokens": int(r[3] or 0),
                "total_tokens": int(r[4] or 0),
                "avg_latency_ms": int(r[5] or 0),
                "failures": int(r[6] or 0),
                "unknown_usage_calls": int(r[7] or 0),
            }
            for r in rows
        ]

    by_provider = await _group("provider", "provider")
    by_model = await _group("model_id", "model_id")
    by_spec = await _group("spec", "spec")
    by_role = await _group("role", "role")

    return {
        "totals": totals,
        "by_provider": by_provider,
        "by_model": by_model,
        "by_spec": by_spec,
        "by_role": by_role,
    }


async def query_session(
    session_id: str, window: str = "24h", *, start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[str, Any]:
    """查询滚动窗口，或明确指定的 [start, end) 时间范围。"""
    if start is not None or end is not None:
        if start is None or end is None or start >= end:
            raise ValueError("Usage range requires start < end")
        return await _query(
            ["session_id = ?", "ts >= ?", "ts < ?"],
            [session_id, start.isoformat(), end.isoformat()], None,
        )
    return await _query(["session_id = ?"], [session_id], _since_iso(window))


async def query_global(window: str = "24h") -> Dict[str, Any]:
    """查询全局所有 session 的统计。"""
    return await _query([], [], _since_iso(window))
