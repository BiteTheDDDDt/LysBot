"""Agent-written SELECT queries over connection-local, session-scoped views.

The SQLite authorizer, not a SQL keyword blacklist, enforces access. Private
view names are generated after receiving SQL and cannot occur in that SQL:
otherwise an attacker could name a CTE after a trusted view and spoof the
authorizer's source context. No persistent schema or data is modified.
"""

from __future__ import annotations

import asyncio
import json
import math
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from .retrieval import failure, result

SQL_TIMEOUT = 8.0
OUTPUT_CHARS = 24000
FUNCTIONS = frozenset(
    [
        "abs",
        "avg",
        "ceil",
        "ceiling",
        "char",
        "coalesce",
        "concat",
        "concat_ws",
        "count",
        "cume_dist",
        "current_date",
        "current_time",
        "current_timestamp",
        "date",
        "datetime",
        "dense_rank",
        "first_value",
        "floor",
        "glob",
        "group_concat",
        "hex",
        "ifnull",
        "iif",
        "instr",
        "json",
        "json_array",
        "json_array_length",
        "json_each",
        "json_error_position",
        "json_extract",
        "json_group_array",
        "json_group_object",
        "json_insert",
        "json_object",
        "json_patch",
        "json_quote",
        "json_remove",
        "json_replace",
        "json_set",
        "json_tree",
        "json_type",
        "json_valid",
        "julianday",
        "lag",
        "last_value",
        "lead",
        "length",
        "like",
        "likelihood",
        "likely",
        "ln",
        "log",
        "log10",
        "log2",
        "lower",
        "ltrim",
        "max",
        "min",
        "mod",
        "nth_value",
        "ntile",
        "nullif",
        "octet_length",
        "percent_rank",
        "pow",
        "power",
        "quote",
        "radians",
        "random",
        "rank",
        "replace",
        "round",
        "row_number",
        "rtrim",
        "sign",
        "sqrt",
        "strftime",
        "substr",
        "substring",
        "sum",
        "time",
        "timediff",
        "total",
        "trim",
        "typeof",
        "unicode",
        "unixepoch",
        "unlikely",
        "upper",
        "->",
        "->>",
    ]
)

SCHEMA = {
    "messages": {
        "columns": "id INTEGER, source_id TEXT, sender_id TEXT, name TEXT, role TEXT, content TEXT, timestamp TEXT",
        "meaning": "当前会话原始消息；已排除内部思考。timestamp 为北京时间 ISO 文本。",
    },
    "summaries": {
        "columns": "id INTEGER, source_id TEXT, content TEXT, time_range TEXT, created_at TEXT, provenance_json TEXT",
        "meaning": "当前会话摘要，只作线索；provenance_json 可用 json_extract 读取来源和事件时间。",
    },
    "facts": {
        "columns": "id INTEGER, source_id TEXT, user_id TEXT, info_type TEXT, content TEXT, created_at TEXT, provenance_json TEXT",
        "meaning": "当前会话提取的历史断言；user_id 是被描述者，不是原话作者。计划、转述、冲突须读原文。",
    },
}


def describe():
    return result(
        schema=SCHEMA,
        dialect="SQLite",
        capabilities=[
            "SELECT",
            "JOIN",
            "CTE",
            "recursive CTE",
            "subquery",
            "UNION",
            "window functions",
            "JSON",
            "custom aggregation",
        ],
        examples=[
            "SELECT sender_id, count(*) AS n, count(DISTINCT date(timestamp)) AS active_days FROM messages GROUP BY sender_id ORDER BY n DESC",
            "WITH ranked AS (SELECT source_id, sender_id, content, timestamp, row_number() OVER (PARTITION BY sender_id ORDER BY timestamp DESC,id DESC) AS rn FROM messages WHERE content LIKE :topic) SELECT * FROM ranked WHERE rn<=2",
            "SELECT m.source_id, m.content, f.source_id AS fact_source FROM messages m JOIN facts f ON m.sender_id=f.user_id WHERE f.info_type=:kind AND m.timestamp>=:since LIMIT 20",
        ],
        limits={"sql_seconds": SQL_TIMEOUT, "max_rows": 100, "output_chars": OUTPUT_CHARS},
        guidance="SQL 只访问上述当前会话视图。语义近义查询先用 recall_history，再按 id 或 source_id 精查。SQL 结果不等于原文核验，具体断言用 source_id 调 read_history。长结果加 WHERE/ORDER BY/LIMIT，翻页需自行写稳定排序及条件。",
    )


def _open(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1, cached_statements=0)
    db.execute("PRAGMA trusted_schema=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-2048")
    if hasattr(db, "setconfig"):
        db.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
    return db


def _scope(db, sid):
    rows = dict(
        db.execute(
            "SELECT key,value FROM kv_meta WHERE key IN (?,?)", ("memory_generation:" + sid, "memory_revision:" + sid)
        )
    )
    return int(rows.get("memory_generation:" + sid, 0)), rows.get("memory_revision:" + sid, "")


def _views(db, sid, gen, sql):
    from .memory_changes import active_sql
    while True:
        prefix = "scope_" + secrets.token_hex(16)
        if prefix not in sql.lower():
            break
    quoted_sid = "'" + sid.replace("'", "''") + "'"
    definitions = {
        "messages": (
            "messages_detail",
            f"id, 'h:{gen}:messages:'||id AS source_id, CAST(sender_id AS TEXT) AS sender_id, name, role, content, timestamp",
            " AND COALESCE(is_thinking,0)=0",
        ),
        "summaries": (
            "summaries",
            f"id, 'h:{gen}:summaries:'||id AS source_id, content, time_range, created_at, provenance_json",
            " AND " + active_sql("summaries"),
        ),
        "facts": (
            "important_info",
            f"id, 'h:{gen}:important_info:'||id AS source_id, CAST(user_id AS TEXT) AS user_id, info_type, content, created_at, provenance_json",
            " AND " + active_sql("important_info"),
        ),
    }
    private = {}
    for public, (table, columns, extra) in definitions.items():
        name = prefix + "_" + public
        private[name] = table
        db.execute(
            f"CREATE TEMP VIEW {name} AS SELECT {columns} FROM main.{table} WHERE session_id={quoted_sid}{extra}"
        )
        db.execute(f"CREATE TEMP VIEW {public} AS SELECT * FROM {name}")
    # Initialize only these known, read-only eponymous tables before authorization.
    for function in ("json_each", "json_tree"):
        if db.execute("SELECT 1 FROM main.sqlite_schema WHERE name=?", (function,)).fetchone():
            raise sqlite3.DatabaseError("JSON table name conflicts with a stored object")
        db.execute(f"SELECT value FROM {function}('[]')").fetchall()
    return private


def _authorize(private):
    def authorize(action, first, second, database, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if (second or "").lower() in FUNCTIONS else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            if database == "main" and private.get(source) == first:
                return sqlite3.SQLITE_OK
            if database == "temp" and first in {*SCHEMA, *private}:
                return sqlite3.SQLITE_OK
            if database in (None, "main") and first in {"json_each", "json_tree"}:
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def _limits(db):
    for category, value in (
        (sqlite3.SQLITE_LIMIT_LENGTH, 262144),
        (sqlite3.SQLITE_LIMIT_SQL_LENGTH, 20000),
        (sqlite3.SQLITE_LIMIT_COLUMN, 64),
        (sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50),
        (sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 20),
        (sqlite3.SQLITE_LIMIT_FUNCTION_ARG, 32),
        (sqlite3.SQLITE_LIMIT_ATTACHED, 0),
        (sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, 1000),
        (sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 200),
        (sqlite3.SQLITE_LIMIT_VDBE_OP, 100000),
    ):
        db.setlimit(category, value)


def _query(path, sid, sql, parameters, limit, cancelled):
    started = time.monotonic()
    db = _open(path)
    try:
        db.set_progress_handler(lambda: cancelled.is_set() or time.monotonic() - started >= SQL_TIMEOUT, 1000)
        db.execute("BEGIN")
        state = _scope(db, sid)
        private = _views(db, sid, state[0], sql)
        db.execute("PRAGMA query_only=ON")
        _limits(db)
        db.set_authorizer(_authorize(private))
        cursor = db.execute(sql, parameters)
        if cursor.description is None:
            return failure("select_required", invalid=True)
        columns = [column[0] for column in cursor.description]
        rows, truncated, warnings = [], False, []
        used = sum(len(name) for name in columns)
        for row in cursor:
            if len(rows) >= limit:
                truncated = True
                break
            values = []
            for value in row:
                if isinstance(value, bytes):
                    value = "[BLOB omitted]"
                    warnings.append("binary_result_omitted")
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                    warnings.append("nonfinite_result_omitted")
                if isinstance(value, str) and len(value) > 4000:
                    value = value[:4000]
                    warnings.append("cell_text_truncated_read_original_or_refine_sql")
                values.append(value)
            size = sum(len(str(value)) for value in values)
            if used + size > OUTPUT_CHARS:
                truncated = True
                warnings.append("output_budget_exhausted_refine_sql")
                break
            used += size
            rows.append(values)
    finally:
        db.close()
    # Do not return evidence invalidated by a concurrent deletion or rewrite.
    current = _open(path)
    try:
        if _scope(current, sid) != state:
            return failure("scope_changed_retry_query", retryable=True)
    finally:
        current.close()
    return result(
        "partial" if warnings or truncated else ("ok" if rows else "empty"),
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        warnings=list(dict.fromkeys(warnings)),
        elapsed_ms=round((time.monotonic() - started) * 1000),
        effective_scope={"session_id": sid, "generation": state[0]},
        execution="agent_sql_on_scoped_views",
        read_hint="回读请在 SELECT 中保留 source_id，并把该完整字符串交给 read_history.ids；数字 id 仅用于 SQL 内关联，不能直接回读。",
    )


async def query_history_sql(path, sid, sql="", parameters=None, limit=30):
    if not hasattr(sqlite3.Connection, "setlimit"):
        return failure("dynamic_sql_requires_python311")
    if not isinstance(sid, str) or not sid or sid == "_unknown_":
        return failure("missing_session", invalid=True)
    if not isinstance(sql, str) or len(sql.encode("utf-8")) > 20000 or type(limit) is not int or not 1 <= limit <= 100:
        return failure("invalid_sql_or_limit", invalid=True)
    if parameters is None:
        parameters = {}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except (ValueError, RecursionError):
            return failure("invalid_parameters_json", invalid=True)
    if not isinstance(parameters, (dict, list)) or len(parameters) > 200:
        return failure("parameters_must_be_object_or_array", invalid=True)
    if isinstance(parameters, dict) and any(not isinstance(key, str) or len(key) > 100 for key in parameters):
        return failure("invalid_parameter_name", invalid=True)
    for value in parameters.values() if isinstance(parameters, dict) else parameters:
        if type(value) not in (str, int, float, bool, type(None)) or (isinstance(value, str) and len(value) > 10000):
            return failure("invalid_parameter_value", invalid=True)
        if isinstance(value, float) and not math.isfinite(value):
            return failure("invalid_parameter_value", invalid=True)
    if not sql.strip():
        return describe()
    cancelled = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(_query, path, sid, sql, parameters, limit, cancelled))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        try:
            await asyncio.shield(task)
        except (sqlite3.Error, MemoryError):
            pass
        raise
    except (sqlite3.Error, OverflowError, MemoryError) as error:
        message = str(error)
        if "interrupted" in message:
            return failure("query_budget_exhausted_refine_sql", retryable=False)
        denied = "prohibited" in message or "not authorized" in message
        payload = failure("sql_access_denied" if denied else "invalid_sql", invalid=True)
        payload["error"]["message"] = (
            "只可 SELECT 当前会话的 messages、summaries、facts 视图。" if denied else message[:300]
        )
        return payload
