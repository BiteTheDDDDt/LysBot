"""Rebuildable SQLite search projections; canonical writes enqueue via triggers.

Connections used here belong to the worker thread, never to the chat writer.
Embedding HTTP calls run outside SQLite transactions and are versioned by model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


from .logging_setup import memory_logger as log

CHUNK_VERSION = "windows-v1-8-800-2-128"
BLOCK_SIZE = 128
EMBEDDING_TIMEOUT = 20.0
EMBEDDING_BATCH_SIZE = 10
SOURCE_TABLES = {"messages": "messages_detail", "summaries": "summaries", "important_info": "important_info"}
MAX_ATTEMPTS = 8


class EmbeddingFailure(RuntimeError):
    def __init__(self, code, *, cooldown=0, permanent=False):
        self.code, self.cooldown, self.permanent = code, cooldown, permanent
        super().__init__(code)


class EmbeddingBatch(list):
    usage = None


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def tokens(text: str) -> list[str]:
    import jieba

    jieba.setLogLevel(logging.WARNING)
    words = list(jieba.cut_for_search(text.lower()))
    words.extend(re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower()))
    return list(dict.fromkeys(w.strip() for w in words if re.search(r"[\w\u3400-\u9fff]", w)))


SCHEMA = [
    """CREATE TABLE IF NOT EXISTS memory_changes (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, generation INTEGER NOT NULL,
        actor TEXT NOT NULL, fact_id INTEGER NOT NULL, status TEXT NOT NULL,
        before_json TEXT NOT NULL, after_json TEXT NOT NULL, reason TEXT NOT NULL,
        created_at REAL NOT NULL, changed_at REAL, epoch TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_memory_changes_scope ON memory_changes(session_id,generation,created_at)",
    """CREATE TABLE IF NOT EXISTS memory_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, generation INTEGER NOT NULL,
        kind TEXT NOT NULL, source_id INTEGER NOT NULL, part TEXT NOT NULL,
        content TEXT NOT NULL, content_hash TEXT NOT NULL, event_start TEXT, event_end TEXT,
        recorded_at TEXT, metadata_json TEXT NOT NULL, chunk_version TEXT NOT NULL,
        vector_version TEXT NOT NULL DEFAULT '', vector_hash TEXT NOT NULL DEFAULT '',
        vector_retry_at REAL NOT NULL DEFAULT 0, vector_attempts INTEGER NOT NULL DEFAULT 0,
        vector_error TEXT, UNIQUE(session_id,generation,kind,source_id,part,chunk_version))""",
    "CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(session_id,generation,kind,event_start)",
    "CREATE INDEX IF NOT EXISTS idx_memory_items_vector ON memory_items(vector_retry_at,vector_version,id)",
    """CREATE TABLE IF NOT EXISTS memory_links (
        item_id INTEGER NOT NULL, source_kind TEXT NOT NULL, source_id INTEGER NOT NULL,
        relation TEXT NOT NULL DEFAULT 'derived_from', PRIMARY KEY(item_id,source_kind,source_id,relation))""",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_kind,source_id)",
    """CREATE TABLE IF NOT EXISTS memory_index_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, kind TEXT NOT NULL,
        source_id INTEGER NOT NULL, generation INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
        attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0, priority INTEGER NOT NULL DEFAULT 0,
        lease_until REAL NOT NULL DEFAULT 0, error TEXT, created_at REAL NOT NULL DEFAULT (unixepoch()),
        UNIQUE(session_id,kind,source_id,generation))""",
    "CREATE INDEX IF NOT EXISTS idx_memory_jobs_due ON memory_index_jobs(retry_at,lease_until,id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_jobs_priority ON memory_index_jobs(priority,id)",
    """CREATE TABLE IF NOT EXISTS agent_runs (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, generation INTEGER NOT NULL,
        status TEXT NOT NULL, state_json TEXT NOT NULL, updated_at REAL NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_scope ON agent_runs(session_id,updated_at)",
]


async def install_schema(db) -> None:
    for sql in SCHEMA:
        await db.execute(sql)
    for table in ("summaries", "important_info"):
        async with db.execute(f"PRAGMA table_info({table})") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if "provenance_json" not in columns:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{{}}'")
    try:
        await db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(text, session_id UNINDEXED)")
    except sqlite3.OperationalError as error:
        log.warning(f"[memory_index] FTS unavailable: {error}")
    for kind, table in SOURCE_TABLES.items():
        for operation, row in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
            source = f"({row}.id / {BLOCK_SIZE}) * {BLOCK_SIZE}" if kind == "messages" else f"{row}.id"
            generation = f"COALESCE((SELECT CAST(value AS INTEGER) FROM kv_meta WHERE key='memory_generation:'||{row}.session_id),0)"
            await db.execute(f"DROP TRIGGER IF EXISTS search_{kind}_{operation.lower()}")
            await db.execute(f"""CREATE TRIGGER search_{kind}_{operation.lower()}
                AFTER {operation} ON {table} BEGIN
                INSERT INTO kv_meta(key,value) VALUES('memory_revision:'||{row}.session_id,'1')
                ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1;
                INSERT INTO memory_index_jobs(session_id,kind,source_id,generation)
                VALUES({row}.session_id,'{kind}',{source},{generation})
                ON CONFLICT(session_id,kind,source_id,generation) DO UPDATE SET
                    revision=revision+1,retry_at=0,attempts=0,error=NULL,priority=0;
                END""")


def generation(db, session_id: str) -> int:
    row = db.execute("SELECT value FROM kv_meta WHERE key=?", ("memory_generation:" + session_id,)).fetchone()
    return int(row[0]) if row else 0


def embedding_settings() -> dict:
    from .local_embedding import DIMENSION, FILES, MODEL, POLICY, REVISION

    return {
        "model": MODEL,
        "dimension": DIMENSION,
        "revision": REVISION,
        "normalization": "l2",
        "chunk_version": CHUNK_VERSION,
        "task_policy": POLICY,
        "artifacts": dict(FILES),
    }


def embedding_version(settings: dict) -> str:
    return digest(encode(settings))[:16]


async def embed(texts: list[str], settings: dict, client=None, *, query=False) -> list[bytes]:
    from .local_embedding import encode_local, validate_settings
    validate_settings(settings)
    vectors = await encode_local(texts, settings, query=query)
    return pack_embeddings([{"index": i, "embedding": vector} for i, vector in enumerate(vectors)],
                           len(texts), settings, {})


def pack_embeddings(data, count, settings, usage):
    if len(data) != count or [item["index"] for item in data] != list(range(count)):
        raise ValueError("embedding_result_count_mismatch")
    result = EmbeddingBatch()
    result.usage = usage
    for item in data:
        vector = item["embedding"]
        if len(vector) != settings["dimension"] or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError("embedding_dimension_or_value_mismatch")
        norm = math.sqrt(sum(float(v) ** 2 for v in vector))
        if not norm:
            raise ValueError("embedding_zero_norm")
        result.append(struct.pack(f"<{len(vector)}f", *(float(v) / norm for v in vector)))
    return result


@contextmanager
def connect(path: Path, *, vectors=False, deadline=None, operation="memory_index"):
    started = time.monotonic()
    db = sqlite3.connect(path, timeout=5)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA synchronous=NORMAL")
        if deadline is not None:
            db.set_progress_handler(lambda: time.monotonic() > deadline, 10000)
        if vectors:
            import sqlite_vec
            db.enable_load_extension(True)
            try:
                sqlite_vec.load(db)
            finally:
                db.enable_load_extension(False)
        with db:
            yield db
    except sqlite3.OperationalError as error:
        log.warning(f"[sqlite_connection] operation={operation} elapsed={time.monotonic()-started:.3f}s "
                    f"code={getattr(error, 'sqlite_errorname', 'unknown')} error={error}")
        raise
    finally:
        elapsed = time.monotonic() - started
        if db.total_changes and elapsed >= 1:
            log.warning(f"[sqlite_write] operation={operation} elapsed={elapsed:.3f}s changes={db.total_changes}")
        db.close()


def _message_chunks(rows: list[dict]) -> list[dict]:
    parts = []
    for row in rows:
        content = row["content"] or ""
        for offset in range(0, len(content), 650):
            parts.append(
                {**row, "content": content[offset : offset + 650], "offset": offset, "original_hash": digest(content)}
            )
    chunks, start = [], 0
    while start < len(parts):
        window, length = [], 0
        for row in parts[start : start + 8]:
            if window:
                try:
                    gap = (
                        datetime.fromisoformat(row["timestamp"]) - datetime.fromisoformat(window[-1]["timestamp"])
                    ).total_seconds()
                except (ValueError, TypeError):
                    gap = 0
                if gap > 600 or length + len(row["content"]) > 800:
                    break
            window.append(row)
            length += len(row["content"])
        text = "\n".join(
            f"[{r['timestamp']}] {r['role']} {r['name'] or ''} (id={r['sender_id']}): {r['content']}" for r in window
        )
        chunks.append(
            {
                "part": f"{window[0]['id']}:{window[0]['offset']}",
                "content": text,
                "event_start": window[0]["timestamp"],
                "event_end": window[-1]["timestamp"],
                "recorded_at": window[-1]["timestamp"],
                "source_ids": list(dict.fromkeys(r["id"] for r in window)),
                "metadata": {
                    "authors": list(dict.fromkeys(str(r["sender_id"]) for r in window if r["sender_id"] is not None)),
                    "source_hashes": {str(r["id"]): r["original_hash"] for r in window},
                    "roles": list(dict.fromkeys(r["role"] for r in window)),
                    "provenance": "original",
                    "names": list(dict.fromkeys(r["name"] for r in window if r["name"])),
                },
            }
        )
        if start + len(window) >= len(parts):
            break
        start += max(1, len(window) - 2)
    return chunks


def _vector_tables(db) -> list[str]:
    """Names of materialized vector tables. Names are hashes, never user input."""
    return [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'memory_vec_*'")
        if re.fullmatch(r"memory_vec_[a-f0-9]{16}", row[0])
    ]


class MemoryIndex:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.task = None
        self.settings = embedding_settings()
        self.version = embedding_version(self.settings)
        self.vector_available = False
        self.last_error = None
        self._closing = False
        self.client = None
        self.embedding_pause_until = 0
        self._last_cache_sweep = 0
        self._threads = set()

    async def _thread(self, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self._threads.add(task)
        task.add_done_callback(self._threads.discard)
        return await asyncio.shield(task)

    async def start(self):
        await self._thread(self._bootstrap)
        try:
            from .local_embedding import model_directory, validate_settings, verify_files
            validate_settings(self.settings)
            await self._thread(verify_files, model_directory())
            await self._thread(self._prepare_vectors)
            self.vector_available = True
        except Exception as error:  # noqa: BLE001 -- extension/provider setup must preserve lexical availability.
            self.last_error = str(error)
            log.warning(f"[memory_index] vector disabled, lexical search remains available: {error}")
        self.task = asyncio.create_task(self._worker(), name="memory-index")

    async def close(self):
        self._closing = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self._threads:
            await asyncio.gather(*tuple(self._threads), return_exceptions=True)
        if self.client:
            await self.client.aclose()

    def _bootstrap(self):
        with connect(self.path, operation="memory_index._bootstrap") as db:
            key = "memory_backfill:" + CHUNK_VERSION
            if not db.execute("SELECT 1 FROM kv_meta WHERE key=?", (key,)).fetchone():
                for kind, table in SOURCE_TABLES.items():
                    source = f"(s.id / {BLOCK_SIZE}) * {BLOCK_SIZE}" if kind == "messages" else "s.id"
                    db.execute(
                        f"""INSERT OR IGNORE INTO memory_index_jobs(session_id,kind,source_id,generation,priority)
                        SELECT DISTINCT s.session_id,?,{source},COALESCE(CAST(k.value AS INTEGER),0),10
                        FROM {table} s LEFT JOIN kv_meta k ON k.key='memory_generation:'||s.session_id""",
                        (kind,),
                    )
                db.execute("INSERT INTO kv_meta(key,value) VALUES(?,?)", (key, str(time.time())))
            db.execute("UPDATE memory_index_jobs SET lease_until=0")
            pause = db.execute(
                "SELECT value FROM kv_meta WHERE key=?", ("memory_embedding_pause:" + self.version,)
            ).fetchone()
            self.embedding_pause_until = float(pause[0]) if pause else 0

    def _prepare_vectors(self):
        with connect(self.path, vectors=True) as db:
            known = db.execute("SELECT 1 FROM kv_meta WHERE key=?", ("memory_embedding:" + self.version,)).fetchone()
            db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec_{self.version} USING vec0(
                item_id INTEGER PRIMARY KEY, embedding float[{int(self.settings["dimension"])}] distance_metric=cosine,
                session_id TEXT PARTITION KEY, +content_hash TEXT)""")
            db.execute(
                "INSERT OR REPLACE INTO kv_meta(key,value) VALUES(?,?)",
                ("memory_embedding:" + self.version, encode(self.settings)),
            )
            # First installation may serve partial vectors with explicit coverage.
            db.execute("INSERT OR IGNORE INTO kv_meta(key,value) VALUES('memory_active_embedding',?)", (self.version,))
            db.execute("INSERT OR IGNORE INTO kv_meta(key,value) VALUES('memory_active_chunk',?)", (CHUNK_VERSION,))
            if not known:
                # Failures belong to the previous backend/version, not the new one.
                db.execute(
                    "UPDATE memory_items SET vector_attempts=0,vector_retry_at=0,vector_error=NULL WHERE vector_version<>?",
                    (self.version,),
                )
            self._retire_vector_tables(db)

    def _retire_vector_tables(self, db) -> None:
        """Drop vector tables that are neither current nor still serving searches.

        A version switch used to create a new table and leave every older one in
        place forever; items are re-embedded lazily, so the old data is redundant.
        The active version's table is kept for one start so a failed rollout can
        roll back without paying for a full re-embed.
        """
        keep = {f"memory_vec_{self.version}"}
        active = db.execute("SELECT value FROM kv_meta WHERE key='memory_active_embedding'").fetchone()
        if active and re.fullmatch(r"[a-f0-9]{16}", str(active[0])):
            keep.add(f"memory_vec_{active[0]}")
        for table in _vector_tables(db):
            if table in keep:
                continue
            db.execute(f"DROP TABLE {table}")
            log.info(f"[memory_index] dropped stale vector table {table}")

    def _project_one(self) -> bool:
        with connect(self.path, operation="memory_index._project_one") as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute(
                """SELECT * FROM memory_index_jobs INDEXED BY idx_memory_jobs_priority WHERE attempts<8 AND retry_at<=? AND lease_until<=?
                ORDER BY priority,id LIMIT 1""",
                (time.time(), time.time()),
            ).fetchone()
            if not job:
                return False
            job = dict(job)
            db.execute("UPDATE memory_index_jobs SET lease_until=? WHERE id=?", (time.time() + 60, job["id"]))
        try:
            with connect(self.path, operation="memory_index._project_one") as db:
                table = SOURCE_TABLES[job["kind"]]
                if job["kind"] == "messages":
                    rows = [
                        dict(r)
                        for r in db.execute(
                            """SELECT * FROM messages_detail
                        WHERE session_id=? AND id>=? AND id<? AND COALESCE(is_thinking,0)=0 ORDER BY id""",
                            (job["session_id"], job["source_id"], job["source_id"] + BLOCK_SIZE),
                        )
                    ]
                    items = _message_chunks(rows)
                else:
                    row = db.execute(
                        f"SELECT * FROM {table} WHERE session_id=? AND id=?", (job["session_id"], job["source_id"])
                    ).fetchone()
                    items = []
                    if row:
                        metadata = json.loads(row["provenance_json"] or "{}")
                        metadata.setdefault("provenance", "legacy_unlinked")
                        if job["kind"] == "important_info":
                            metadata.update(subjects=[str(row["user_id"])], info_type=row["info_type"])
                        for start in range(0, len(row["content"]), 800):
                            items.append(
                                {
                                    "part": str(start),
                                    "content": row["content"][start : start + 800],
                                    "event_start": metadata.get("event_start"),
                                    "event_end": metadata.get("event_end"),
                                    "recorded_at": row["created_at"],
                                    "metadata": metadata,
                                    "source_ids": [row["id"]],
                                }
                            )
            for item in items:
                item["fts"] = " ".join(tokens(item["content"]))
                item["hash"] = digest(encode([item["content"], item["metadata"], item["source_ids"]]))
            with connect(self.path, operation="memory_index._project_one") as db:
                db.execute("BEGIN IMMEDIATE")
                live = db.execute("SELECT revision FROM memory_index_jobs WHERE id=?", (job["id"],)).fetchone()
                if not live or live[0] != job["revision"] or generation(db, job["session_id"]) != job["generation"]:
                    db.execute("UPDATE memory_index_jobs SET lease_until=0 WHERE id=?", (job["id"],))
                    return True
                scope = (job["session_id"], job["generation"], job["kind"], job["source_id"], CHUNK_VERSION)
                old = {
                    r["part"]: dict(r)
                    for r in db.execute(
                        """SELECT id,part,content_hash FROM memory_items
                    WHERE session_id=? AND generation=? AND kind=? AND source_id=? AND chunk_version=?""",
                        scope,
                    )
                }
                fts = bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_fts'").fetchone())
                for item in items:
                    prior = old.pop(item["part"], None)
                    if prior and prior["content_hash"] == item["hash"]:
                        continue
                    db.execute(
                        """INSERT INTO memory_items(session_id,generation,kind,source_id,part,
                        content,content_hash,event_start,event_end,recorded_at,metadata_json,chunk_version,vector_retry_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id,generation,kind,source_id,part,chunk_version)
                        DO UPDATE SET content=excluded.content,content_hash=excluded.content_hash,event_start=excluded.event_start,
                        event_end=excluded.event_end,recorded_at=excluded.recorded_at,metadata_json=excluded.metadata_json,
                        vector_retry_at=excluded.vector_retry_at,vector_attempts=0""",
                        (
                            *scope[:4],
                            item["part"],
                            item["content"],
                            item["hash"],
                            item["event_start"],
                            item["event_end"],
                            item["recorded_at"],
                            encode(item["metadata"]),
                            CHUNK_VERSION,
                            -1 if job["priority"] == 0 else 0,
                        ),
                    )
                    item_id = db.execute(
                        """SELECT id FROM memory_items WHERE session_id=? AND generation=?
                        AND kind=? AND source_id=? AND chunk_version=? AND part=?""",
                        (*scope, item["part"]),
                    ).fetchone()[0]
                    if fts:
                        db.execute("DELETE FROM memory_fts WHERE rowid=?", (item_id,))
                        db.execute(
                            "INSERT INTO memory_fts(rowid,text,session_id) VALUES(?,?,?)",
                            (item_id, item["fts"], job["session_id"]),
                        )
                    db.execute("DELETE FROM memory_links WHERE item_id=?", (item_id,))
                    db.executemany(
                        "INSERT OR IGNORE INTO memory_links VALUES(?,?,?,'derived_from')",
                        [(item_id, job["kind"], source_id) for source_id in item["source_ids"]],
                    )
                    for source_id in item["metadata"].get("source_ids", []):
                        db.execute(
                            "INSERT OR IGNORE INTO memory_links VALUES(?,'messages',?,'derived_from')",
                            (item_id, int(source_id)),
                        )
                    for source_id in item["metadata"].get("supersedes", []):
                        db.execute(
                            "INSERT OR IGNORE INTO memory_links VALUES(?,'important_info',?,'supersedes')",
                            (item_id, int(source_id)),
                        )
                for prior in old.values():
                    if fts:
                        db.execute("DELETE FROM memory_fts WHERE rowid=?", (prior["id"],))
                    db.execute("DELETE FROM memory_links WHERE item_id=?", (prior["id"],))
                    db.execute("DELETE FROM memory_items WHERE id=?", (prior["id"],))
                if old:
                    db.execute(
                        "INSERT OR REPLACE INTO kv_meta(key,value) VALUES(?,'pending')",
                        ("memory_purge:" + job["session_id"],),
                    )
                db.execute("DELETE FROM memory_index_jobs WHERE id=? AND revision=?", (job["id"], job["revision"]))
            return True
        except Exception as error:
            with connect(self.path, operation="memory_index._project_one") as db:
                db.execute(
                    """UPDATE memory_index_jobs SET attempts=attempts+1,lease_until=0,retry_at=?,error=?
                    WHERE id=? AND revision=?""",
                    (
                        time.time() + min(3600, 30 * 2 ** min(job["attempts"], 7)),
                        str(error)[:500],
                        job["id"],
                        job["revision"],
                    ),
                )
            raise

    def _pending_vectors(self) -> list[dict]:
        with connect(self.path, operation="memory_index._pending_vectors") as db:
            return [
                dict(r)
                for r in db.execute(
                    """SELECT * FROM memory_items WHERE chunk_version=? AND vector_attempts<8 AND vector_retry_at<=?
                AND (vector_version<>? OR vector_hash<>content_hash) ORDER BY vector_retry_at,id LIMIT ?""",
                    (CHUNK_VERSION, time.time(), self.version, max(1, min(10, EMBEDDING_BATCH_SIZE)) * 3),
                )
            ]

    def _commit_vectors(self, rows: list[dict], vectors: list[bytes]):
        with connect(self.path, vectors=True) as db:
            db.execute("BEGIN IMMEDIATE")
            for row, vector in zip(rows, vectors):
                current = db.execute(
                    "SELECT content_hash,generation FROM memory_items WHERE id=?", (row["id"],)
                ).fetchone()
                if (
                    not current
                    or current[0] != row["content_hash"]
                    or generation(db, row["session_id"]) != row["generation"]
                ):
                    continue
                if db.execute(
                    "SELECT 1 FROM memory_index_jobs WHERE session_id=? AND kind=? AND source_id=? AND generation=? AND priority=0",
                    (row["session_id"], row["kind"], row["source_id"], row["generation"]),
                ).fetchone():
                    continue
                db.execute(f"DELETE FROM memory_vec_{self.version} WHERE item_id=?", (row["id"],))
                db.execute(
                    f"INSERT INTO memory_vec_{self.version}(item_id,embedding,session_id,content_hash) VALUES(?,?,?,?)",
                    (row["id"], vector, row["session_id"], row["content_hash"]),
                )
                db.execute(
                    "UPDATE memory_items SET vector_version=?,vector_hash=?,vector_retry_at=0,vector_attempts=0,vector_error=NULL WHERE id=?",
                    (self.version, row["content_hash"], row["id"]),
                )
            usage = getattr(vectors, "usage", None) or {}
            for metric, value in {
                "requests": 1,
                "texts": len(vectors),
                "tokens": int(usage.get("total_tokens", 0) or 0),
            }.items():
                db.execute(
                    """INSERT INTO kv_meta(key,value) VALUES(?,?) ON CONFLICT(key)
                    DO UPDATE SET value=CAST(value AS INTEGER)+CAST(excluded.value AS INTEGER)""",
                    (f"memory_embedding_usage:{self.version}:{metric}", str(value)),
                )

    def _defer_vectors(self, rows, error):
        with connect(self.path, operation="memory_index._defer_vectors") as db:
            cooldown = getattr(error, "cooldown", 0)
            if cooldown:
                self.embedding_pause_until = time.time() + cooldown
                db.execute(
                    "INSERT OR REPLACE INTO kv_meta(key,value) VALUES(?,?)",
                    ("memory_embedding_pause:" + self.version, str(self.embedding_pause_until)),
                )
            db.executemany(
                """UPDATE memory_items SET vector_attempts=?,vector_retry_at=?,vector_error=?
                WHERE id=? AND content_hash=?""",
                [
                    (
                        MAX_ATTEMPTS
                        if getattr(error, "permanent", False)
                        else row["vector_attempts"] + (0 if cooldown else 1),
                        time.time() + max(cooldown, min(3600, 30 * 2 ** min(row["vector_attempts"], 7))),
                        str(error)[:500],
                        row["id"],
                        row["content_hash"],
                    )
                    for row in rows
                ],
            )

    def _activate_if_complete(self):
        with connect(self.path, operation="memory_index._activate_if_complete") as db:
            db.execute("BEGIN IMMEDIATE")
            incomplete = db.execute(
                "SELECT 1 FROM memory_items WHERE chunk_version=? AND (vector_version<>? OR vector_hash<>content_hash) LIMIT 1",
                (CHUNK_VERSION, self.version),
            ).fetchone()
            jobs = db.execute("SELECT 1 FROM memory_index_jobs LIMIT 1").fetchone()
            if not incomplete and not jobs:
                db.execute(
                    "INSERT OR REPLACE INTO kv_meta(key,value) VALUES('memory_active_embedding',?)", (self.version,)
                )
                db.execute(
                    "INSERT OR REPLACE INTO kv_meta(key,value) VALUES('memory_active_chunk',?)", (CHUNK_VERSION,)
                )

    def _purge(self):
        from .web_search import expire_cache, purge_cache

        with connect(self.path, operation="memory_index._purge") as db:
            for row in db.execute("SELECT key FROM kv_meta WHERE key LIKE 'memory_purge:%' AND value='pending'"):
                sid = row[0].split(":", 1)[1]
                purge_cache(self.path, sid, generation(db, sid))
        if time.time() - self._last_cache_sweep > 300:
            expire_cache(self.path)
            self._last_cache_sweep = time.time()
        if not self.vector_available:
            return
        with connect(self.path, vectors=True) as db:
            pending = db.execute(
                "SELECT key FROM kv_meta WHERE key LIKE 'memory_purge:%' AND value='pending'"
            ).fetchall()
            tables = _vector_tables(db)
            for row in pending:
                sid = row[0].split(":", 1)[1]
                for table in tables:
                    # Keep only fresh items that arrived after the deletion.
                    ids = db.execute(f"SELECT item_id FROM {table} WHERE session_id=?", (sid,)).fetchall()
                    for item in ids:
                        if not db.execute(
                            "SELECT 1 FROM memory_items WHERE id=? AND session_id=?", (item[0], sid)
                        ).fetchone():
                            db.execute(f"DELETE FROM {table} WHERE item_id=?", (item[0],))
                db.execute("UPDATE kv_meta SET value='complete' WHERE key=?", (row[0],))

    async def _worker(self):
        while not self._closing:
            worked = False
            try:
                for _ in range(32):
                    if not await self._thread(self._project_one):
                        break
                    worked = True
                await self._thread(self._purge)
                if self.vector_available:
                    rows = (
                        await self._thread(self._pending_vectors) if time.time() >= self.embedding_pause_until else []
                    )
                    if rows:
                        batches = [rows[i : i + 10] for i in range(0, len(rows), 10)]
                        outcomes = await asyncio.gather(
                            *(embed([r["content"] for r in batch], self.settings, self.client) for batch in batches),
                            return_exceptions=True,
                        )
                        for batch, outcome in zip(batches, outcomes):
                            if isinstance(outcome, BaseException):
                                if isinstance(outcome, asyncio.CancelledError):
                                    raise outcome
                                self.last_error = str(outcome)
                                await self._thread(self._defer_vectors, batch, outcome)
                                log.warning(
                                    f"[memory_index] embedding deferred: {type(outcome).__name__} code={str(outcome)[:80]}"
                                )
                            else:
                                await self._thread(self._commit_vectors, batch, outcome)
                                worked = True
                                self.last_error = None
                    else:
                        await self._thread(self._activate_if_complete)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 -- persistent worker retries unexpected I/O and backend failures.
                self.last_error = str(error)
                log.warning(f"[memory_index] worker retry: {error}")
            await asyncio.sleep(0.05 if worked else 5)
