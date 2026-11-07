"""Explicit database targets for staging backfill and aggregate health evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def snapshot(path, target_version=None):
    with closing(sqlite3.connect(path, timeout=10)) as db:
        active = db.execute("SELECT value FROM kv_meta WHERE key='memory_active_embedding'").fetchone()
        active_version = active[0] if active else ""
        target_version = target_version if target_version is not None else active_version
        jobs = db.execute("SELECT count(*),sum(attempts>0),sum(attempts>=8) FROM memory_index_jobs").fetchone()
        items = db.execute(
            "SELECT count(*),sum(vector_version=? AND vector_version<>'' AND vector_hash=content_hash),sum(vector_error IS NOT NULL),sum(vector_attempts>=8) FROM memory_items",
            (target_version,),
        ).fetchone()
        return {
            "projection_jobs": jobs[0],
            "projection_errors": jobs[1] or 0,
            "items": items[0],
            "vectors_ready": items[1] or 0,
            "embedding_version": target_version,
            "active_embedding_version": active_version,
            "embedding_errors": items[2] or 0,
            "projection_failed": jobs[2] or 0,
            "embedding_failed": items[3] or 0,
            "embedding_state": dict(
                db.execute(
                    "SELECT key,value FROM kv_meta WHERE key LIKE 'memory_embedding_pause:%' OR key LIKE 'memory_embedding_usage:%'"
                )
            ),
        }


def import_index(database, source):
    """Import disposable projections, never canonical rows or deletion generations.

    The target schema must be installed and its index empty. Call while the bot
    is stopped. One transaction excludes changed/deleted source projections and
    enqueues every new/changed canonical row for catch-up.
    """
    from src.plugins.llm_chat.memory_index import BLOCK_SIZE, SOURCE_TABLES, connect

    fields = {
        "messages": "session_id,role,name,sender_id,content,timestamp,is_thinking",
        "summaries": "session_id,content,created_at,provenance_json",
        "important_info": "session_id,user_id,info_type,content,created_at,provenance_json",
    }
    with connect(database, vectors=True) as db:
        db.execute("ATTACH DATABASE ? AS staged", (str(source.resolve()),))
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM memory_items LIMIT 1").fetchone():
            raise ValueError("Target index is not empty; refusing to replace existing projections")
        before = {
            table: db.execute(f"SELECT count(*),max(id) FROM {table}").fetchone()[:] for table in SOURCE_TABLES.values()
        }
        db.execute("CREATE TEMP TABLE rejected_items(id INTEGER PRIMARY KEY)")
        db.execute("""INSERT OR IGNORE INTO rejected_items SELECT m.id FROM staged.memory_items m
            LEFT JOIN kv_meta k ON k.key='memory_generation:'||m.session_id
            WHERE m.generation<>COALESCE(CAST(k.value AS INTEGER),0)""")
        for kind, table in SOURCE_TABLES.items():
            columns = fields[kind].split(",")
            comparison = (
                "(" + ",".join("p." + c for c in columns) + ") IS NOT (" + ",".join("s." + c for c in columns) + ")"
            )
            db.execute(
                f"""INSERT OR IGNORE INTO rejected_items SELECT l.item_id FROM staged.memory_links l
                LEFT JOIN {table} p ON p.id=l.source_id LEFT JOIN staged.{table} s ON s.id=l.source_id
                WHERE l.source_kind=? AND l.relation='derived_from' AND (p.id IS NULL OR s.id IS NULL OR {comparison})""",
                (kind,),
            )
            source_id = f"(p.id/{BLOCK_SIZE})*{BLOCK_SIZE}" if kind == "messages" else "p.id"
            db.execute(
                f"""INSERT INTO memory_index_jobs(session_id,kind,source_id,generation,priority)
                SELECT DISTINCT p.session_id,?,{source_id},COALESCE(CAST(k.value AS INTEGER),0),0
                FROM {table} p LEFT JOIN staged.{table} s ON s.id=p.id
                LEFT JOIN kv_meta k ON k.key='memory_generation:'||p.session_id
                WHERE s.id IS NULL OR {comparison}
                ON CONFLICT(session_id,kind,source_id,generation) DO UPDATE SET revision=revision+1,priority=0""",
                (kind,),
            )
        db.execute("""INSERT INTO memory_index_jobs(session_id,kind,source_id,generation,priority)
            SELECT DISTINCT m.session_id,m.kind,m.source_id,COALESCE(CAST(k.value AS INTEGER),0),0
            FROM staged.memory_items m LEFT JOIN kv_meta k ON k.key='memory_generation:'||m.session_id
            WHERE m.id IN (SELECT id FROM rejected_items)
            ON CONFLICT(session_id,kind,source_id,generation) DO NOTHING""")
        db.execute(
            "INSERT INTO memory_items SELECT * FROM staged.memory_items WHERE id NOT IN (SELECT id FROM rejected_items)"
        )
        db.execute(
            "INSERT INTO memory_links SELECT * FROM staged.memory_links WHERE item_id IN (SELECT id FROM memory_items)"
        )
        db.execute("""INSERT INTO memory_fts(rowid,text,session_id)
            SELECT rowid,text,session_id FROM staged.memory_fts WHERE rowid IN (SELECT id FROM memory_items)""")
        db.execute("""INSERT INTO memory_index_jobs(session_id,kind,source_id,generation,revision,attempts,retry_at,priority,lease_until,error,created_at)
            SELECT j.session_id,j.kind,j.source_id,j.generation,j.revision,j.attempts,j.retry_at,10,0,j.error,j.created_at
            FROM staged.memory_index_jobs j LEFT JOIN kv_meta k ON k.key='memory_generation:'||j.session_id
            WHERE j.generation=COALESCE(CAST(k.value AS INTEGER),0)
            ON CONFLICT(session_id,kind,source_id,generation) DO NOTHING""")
        for name, sql in db.execute(
            "SELECT name,sql FROM staged.sqlite_master WHERE type='table' AND name GLOB 'memory_vec_*'"
        ).fetchall():
            if not re.fullmatch(r"memory_vec_[a-f0-9]{16}", name):
                continue
            settings = db.execute(
                "SELECT value FROM staged.kv_meta WHERE key=?", ("memory_embedding:" + name[11:],)
            ).fetchone()
            dimension = int(json.loads(settings[0])["dimension"])
            if not 1 <= dimension <= 4096:
                raise ValueError("Invalid staged dimension")
            db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(item_id INTEGER PRIMARY KEY,
                embedding float[{dimension}] distance_metric=cosine,session_id TEXT PARTITION KEY,+content_hash TEXT)""")
            db.execute(f"""INSERT INTO {name}(item_id,embedding,session_id,content_hash)
                SELECT v.item_id,v.embedding,v.session_id,v.content_hash FROM staged.{name} v
                JOIN memory_items m ON m.id=v.item_id AND m.content_hash=v.content_hash""")
        db.execute("""INSERT OR REPLACE INTO kv_meta SELECT key,value FROM staged.kv_meta
            WHERE key LIKE 'memory_embedding:%' OR key LIKE 'memory_backfill:%'
            OR key IN ('memory_active_embedding','memory_active_chunk')""")
        after = {
            table: db.execute(f"SELECT count(*),max(id) FROM {table}").fetchone()[:] for table in SOURCE_TABLES.values()
        }
        if after != before:
            raise RuntimeError("Canonical row counts changed during index import")
        report = {
            "canonical": after,
            "excluded_stale_items": db.execute("SELECT count(*) FROM rejected_items").fetchone()[0],
            "imported_items": db.execute("SELECT count(*) FROM memory_items").fetchone()[0],
            "catchup_jobs": db.execute("SELECT count(*) FROM memory_index_jobs").fetchone()[0],
        }
    return report


async def administer(args):
    import nonebot

    nonebot.init()
    from src.plugins.llm_chat.memory import MemoryManager
    from src.plugins.llm_chat.memory_index import connect

    manager = MemoryManager()
    manager._db_path = args.database
    manager._closing = True
    try:
        await manager._get_db()
        if args.action == "import-index":
            if not args.source or args.database.resolve() == args.source.resolve():
                raise ValueError("A distinct --source is required")
            print(json.dumps(await asyncio.to_thread(import_index, args.database, args.source)), flush=True)
        else:
            with connect(args.database) as db:
                db.execute(
                    "UPDATE memory_index_jobs SET attempts=0,retry_at=0,lease_until=0,error=NULL WHERE attempts>0"
                )
                db.execute(
                    "UPDATE memory_items SET vector_attempts=0,vector_retry_at=0,vector_error=NULL WHERE vector_error IS NOT NULL"
                )
                db.execute("DELETE FROM kv_meta WHERE key LIKE 'memory_embedding_pause:%'")
            print(json.dumps(snapshot(args.database)))
    finally:
        await manager.close()


async def backfill(args):
    import nonebot

    nonebot.init()
    from src.plugins.llm_chat.memory import MemoryManager
    from src.plugins.llm_chat.memory_index import MemoryIndex

    manager = MemoryManager()
    manager._db_path = args.database
    manager._closing = True
    await manager._get_db()
    index = MemoryIndex(args.database)
    await index.start()
    if args.lexical_only:
        index.vector_available = False
    start = time.monotonic()
    try:
        while True:
            state = await asyncio.to_thread(snapshot, args.database, index.version)
            print(json.dumps({"elapsed_seconds": round(time.monotonic() - start), **state}), flush=True)
            if state["projection_jobs"] == 0 and (
                args.lexical_only
                or (state["items"] == state["vectors_ready"] and state["active_embedding_version"] == index.version)
            ):
                break
            if args.seconds and time.monotonic() - start >= args.seconds:
                break
            await asyncio.sleep(30)
    finally:
        await index.close()
        await manager.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["clone", "backfill", "status", "import-index", "retry-failed"])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()
    args.database = args.database.resolve()
    if os.name == "nt":
        production = Path(__file__).resolve().parents[1] / "data" / "history.db"
        targets = [args.database, *([args.source.resolve()] if args.source else [])]
        if production in targets:
            parser.error(
                "Access the production SQLite database inside the Linux bot container; Windows WAL access is unsafe"
            )
    os.environ["LLM_CHAT_DATA_DIR"] = str(args.database.parent)
    if args.action == "clone":
        if not args.source or args.database.exists():
            parser.error("clone requires --source and a new destination")
        args.database.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(
                sqlite3.connect(f"file:{args.source.resolve().as_posix()}?mode=ro", uri=True, timeout=30)
            ) as source,
            closing(sqlite3.connect(args.database)) as target,
        ):
            source.backup(target, pages=256, sleep=0.1)
        print(json.dumps({"cloned": str(args.database), "bytes": args.database.stat().st_size}))
    elif args.action == "status":
        print(json.dumps(snapshot(args.database)))
    elif args.action in {"import-index", "retry-failed"}:
        asyncio.run(administer(args))
    else:
        asyncio.run(backfill(args))


if __name__ == "__main__":
    main()
