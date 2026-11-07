"""Read production readiness, index progress and business counters; record observations without modifying SQLite."""

import argparse
import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def observe(args):
    now = time.time()
    previous = json.loads(args.record.read_text(encoding="utf-8")) if args.record.exists() else {}
    if previous and previous["started_at"] != args.started_at:
        raise ValueError("Use a new observation record for a different deployment")
    health = json.loads(args.health.read_text(encoding="utf-8"))
    healthy = (
        all(health.get(key) for key in ("plugin_ready", "database_ok", "onebot_connected"))
        and 0 <= now - health["updated_at"] <= 120
    )
    since = datetime.fromtimestamp(args.started_at, ZoneInfo("Asia/Shanghai")).replace(tzinfo=None).isoformat()
    with sqlite3.connect(args.database.as_uri() + "?mode=ro", uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        active = db.execute("SELECT value FROM kv_meta WHERE key='memory_active_embedding'").fetchone()
        total, ready, errors = db.execute(
            "SELECT count(*),sum(vector_version=? AND vector_hash=content_hash),sum(vector_error IS NOT NULL) FROM memory_items",
            (args.version,),
        ).fetchone()
        jobs, failed_jobs = db.execute("SELECT count(*),sum(attempts>=8) FROM memory_index_jobs").fetchone()
        messages = db.execute("SELECT count(*),max(id) FROM messages_detail").fetchone()
        usages = db.execute(
            "SELECT role,success,count(*) FROM llm_usage WHERE ts>=? GROUP BY role,success", (since,)
        ).fetchall()
        latencies = [
            r[0]
            for r in db.execute(
                "SELECT latency_ms FROM llm_usage WHERE ts>=? AND role='chat' AND success=1 AND latency_ms IS NOT NULL ORDER BY latency_ms",
                (since,),
            )
        ]
        runs = db.execute("SELECT status,state_json FROM agent_runs WHERE updated_at>=?", (args.started_at,)).fetchall()
    actions = Counter()
    for _, raw in runs:
        for action in json.loads(raw).get("actions", {}).values():
            actions[action.get("status", "unknown")] += 1
    snapshot = {
        "at": now,
        "healthy": healthy,
        "health_error": health.get("error"),
        "target_version": args.version,
        "active_version": active[0] if active else "",
        "items": total,
        "ready": ready or 0,
        "remaining": total - (ready or 0),
        "embedding_errors": errors or 0,
        "projection_jobs": jobs,
        "projection_failed": failed_jobs or 0,
        "messages": messages[0],
        "message_high_watermark": messages[1],
        "model_calls_since_deploy": [{"role": role, "success": bool(ok), "count": count} for role, ok, count in usages],
        "runs_by_status": dict(Counter(status for status, _ in runs)),
        "actions_by_status": dict(actions),
        "chat_success_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
    }
    # Initial historical backfill must have activated. Small live deltas are expected in a running chat bot.
    stable = (
        healthy
        and snapshot["active_version"] == args.version
        and not errors
        and not failed_jobs
        and jobs <= 100
        and snapshot["remaining"] <= 500
        and not actions.get("unknown")
    )
    contiguous = now - previous.get("last_at", now) <= 900
    stable_since = (previous.get("stable_since") if contiguous else None) if stable else None
    stable_since = stable_since or (now if stable else None)
    snapshot["stable_seconds"] = int(now - stable_since) if stable_since else 0
    successful_chat = any(role == "chat" and ok and count for role, ok, count in usages)
    snapshot["ready_for_final_review"] = snapshot["stable_seconds"] >= 3600 and successful_chat
    record = {
        "started_at": args.started_at,
        "last_at": now,
        "stable_since": stable_since,
        "history": [*previous.get("history", []), snapshot][-144:],
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.record.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.record)
    print(json.dumps(snapshot, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--started-at", type=float, required=True)
    args = parser.parse_args()
    args.database = args.database.resolve()
    if os.name == "nt":
        raise ValueError("Observe production SQLite inside its Linux container")
    if args.record.resolve() == args.database or args.health.resolve() == args.record.resolve():
        raise ValueError("Observation output must not overwrite the database or health file")
    observe(args)


if __name__ == "__main__":
    main()
