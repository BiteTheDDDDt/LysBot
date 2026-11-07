"""Measure full-size search with synthetic vectors in an isolated database copy.

First clone canonical/index data with agentic_search_ops.py into an
agentic-performance directory. This measures latency, never semantic accuracy.
"""

import argparse
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    path = args.database.resolve()
    if "agentic-performance" not in path.parts or not path.is_file() or not 1 <= args.repeats <= 20:
        parser.error("Use an existing isolated agentic-performance copy and 1-20 repeats")
    os.environ["LLM_CHAT_DATA_DIR"] = str(path.parent)
    import nonebot

    nonebot.init()
    from src.plugins.llm_chat.memory_index import connect, embedding_settings, embedding_version
    from src.plugins.llm_chat.retrieval import HistorySearch

    settings = embedding_settings()
    settings["revision"] = "synthetic-latency-only"
    version = embedding_version(settings)
    dimension = int(settings["dimension"])
    vector = struct.pack(f"<{dimension}f", 1, *([0] * (dimension - 1)))
    with connect(path, vectors=True) as db:
        if args.initialize:
            db.execute(f"""CREATE VIRTUAL TABLE memory_vec_{version} USING vec0(item_id INTEGER PRIMARY KEY,
                embedding float[{dimension}] distance_metric=cosine,session_id TEXT PARTITION KEY,+content_hash TEXT)""")
            db.execute(
                f"""INSERT INTO memory_vec_{version}(item_id,embedding,session_id,content_hash)
                SELECT id,?,session_id,content_hash FROM memory_items""",
                (vector,),
            )
        sid, messages = db.execute(
            "SELECT session_id,count(*) n FROM messages_detail GROUP BY session_id ORDER BY n DESC LIMIT 1"
        ).fetchone()
        vectors = db.execute(f"SELECT count(*) FROM memory_vec_{version} WHERE session_id=?", (sid,)).fetchone()[0]
        author = str(
            db.execute(
                "SELECT sender_id,count(*) n FROM messages_detail WHERE session_id=? AND sender_id IS NOT NULL GROUP BY sender_id ORDER BY n DESC LIMIT 1",
                (sid,),
            ).fetchone()[0]
        )
    print(
        json.dumps({"synthetic_vectors": True, "largest_group_messages": messages, "largest_group_vectors": vectors}),
        flush=True,
    )
    search = HistorySearch(path)
    filters = {
        "sources": ["messages", "summaries", "important_info"],
        "author_ids": [],
        "subject_ids": [],
        "sender_names": [],
        "since": None,
        "until": None,
        "roles": [],
        "exact_phrase": "",
        "match_mode": "any",
        "sort": "relevance",
    }
    for name, changes in [("all", {}), ("author", {"author_ids": [author]}), ("recent", {"since": "2026-09-01"})]:
        elapsed = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            items, _truncated, warnings = search._retrieve(
                sid, 0, version, "", "semantic", {**filters, **changes}, vector, 0, 8
            )
            elapsed.append(round((time.perf_counter() - started) * 1000, 1))
        print(
            json.dumps(
                {
                    "variant": name,
                    "ms": elapsed,
                    "p50_ms": statistics.median(elapsed),
                    "max_ms": max(elapsed),
                    "returned": len(items),
                    "warnings": warnings,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
