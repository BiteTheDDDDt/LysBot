"""Run reproducible, isolated Chinese retrieval and agent evaluations; never send QQ messages."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests/fixtures/agentic_search_eval.json"
SID = "group_900000001"


def write_result(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


async def run(args):
    import nonebot

    nonebot.init()
    from src.plugins.llm_chat import memory, models, tools
    from src.plugins.llm_chat.agent_runner import configure_store
    from src.plugins.llm_chat.config import get_tool_usage_prompt
    from src.plugins.llm_chat.llm_logger import drain_usage_tasks
    from src.plugins.llm_chat.memory_index import MemoryIndex, encode
    from src.plugins.llm_chat.retrieval import HistorySearch
    from src.plugins.llm_chat.usage_db import close_db

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    manager = memory.memory_manager
    manager._db_path = args.database
    manager._closing = True
    db = await manager._get_db()
    search = HistorySearch(args.database)
    index = MemoryIndex(args.database)
    try:
        if args.action == "prepare":
            existing = await db.execute("SELECT count(*) FROM messages_detail")
            if (await existing.fetchone())[0]:
                raise ValueError("Evaluation preparation requires an empty database")
            await db.executemany(
                """INSERT INTO messages_detail(id,session_id,role,sender_id,name,content,timestamp)
                VALUES(?,?,'user',?,?,?,?)""",
                [(r["id"], SID, r["sender_id"], r["name"], r["content"], r["timestamp"]) for r in fixture["messages"]],
            )
            await db.commit()
            await asyncio.to_thread(index._prepare_vectors)
            while await asyncio.to_thread(index._project_one):
                pass
            print(encode({"prepared_messages": len(fixture["messages"]), "cases": len(fixture["cases"])}))
            return
        cases = [c for c in fixture["cases"] if not args.case or args.case in c["id"]]
        if args.limit:
            cases = cases[: args.limit]
        done = set()
        if args.output.exists():
            done = {
                (r["variant"], r["id"])
                for line in args.output.read_text(encoding="utf-8").splitlines()
                if (r := json.loads(line)) and not r.get("error")
            }
        configure_store(args.database)
        tokens_spent = 0
        original_execute = tools.execute_tool

        async def execute(name, arguments):
            if name not in {"recall_history", "read_history", "search_web", "read_web"}:
                return '{"status":"invalid_arguments","error":{"code":"evaluation_read_only"}}'
            if name == "recall_history" and args.variant in {"like", "fts", "hybrid"}:
                arguments = dict(arguments)
                if arguments.get("mode") not in {"aggregate", "timeline"}:
                    arguments["mode"] = "hybrid" if args.variant == "hybrid" else "keyword"
            if args.variant == "like":
                with patch("src.plugins.llm_chat.retrieval.tokens", return_value=[]):
                    return await original_execute(name, arguments)
            return await original_execute(name, arguments)

        for case in cases:
            if (args.variant, case["id"]) in done:
                continue
            if args.max_model_tokens and tokens_spent >= args.max_model_tokens:
                print("Evaluation paused at its per-run token cap; remaining cases can be resumed.", flush=True)
                break
            started = time.monotonic()
            try:
                if args.action == "retrieve":
                    if case["kind"] == "aggregate":
                        found = await search.search(
                            SID, mode="aggregate", sources=["messages"], author_ids=[case["author_id"]]
                        )
                        score = found["aggregation"]["values"] == 3
                    else:
                        query = case["seed_query"]
                        # Resolve the known name through the documented filter.
                        owner = next(r["name"] for r in fixture["messages"] if r["sender_id"] == case["author_id"])
                        named = owner in query
                        kwargs = {
                            "query": "" if named else query,
                            "sender_names": [owner] if named else None,
                            "sources": ["messages"],
                            "mode": "hybrid" if args.variant == "hybrid" else "keyword",
                            "limit": 8,
                        }
                        if args.variant == "like":
                            with patch("src.plugins.llm_chat.retrieval.tokens", return_value=[]):
                                found = await search.search(SID, **kwargs)
                        else:
                            found = await search.search(SID, **kwargs)
                        refs = {
                            int(ref.split(":")[-1])
                            for item in found["items"]
                            for ref in item["source_ids"]
                            if ref.startswith("h:")
                        }
                        expected = set(case["expected_ids"])
                        score = len(refs & expected) / len(expected) if expected else None
                    record = {
                        "status": found["status"],
                        "source_recall_at_8": score,
                        "warnings": found.get("warnings", []),
                    }
                else:
                    prompt = (
                        "你是中文历史查证助手。只按当前会话的可核对来源回答；信息不足就说明无法确认。"
                        "你自己判断查什么、怎样查和何时继续。回答不超过500字，附已读历史来源ID。\n"
                        + get_tool_usage_prompt()
                    )
                    selected = [
                        t for t in tools.BASE_CHAT_TOOLS if t["function"]["name"] in {"recall_history", "read_history"}
                    ]
                    with patch.object(tools, "BASE_CHAT_TOOLS", selected), patch.object(tools, "execute_tool", execute):
                        response = await models.model_manager.chat(
                            [
                                models.Message(role="system", content=prompt),
                                models.Message(role="user", content=case["question"], sender_id=case["author_id"]),
                            ],
                            model=args.model or None,
                            role="chat",
                            caller="agentic-eval",
                            session_id=SID,
                            allow_fallback=False,
                        )
                    answer = str(response)
                    agent = response.agent_run
                    tokens_spent += int((getattr(response, "usage", None) or {}).get("total_tokens", 0) or 0)
                    supported = [key for key, value in agent.evidence.items() if value.get("read_state") == "original"]
                    record = {
                        "answer": answer,
                        "expected_answer": case["expected_answer"],
                        "expected_ids": case["expected_ids"],
                        "read_evidence_ids": supported,
                        "tool_calls": agent.tool_calls,
                        "tool_rounds": agent.tool_round,
                        "model": getattr(response, "model_spec", args.model),
                        "usage": getattr(response, "usage", None),
                        "trace": agent.transcript,
                        "requires_manual_judgment": True,
                    }
                write_result(
                    args.output,
                    {
                        "id": case["id"],
                        "variant": args.variant,
                        "kind": case["kind"],
                        "seconds": round(time.monotonic() - started, 3),
                        **record,
                    },
                )
            except Exception as error:  # noqa: BLE001 -- persist the failed case before pausing remote evaluations.
                message = str(error).lower()
                failed_usage = getattr(error, "usage", None) or {}
                tokens_spent += int(failed_usage.get("total_tokens", 0) or 0)
                blocked = any(
                    code in message for code in ("model_cooldown", "arrearage", "insufficient_quota", "invalid_api_key")
                )
                write_result(
                    args.output,
                    {
                        "id": case["id"],
                        "variant": args.variant,
                        "error": type(error).__name__,
                        "account_blocked": blocked,
                        "usage": failed_usage,
                        "seconds": round(time.monotonic() - started, 3),
                    },
                )
                if blocked or args.action == "agent":
                    print("Evaluation paused at model error; inspect the failed case before resuming.", flush=True)
                    break
    finally:
        await manager.close()
        await drain_usage_tasks()
        await close_db()
        for adapter in models.model_manager._adapters.values():
            if getattr(adapter, "client", None):
                await adapter.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "retrieve", "agent"])
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--variant", choices=["like", "fts", "hybrid", "adaptive"], default="fts")
    parser.add_argument("--model", default="")
    parser.add_argument("--case", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-tokens", type=int, default=60000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/agentic-eval/results.jsonl")
    args = parser.parse_args()
    args.database = args.database.resolve()
    if args.database == ROOT / "data/history.db" or "agentic-eval" not in args.database.parts:
        parser.error("Database must be inside a dedicated agentic-eval directory")
    args.database.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_CHAT_DATA_DIR"] = str(args.database.parent)
    os.environ["LLM_CHAT_DATABASE_PATH"] = str(args.database)
    os.environ["MEMORY_INDEX_ENABLED"] = "false"
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
