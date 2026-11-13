import asyncio
import json
import sqlite3
import struct
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from contextlib import contextmanager

import nonebot

nonebot.init()

from src.plugins.llm_chat import agent_runner, models, tools
from src.plugins.llm_chat import memory as memory_module
from src.plugins.llm_chat.agent_runner import AgentRunner, current_run
from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.memory_index import MemoryIndex, connect, generation
from src.plugins.llm_chat.retrieval import HistorySearch, cursor_offset


class SearchStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.tmp.name) / "history.db"
        self.manager._closing = True
        db = await self.manager._get_db()
        await db.executemany(
            """INSERT INTO messages_detail(session_id,role,name,sender_id,content,timestamp)
            VALUES(?,?,?,?,?,?)""",
            [
                ("group_7", "user", "甲", 101, "我打算买 RTX 5090 显卡", "2026-09-01T10:00:00"),
                ("group_7", "user", "乙", 202, "建议选量化版本 Orion 32B", "2026-09-01T10:01:00"),
                ("group_7", "user", "甲", 101, "最终买了 RTX 4090，没买 5090", "2026-09-10T10:00:00"),
                ("group_7", "user", "甲", 101, "我的图形处理器有 24GB 显存", "2026-09-10T10:01:00"),
                ("group_8", "user", "甲", 101, "绝密跨群内容 RTX 5090", "2026-09-10T10:00:00"),
            ],
        )
        await db.execute(
            "INSERT INTO summaries(session_id,content,created_at) VALUES('group_7','讨论过 RTX 5090','2026-09-15T00:00:00')"
        )
        await db.execute(
            "INSERT INTO important_info(session_id,user_id,info_type,content,created_at) VALUES('group_7',101,'user_preference','甲偏好绿色','2026-09-15T00:00:00')"
        )
        await db.commit()
        self.index = MemoryIndex(self.manager._db_path)
        await asyncio.to_thread(self.index._prepare_vectors)
        while await asyncio.to_thread(self.index._project_one):
            pass
        self.search = HistorySearch(self.manager._db_path)

    async def asyncTearDown(self):
        await self.manager.close()
        MemoryManager._instance = None
        self.tmp.cleanup()

    async def test_chinese_fulltext_and_original_expansion(self):
        found = await self.search.search("group_7", query="处理器", mode="keyword")
        self.assertTrue(found["items"])
        source = next(i for i in found["items"] if "图形处理器" in i["content"])
        original = await self.search.read("group_7", ids=[source["id"]])
        self.assertEqual(original["status"], "ok")
        self.assertTrue(all(i["read_state"] == "original" for i in original["items"]))

    async def test_current_request_exclusion_removes_text_from_windows_and_keeps_old_sources(self):
        found = await self.search.search("group_7", query="RTX", mode="keyword",
                                        sources=["messages"], exclude_message_ids=[3])
        self.assertTrue(found["items"])
        for item in found["items"]:
            self.assertNotIn("h:0:messages:3", item["source_ids"])
            self.assertNotIn("最终买了", item["excerpt"])
        self.assertTrue(any("h:0:messages:1" in i["source_ids"] for i in found["items"]))
        self.assertIn("embedding_ms", found["metrics"])
        self.assertIn("retrieval_ms", found["metrics"])
        self.assertIn("original_messages", found["metrics"]["candidate_counts"])
        # Explicit source reads remain available; exclusion is a recall policy, not deletion.
        read = await self.search.read("group_7", ids=["h:0:messages:3"])
        self.assertEqual(read["status"], "ok")

    async def test_request_source_ids_are_scoped_and_resolved_once(self):
        tools.set_poke_context(7, 101, history_request_texts=["最终买了 RTX 4090，没买 5090", "绝密跨群内容 RTX 5090"])
        try:
            runner = AgentRunner("group_7")
            runner.memory_path = self.manager._db_path
            await runner.start([models.Message(role="user", content="追问")])
            self.assertEqual(runner.request_source_ids, [3])
            await runner.finish("complete")
        finally:
            tools.clear_poke_context()

    async def test_author_filter_never_returns_other_speakers_text(self):
        found = await self.search.search("group_7", query="RTX", mode="keyword", author_ids=["101"])
        self.assertTrue(found["items"])
        self.assertTrue(all(set(i["authors"]) <= {"101"} for i in found["items"]))
        self.assertNotIn("Orion", json.dumps(found, ensure_ascii=False))

    async def test_group_scope_applies_to_search_and_direct_ids(self):
        found = await self.search.search("group_7", query="5090", mode="keyword")
        self.assertNotIn("绝密", json.dumps(found, ensure_ascii=False))
        denied = await self.search.read("group_7", ids=["h:0:messages:5"])
        self.assertEqual(denied["status"], "invalid_arguments")

    async def test_exact_counts_are_canonical_not_top_k(self):
        counted = await self.search.search(
            "group_7", mode="aggregate", sources=["messages"], group_by="author", limit=1
        )
        self.assertEqual(counted["aggregation"]["values"], [{"key": 101, "count": 3}, {"key": 202, "count": 1}])

    async def test_unknown_summary_event_time_not_creation_time(self):
        found = await self.search.search(
            "group_7", query="5090", mode="keyword", sources=["summaries"], since="2026-09-14"
        )
        self.assertFalse(found["items"])

    async def test_anchor_context_preserves_speakers(self):
        opened = await self.search.read("group_7", anchor_id="h:0:messages:2", before=1, after=1)
        self.assertEqual([i["sender_id"] for i in opened["items"]], [101, 202, 101])

    async def test_cursor_is_scoped_and_tamper_evident(self):
        first = await self.search.read("group_7", anchor_id="h:0:messages:2", before=1, after=2, limit=1)
        self.assertTrue(first["next_cursor"])
        second = await self.search.read(
            "group_7", anchor_id="h:0:messages:2", before=1, after=2, limit=1, cursor=first["next_cursor"]
        )
        self.assertEqual(second["items"][0]["sender_id"], 202)
        denied = await self.search.read(
            "group_8", anchor_id="h:0:messages:2", before=1, after=2, limit=1, cursor=first["next_cursor"]
        )
        self.assertEqual(denied["status"], "invalid_arguments")
        with self.assertRaises(ValueError):
            cursor_offset(first["next_cursor"][:-4] + "AAAA", "x")

    async def test_search_pages_continue_only_while_source_snapshot_is_unchanged(self):
        query = {"mode": "keyword", "sources": ["messages"], "exact_phrase": "RTX", "limit": 1}
        first = await self.search.search("group_7", **query)
        second = await self.search.search("group_7", cursor=first["next_cursor"], **query)
        self.assertEqual(first["items"][0]["id"], "h:0:messages:3")
        self.assertEqual(second["items"][0]["id"], "h:0:messages:1")
        with connect(self.manager._db_path) as db:
            db.execute(
                "INSERT INTO messages_detail(session_id,role,content,timestamp) "
                "VALUES('group_7','user','新增 RTX 讨论','2026-09-16T00:00:00')"
            )
        expired = await self.search.search("group_7", cursor=first["next_cursor"], **query)
        self.assertEqual(expired["error"]["code"], "cursor_expired_retry_search")
        self.assertFalse(expired["items"])

    async def test_source_deletion_expires_timeline_cursor_instead_of_skipping(self):
        query = {"mode": "timeline", "sources": ["messages"], "sort": "newest", "limit": 1}
        first = await self.search.search("group_7", **query)
        with connect(self.manager._db_path) as db:
            db.execute("DELETE FROM messages_detail WHERE id=4")
        expired = await self.search.search("group_7", cursor=first["next_cursor"], **query)
        self.assertEqual(expired["error"]["code"], "cursor_expired_retry_search")

    async def test_vector_backfill_keeps_keyword_cursor_when_results_are_unchanged(self):
        query = {"mode": "keyword", "query": "RTX", "sources": ["messages"], "limit": 1}
        first = await self.search.search("group_7", **query)
        revision = first["coverage"]["source_revision"]
        rows = await asyncio.to_thread(self.index._pending_vectors)
        vector = struct.pack("<512f", 1, *([0] * 511))
        await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        self.assertEqual(self.search._snapshot("group_7")[3]["source_revision"], revision)
        second = await self.search.search("group_7", cursor=first["next_cursor"], **query)
        self.assertNotIn("error", second)
        self.assertTrue(second["items"])
        self.assertEqual(first["coverage"], second["coverage"])
        self.assertTrue({row["id"] for row in first["items"]}.isdisjoint(row["id"] for row in second["items"]))

    async def test_backend_recovery_expires_cursor_when_candidate_ranks_change(self):
        vector = struct.pack("<512f", 1, *([0] * 511))
        rows = await asyncio.to_thread(self.index._pending_vectors)
        await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        connection = self.search._connect

        def failed_query_only(*, vectors=False, deadline=None):
            if vectors and deadline is not None:
                raise sqlite3.OperationalError("vector query unavailable")
            return connection(vectors=vectors, deadline=deadline)

        query = {"mode": "semantic", "query": "RTX", "sources": ["messages"], "limit": 1}
        with patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(return_value=[vector])):
            with patch.object(self.search, "_connect", side_effect=failed_query_only):
                first = await self.search.search("group_7", **query)
            self.assertTrue(first["next_cursor"])
            self.assertEqual(first["coverage"], self.search._snapshot("group_7")[3])
            expired = await self.search.search("group_7", cursor=first["next_cursor"], **query)
        self.assertEqual(expired["error"]["code"], "cursor_expired_retry_search")

    async def test_source_write_and_job_rollback_together(self):
        db = await self.manager._get_db()
        await db.execute(
            "INSERT INTO messages_detail(session_id,role,content,timestamp) VALUES('group_99','user','未提交','2026-09-15')"
        )
        await db.rollback()
        with connect(self.manager._db_path) as read:
            self.assertEqual(
                read.execute("SELECT count(*) FROM memory_index_jobs WHERE session_id='group_99'").fetchone()[0], 0
            )

    async def test_deletion_rejects_old_vector_commit_and_old_source_id(self):
        rows = await asyncio.to_thread(self.index._pending_vectors)
        rows = [r for r in rows if r["session_id"] == "group_7"]
        await self.manager.clear_history(0, 7, full=True)
        vector = struct.pack("<512f", 1, *([0] * 511))
        await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        with connect(self.manager._db_path, vectors=True) as db:
            self.assertEqual(generation(db, "group_7"), 1)
            self.assertEqual(
                db.execute(
                    f"SELECT count(*) FROM memory_vec_{self.index.version} WHERE session_id='group_7'"
                ).fetchone()[0],
                0,
            )
        denied = await self.search.read("group_7", ids=["h:0:messages:1"])
        self.assertEqual(denied["status"], "invalid_arguments")
        self.assertFalse((await self.search.search("group_7", query="RTX", mode="keyword"))["items"])

    async def test_semantic_query_uses_real_sqlite_vec_and_respects_scope(self):
        vector = struct.pack("<512f", 1, *([0] * 511))
        rows = await asyncio.to_thread(self.index._pending_vectors)
        await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        with patch("src.plugins.llm_chat.retrieval.embed", new=AsyncMock(return_value=[vector])):
            found = await self.search.search("group_7", query="新配的计算设备", mode="semantic")
        self.assertTrue(found["items"])
        self.assertEqual(found["effective_mode"], "semantic")
        self.assertNotIn("绝密", json.dumps(found, ensure_ascii=False))

    async def test_embedding_failure_is_partial_not_empty_success(self):
        with patch("src.plugins.llm_chat.retrieval.embed", new=AsyncMock(side_effect=TimeoutError())):
            found = await self.search.search("group_7", query="4090")
        self.assertEqual(found["status"], "partial")
        self.assertEqual(found["effective_mode"], "keyword")
        self.assertTrue(found["items"])

    async def test_semantic_backend_failure_really_falls_back_to_originals(self):
        connection = self.search._connect

        def without_vectors(*, vectors=False, deadline=None):
            if vectors:
                raise sqlite3.OperationalError("vector table unavailable")
            return connection(deadline=deadline)

        vector = struct.pack("<512f", 1, *([0] * 511))
        with (
            patch.object(self.search, "_connect", side_effect=without_vectors),
            patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(return_value=[vector])),
        ):
            found = await self.search.search("group_7", query="4090", mode="semantic")
        self.assertEqual(found["status"], "partial")
        self.assertEqual(found["effective_mode"], "keyword")
        self.assertIn("vector_backend_unavailable", found["warnings"])
        self.assertTrue(any("4090" in item["content"] for item in found["items"]))
        self.assertNotIn("绝密", json.dumps(found, ensure_ascii=False))

    async def test_semantic_timeout_gives_lexical_fallback_a_fresh_bounded_budget(self):
        connection = self.search._connect
        # Cross a float exponent boundary when refreshing the SQL deadline.
        # Use the same fake clock for retrieval and SQLite progress checks.
        clock = [1015.8]
        fallback_deadlines = []
        vector_failed = False

        def interrupted_vectors(*, vectors=False, deadline=None):
            nonlocal vector_failed
            if vectors and deadline is not None:
                clock[0] = deadline + 0.1
                vector_failed = True
                raise sqlite3.OperationalError("interrupted")
            if vector_failed and deadline is not None:
                fallback_deadlines.append((clock[0], deadline))
            return connection(vectors=vectors, deadline=deadline)

        vector = struct.pack("<512f", 1, *([0] * 511))
        with (
            patch("src.plugins.llm_chat.retrieval.time", SimpleNamespace(monotonic=lambda: clock[0])),
            patch("src.plugins.llm_chat.memory_index.time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)),
            patch.object(self.search, "_connect", side_effect=interrupted_vectors),
            patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(return_value=[vector])),
        ):
            found = await self.search.search("group_7", query="4090", mode="semantic")
        self.assertIn("vector_query_budget_exhausted", found["warnings"])
        self.assertEqual(found["effective_mode"], "keyword")
        self.assertTrue(found["items"])
        self.assertTrue(fallback_deadlines)
        for started, deadline in fallback_deadlines:
            # Compare absolute deadlines: subtracting float timestamps can
            # turn an exact eight-second budget into 8.000000000000114.
            self.assertEqual(deadline, started + 8)

    async def test_source_change_during_embedding_requires_a_fresh_search(self):
        async def embedding_with_concurrent_write(*args, **kwargs):
            with connect(self.manager._db_path) as db:
                db.execute("UPDATE messages_detail SET content='更正后的显卡' WHERE id=3")
            return [struct.pack("<512f", 1, *([0] * 511))]

        with patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(side_effect=embedding_with_concurrent_write)):
            found = await self.search.search("group_7", query="4090", mode="semantic")
        self.assertEqual(found["error"]["code"], "scope_changed_retry_search")
        self.assertTrue(found["error"]["retryable"])
        self.assertFalse(found["items"])

    async def test_new_message_id_and_fact_provenance_are_committed(self):
        with patch.object(memory_module.config, "enable_summary", False):
            await self.manager.add_message(0, "user", "我准备学习日语", 7, name="甲", sender_id=101)
        message = self.manager._cache["group_7"][-1]
        self.assertIsInstance(message.source_id, int)
        answer = json.dumps(
            [
                {
                    "user_id": 101,
                    "type": "key_event",
                    "content": "计划学习日语",
                    "source_ids": [message.source_id],
                    "assertion": "planned",
                    "attribute": "学习计划",
                }
            ]
        )
        with patch.object(models.model_manager, "chat", AsyncMock(return_value=answer)):
            entries = await self.manager._extract_important_info("group_7", [message])
        self.assertEqual(entries[0].provenance["source_ids"], [message.source_id])
        self.assertEqual(entries[0].provenance["assertion"], "planned")
        with connect(self.manager._db_path) as db:
            saved = json.loads(db.execute("SELECT messages FROM sessions WHERE session_id='group_7'").fetchone()[0])
        self.assertEqual(saved[-1]["source_id"], message.source_id)

    async def test_fact_extractor_rejects_a_source_from_another_group(self):
        message = models.Message(
            role="user", content="我打算买 RTX 5090 显卡", sender_id=101, source_id=1, timestamp="2026-09-01T10:00:00"
        )
        answer = json.dumps(
            [
                {
                    "user_id": 101,
                    "type": "key_event",
                    "content": "已经购买显卡",
                    "source_ids": [5],
                    "assertion": "asserted",
                }
            ]
        )
        with (
            patch.object(models.model_manager, "chat", AsyncMock(return_value=answer)),
            self.assertRaisesRegex(ValueError, "事实来源"),
        ):
            await self.manager._extract_important_info("group_7", [message])

    async def test_profile_keeps_plan_separate_and_hides_explicitly_superseded_fact(self):
        db = await self.manager._get_db()
        old = await db.execute(
            """INSERT INTO important_info(session_id,user_id,info_type,content,created_at,provenance_json)
            VALUES('group_7',101,'user_identity','使用旧显卡','2026-09-01',?)""",
            (
                json.dumps(
                    {"source_ids": [1], "provenance": "derived", "assertion": "asserted", "attribute": "显卡型号"}
                ),
            ),
        )
        await db.execute(
            """INSERT INTO important_info(session_id,user_id,info_type,content,created_at,provenance_json)
            VALUES('group_7',101,'user_identity','现在使用 RTX 4090','2026-09-10',?)""",
            (
                json.dumps(
                    {
                        "source_ids": [3],
                        "provenance": "derived",
                        "assertion": "asserted",
                        "attribute": "显卡型号",
                        "supersedes": [old.lastrowid],
                    }
                ),
            ),
        )
        await db.execute(
            """INSERT INTO important_info(session_id,user_id,info_type,content,created_at,provenance_json)
            VALUES('group_7',101,'key_event','计划升级显卡','2026-09-11',?)""",
            (
                json.dumps(
                    {"source_ids": [1], "provenance": "derived", "assertion": "planned", "attribute": "升级计划"}
                ),
            ),
        )
        await db.commit()
        profile = await self.search.profile("group_7", [101])
        self.assertNotIn("使用旧显卡", profile)
        self.assertIn("陈述：现在使用 RTX 4090", profile)
        self.assertIn("计划：计划升级显卡", profile)

    async def test_long_original_is_readable_in_complete_pages(self):
        db = await self.manager._get_db()
        source = await db.execute(
            "INSERT INTO messages_detail(session_id,role,content,timestamp) VALUES('group_7','user',?,'2026-09-15')",
            ("长" * 25000,),
        )
        await db.commit()
        ref = f"h:0:messages:{source.lastrowid}"
        pages = []
        cursor = None
        for _ in range(4):
            page = await self.search.read("group_7", ids=[ref], cursor=cursor)
            pages.extend(i["content"] for i in page["items"])
            cursor = page["next_cursor"]
            if not cursor:
                break
        self.assertEqual("".join(pages), "长" * 25000)

    async def test_persistent_unknown_delivery_is_not_replayed(self):
        with patch.object(agent_runner, "_runtime_path", self.manager._db_path):
            runner = AgentRunner("group_7")
        await runner.start([models.Message(role="user", content="查证一下", sender_id=101)])
        send = AsyncMock(side_effect=TimeoutError())
        for _ in range(2):
            with self.assertRaises(agent_runner.DeliveryUncertain):
                await runner.deliver("这是一条答复", send)
        self.assertEqual(send.await_count, 1)
        await runner.finish("interrupted")
        with connect(self.manager._db_path) as db:
            state = json.loads(db.execute("SELECT state_json FROM agent_runs WHERE id=?", (runner.id,)).fetchone()[0])
        self.assertEqual(next(iter(state["actions"].values()))["status"], "unknown")
        with patch.object(agent_runner, "_runtime_path", self.manager._db_path):
            resumed = AgentRunner("group_7")
        await resumed.start([models.Message(role="user", content="继续", sender_id=101)])
        with self.assertRaises(agent_runner.DeliveryUncertain):
            await resumed.deliver("这是一条答复", send)
        self.assertEqual(send.await_count, 1)
        await resumed.finish("complete")

    async def test_run_recovery_does_not_require_the_vector_worker(self):
        db = await self.manager._get_db()
        await db.execute("INSERT INTO agent_runs VALUES('interrupted-example','group_7',0,'running','{}',9999999999)")
        await db.commit()
        with (
            patch.object(memory_module.config, "memory_index_enabled", False),
            patch.object(memory_module.config, "enable_important_info", False),
        ):
            await self.manager.start()
        with connect(self.manager._db_path) as read:
            self.assertEqual(
                read.execute("SELECT status FROM agent_runs WHERE id='interrupted-example'").fetchone()[0],
                "interrupted",
            )
        self.assertIsNone(self.manager.search_index)

    async def test_invalid_embedding_settings_keep_lexical_worker_available(self):
        index = MemoryIndex(self.manager._db_path)
        index.settings["model"] = "invalid-no-provider"
        await index.start()
        try:
            self.assertFalse(index.vector_available)
            self.assertIsNotNone(index.task)
            found = await self.search.search("group_7", query="4090", mode="keyword")
            self.assertTrue(found["items"])
        finally:
            await index.close()

    async def test_new_message_vectors_take_priority_over_historical_backfill(self):
        with connect(self.manager._db_path) as db:
            db.execute("UPDATE memory_items SET vector_retry_at=0")
        db = await self.manager._get_db()
        await db.execute(
            "INSERT INTO important_info(session_id,user_id,info_type,content,created_at) VALUES('group_7',101,'key_event','本轮最新确认的情况','2026-09-16')"
        )
        await db.commit()
        while await asyncio.to_thread(self.index._project_one):
            pass
        pending = await asyncio.to_thread(self.index._pending_vectors)
        self.assertEqual(pending[0]["content"], "本轮最新确认的情况")

    async def test_sql_progress_deadline_interrupts_work_inside_sqlite(self):
        import sqlite3
        import time

        with (
            connect(self.manager._db_path, deadline=time.monotonic() - 1) as db,
            self.assertRaisesRegex(sqlite3.OperationalError, "interrupted"),
        ):
            db.execute(
                "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<10000000) SELECT sum(x) FROM n"
            ).fetchone()

    async def test_deleting_read_evidence_cancels_before_a_reply_is_sent(self):
        from src.plugins.llm_chat.conversation import TurnSuperseded

        with patch.object(agent_runner, "_runtime_path", self.manager._db_path):
            run = AgentRunner("group_7")
        await run.start([models.Message(role="user", content="查一下", sender_id=101)])
        opened = await self.search.read("group_7", ids=["h:0:messages:1"])
        run.evidence = {item["id"]: item for item in opened["items"]}
        db = await self.manager._get_db()
        await db.execute("DELETE FROM messages_detail WHERE id=1")
        await db.commit()
        send = AsyncMock()
        with self.assertRaises(TurnSuperseded):
            await run.deliver("原来的内容", send)
        send.assert_not_awaited()
        await run.finish("interrupted")

    async def test_canonical_update_rejects_a_vector_computed_before_the_update(self):
        rows = await asyncio.to_thread(self.index._pending_vectors)
        row = next(r for r in rows if r["kind"] == "messages" and r["session_id"] == "group_7")
        db = await self.manager._get_db()
        await db.execute("UPDATE messages_detail SET content='内容已经改变' WHERE id=1")
        await db.commit()
        vector = struct.pack("<512f", 1, *([0] * 511))
        await asyncio.to_thread(self.index._commit_vectors, [row], [vector])
        with connect(self.manager._db_path, vectors=True) as read:
            self.assertIsNone(
                read.execute(
                    f"SELECT item_id FROM memory_vec_{self.index.version} WHERE item_id=?", (row["id"],)
                ).fetchone()
            )

    async def test_deleted_support_invalidates_a_fact_even_by_direct_id(self):
        db = await self.manager._get_db()
        fact = await db.execute(
            "INSERT INTO important_info(session_id,user_id,info_type,content,created_at,provenance_json) VALUES('group_7',101,'key_event','秘密学习计划','2026-09-15',?)",
            (json.dumps({"provenance": "derived", "source_ids": [1]}),),
        )
        await db.commit()
        await db.execute("DELETE FROM messages_detail WHERE id=1")
        await db.commit()
        found = await self.search.search("group_7", query="秘密学习", sources=["important_info"], mode="keyword")
        self.assertFalse(found["items"])
        read = await self.search.read("group_7", ids=[f"h:0:important_info:{fact.lastrowid}"])
        self.assertEqual(read["status"], "invalid_arguments")

    async def test_wrong_filter_types_are_rejected_instead_of_coerced(self):
        for kwargs in (
            {"author_ids": "101"},
            {"sender_names": "甲"},
            {"author_ids": [True]},
            {"sort": "random"},
            {"exact_phrase": 123},
        ):
            self.assertEqual((await self.search.search("group_7", **kwargs))["status"], "invalid_arguments")
        self.assertEqual(
            (await self.search.read("group_7", ids=["h:0:messages:1"], before=True))["status"], "invalid_arguments"
        )

    async def test_index_import_keeps_new_records_and_excludes_stale_projection(self):
        import sqlite3
        from contextlib import closing

        from scripts.agentic_search_ops import import_index

        target = Path(self.tmp.name) / "target.db"
        with closing(sqlite3.connect(self.manager._db_path)) as source, closing(sqlite3.connect(target)) as destination:
            source.backup(destination)
        with connect(target, vectors=True) as db:
            db.execute("DELETE FROM memory_links")
            db.execute("DELETE FROM memory_items")
            db.execute("DELETE FROM memory_fts")
            db.execute(f"DELETE FROM memory_vec_{self.index.version}")
            db.execute("UPDATE messages_detail SET content='生产新增的修正' WHERE id=1")
            db.execute(
                "INSERT INTO messages_detail(session_id,role,content,timestamp) VALUES('group_7','user','快照之后的新消息','2026-09-16')"
            )
            db.execute("DELETE FROM messages_detail WHERE id=2")
        report = await asyncio.to_thread(import_index, target, self.manager._db_path)
        self.assertGreater(report["excluded_stale_items"], 0)
        self.assertGreater(report["catchup_jobs"], 0)
        with connect(target) as db:
            self.assertEqual(
                db.execute("SELECT content FROM messages_detail WHERE id=1").fetchone()[0], "生产新增的修正"
            )
            self.assertIsNone(db.execute("SELECT 1 FROM messages_detail WHERE id=2").fetchone())
            self.assertEqual(
                db.execute("SELECT count(*) FROM messages_detail WHERE content='快照之后的新消息'").fetchone()[0], 1
            )
        found = await HistorySearch(target).search("group_7", query="Orion", mode="keyword")
        self.assertFalse(found["items"])

    async def test_rare_subject_filter_is_applied_before_vector_top_k(self):
        db = await self.manager._get_db()
        await db.executemany(
            "INSERT INTO important_info(session_id,user_id,info_type,content,created_at) VALUES('group_7',?,'key_event',?,'2026-09-15')",
            [(303, f"日语学习记录 {i}") for i in range(310)] + [(404, "日语学习独特目标")],
        )
        await db.commit()
        while await asyncio.to_thread(self.index._project_one):
            pass
        vector = struct.pack("<512f", 1, *([0] * 511))
        while rows := await asyncio.to_thread(self.index._pending_vectors):
            await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        with patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(return_value=[vector])):
            found = await self.search.search(
                "group_7", query="语言训练", mode="semantic", subject_ids=[404], sources=["important_info"]
            )
        self.assertTrue(found["items"])
        self.assertTrue(all(i["subjects"] == ["404"] for i in found["items"]))
        capped = await self.search.search("group_7", query="日语", mode="keyword", sources=["important_info"])
        self.assertEqual(capped["status"], "partial")
        self.assertTrue(any("candidate_budget" in w for w in capped["warnings"]))

    async def test_missing_fts_returns_original_results_with_partial_status(self):
        with connect(self.manager._db_path) as db:
            db.execute("DROP TABLE memory_fts")
        found = await self.search.search("group_7", query="4090", mode="keyword")
        self.assertEqual(found["status"], "partial")
        self.assertIn("fts_backend_unavailable", found["warnings"])
        self.assertTrue(found["items"])

    async def test_embedding_rebuild_keeps_old_version_until_all_items_are_ready(self):
        from scripts.agentic_search_ops import snapshot as operation_snapshot

        vector = struct.pack("<512f", 1, *([0] * 511))
        rows = await asyncio.to_thread(self.index._pending_vectors)
        await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        with patch("src.plugins.llm_chat.memory_index.embedding_settings", return_value={**self.index.settings, "revision": "test-next"}):
            rebuilt = MemoryIndex(self.manager._db_path)
        with connect(self.manager._db_path) as db:
            db.execute("UPDATE memory_items SET vector_attempts=8,vector_error='old_backend_error'")
        await asyncio.to_thread(rebuilt._prepare_vectors)
        initial = operation_snapshot(self.manager._db_path, rebuilt.version)
        self.assertEqual(initial["vectors_ready"], 0)
        self.assertEqual(initial["active_embedding_version"], self.index.version)
        pending = await asyncio.to_thread(rebuilt._pending_vectors)
        await asyncio.to_thread(rebuilt._commit_vectors, pending[:1], [vector])
        await asyncio.to_thread(rebuilt._activate_if_complete)
        snapshot = self.search._snapshot("group_7")
        self.assertEqual(snapshot[1], self.index.version)
        self.assertGreater(operation_snapshot(self.manager._db_path, self.index.version)["vectors_ready"], 0)
        await asyncio.to_thread(rebuilt._commit_vectors, pending[1:], [vector] * len(pending[1:]))
        await asyncio.to_thread(rebuilt._activate_if_complete)
        self.assertEqual(self.search._snapshot("group_7")[1], rebuilt.version)
        complete = operation_snapshot(self.manager._db_path, rebuilt.version)
        self.assertEqual(complete["vectors_ready"], complete["items"])
        self.assertEqual(complete["active_embedding_version"], rebuilt.version)

    async def test_local_rebuild_does_not_call_old_remote_query_backend(self):
        # Seed a historical remote fingerprint without enabling remote inference.
        with connect(self.manager._db_path) as db:
            db.execute("UPDATE kv_meta SET value=? WHERE key=?", (
                json.dumps({**self.index.settings, "model": "dashscope:old-embedding"}),
                "memory_embedding:" + self.index.version,
            ))
        with patch("src.plugins.llm_chat.retrieval.embed", AsyncMock()) as embedding:
            found = await self.search.search("group_7", query="图形处理器", mode="hybrid")
        embedding.assert_not_awaited()
        self.assertIn("local_embedding_rebuild_pending", found["warnings"])
        self.assertTrue(found["items"])

    async def test_semantic_mode_preserves_nearest_vector_over_keyword_match(self):
        closest = struct.pack("<512f", 1, *([0] * 511))
        unrelated = struct.pack("<512f", 0, 1, *([0] * 510))
        rows = await asyncio.to_thread(self.index._pending_vectors)
        target = next(row for row in rows if row["session_id"] == "group_7" and "打算买" in row["content"])
        vectors = [closest if row["id"] == target["id"] else unrelated for row in rows]
        await asyncio.to_thread(self.index._commit_vectors, rows, vectors)
        with patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(return_value=[closest])):
            found = await self.search.search("group_7", query="图形处理器", mode="semantic")
        self.assertEqual(found["items"][0]["id"], f"i:0:{target['id']}")

    async def test_web_cache_is_purged_even_when_vector_extension_is_disabled(self):
        from src.plugins.llm_chat.web_search import cache_root

        root = cache_root(self.manager._db_path, "group_7") / "0" / ("a" * 32)
        root.mkdir(parents=True)
        cached = root / "page.json"
        cached.write_text("private cached page")
        await self.manager.clear_history(0, 7, full=True)
        self.index.vector_available = False
        await asyncio.to_thread(self.index._purge)
        self.assertFalse(cached.exists())

    async def test_source_edit_invalidates_run_read_cache(self):
        with patch.object(agent_runner, "_runtime_path", self.manager._db_path):
            run = AgentRunner("group_7")
        await run.start([models.Message(role="user", content="查一下", sender_id=101)])
        execute = AsyncMock(return_value='{"status":"ok","items":[]}')
        await run.execute("recall_history", '{"query":"显卡"}', execute)
        await run.execute("recall_history", '{"query":"显卡"}', execute)
        db = await self.manager._get_db()
        await db.execute("UPDATE messages_detail SET content='我换了显卡' WHERE id=1")
        await db.commit()
        await run.execute("recall_history", '{"query":"显卡"}', execute)
        self.assertEqual(execute.await_count, 2)
        await run.finish("complete")


    async def test_keyword_sources_are_fused_without_fact_table_starvation(self):
        db = await self.manager._get_db()
        await db.execute(
            "INSERT INTO messages_detail(session_id,role,name,sender_id,content,timestamp) VALUES(?,?,?,?,?,?)",
            ("group_7", "user", "甲", 101, "[引用回复] 被回复者：RiverFox\n[当前消息] 下楼吃饭吧，R神", "2026-09-11T10:00:00"),
        )
        await db.execute(
            "INSERT INTO summaries(session_id,content,created_at) VALUES(?,?,?)",
            ("group_7", "讨论 R神 的称呼", "2026-09-11T10:00:00"),
        )
        await db.executemany(
            "INSERT INTO important_info(session_id,user_id,info_type,content,created_at) VALUES(?,?,?,?,?)",
            [("group_7", 202, "nickname", "昵称疑为 R神，只是提到过该称呼", "2026-09-11T10:00:00")] * 20,
        )
        await db.commit()
        found = await self.search.search("group_7", mode="keyword", exact_phrase="R神", limit=3)
        self.assertEqual({i["kind"] for i in found["items"]}, {"messages", "summaries", "important_info"})
        self.assertEqual(found["items"][0]["kind"], "messages")
        fact = next(i for i in found["items"] if i["kind"] == "important_info")
        self.assertIsNone(fact["sender_id"])
        self.assertEqual(fact["subjects"], ["202"])
        self.assertEqual(fact["evidence_kind"], "derived_assertion")

    async def test_snapshot_does_not_scan_memory_items_or_physical_vectors(self):
        statements = []
        original = self.search._connect

        @contextmanager
        def traced_connection(**kwargs):
            with original(**kwargs) as db:
                db.set_trace_callback(statements.append)
                yield db

        with patch.object(self.search, "_connect", side_effect=traced_connection) as connection:
            self.search._snapshot("group_7")
        self.assertFalse(any(call.kwargs.get("vectors") for call in connection.call_args_list))
        self.assertFalse(any("memory_items" in sql.lower() or "memory_vec_" in sql.lower() for sql in statements))

    async def test_search_does_not_report_vector_coverage(self):
        vector = struct.pack("<512f", 1, *([0] * 511))
        rows = await asyncio.to_thread(self.index._pending_vectors)
        await asyncio.to_thread(self.index._commit_vectors, rows, [vector] * len(rows))
        with patch("src.plugins.llm_chat.retrieval.embed", AsyncMock(return_value=[vector])):
            for mode in ("keyword", "hybrid", "semantic"):
                with self.subTest(mode=mode):
                    found = await self.search.search("group_7", mode=mode, query="4090")
                    self.assertTrue(found["items"])
                    for field in ("items", "vectors_ready", "vectors_verified"):
                        self.assertNotIn(field, found["coverage"])
                    self.assertNotIn("vector_coverage_incomplete", found["warnings"])

    async def test_interrupted_context_keeps_attempts_without_reusing_answer_as_evidence_or_actions(self):
        from src.plugins.llm_chat.working_context import restore_context

        old = AgentRunner("group_7")
        old.memory_path, old.memory_generation, old.requester_id = self.manager._db_path, 0, 101
        old.tool_calls = 1
        old.transcript = [{"role": "assistant", "tool_calls": [{"function": {
            "name": "query_history", "arguments": '{"sql":"SELECT name FROM messages LIMIT 1"}'
        }}]}]
        old.actions = {"reply": {"status": "sent", "receipt": 123}}
        await old.finish("interrupted", "错误候选")
        new = AgentRunner("group_7")
        new.memory_path, new.memory_generation, new.requester_id = self.manager._db_path, 0, 101
        await restore_context(new)
        self.assertIn("SELECT name", new.transcript[0]["content"])
        note = json.loads(new.transcript[0]["content"].split("\n", 1)[1])[0]
        self.assertEqual(note["status"], "interrupted")
        self.assertEqual(note["answer"], "")
        self.assertFalse(new.actions)
        self.assertFalse(new.evidence)
        for owner, gen in [(202, 0), (101, 1)]:
            other = AgentRunner("group_7")
            other.memory_path, other.memory_generation, other.requester_id = self.manager._db_path, gen, owner
            await restore_context(other)
            self.assertFalse(other.transcript)


class RunnerContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        from src.plugins.llm_chat.llm_logger import drain_usage_tasks
        from src.plugins.llm_chat.usage_db import close_db

        await drain_usage_tasks()
        await close_db()

    async def test_research_deadline_finishes_with_existing_fact_and_reserves_chat_time(self):
        from test_tool_rounds import FakeToolMessage, make_response
        from src.plugins.llm_chat.capabilities import READ_TOOLS

        adapter = models.OpenAICompatibleAdapter("openai:one", "dummy", "https://example.test/v1", "one")
        run = AgentRunner(budget=180)
        run.transcript = [{"role": "user", "content": "已查证：价格为2美元，来源 https://example.org/pricing"}]
        token = current_run.set(run)
        plan = json.dumps({"response_mode": "factual", "verify_group_identity": False,
                           "content": "价格为2美元", "answer_draft": "价格为2美元 https://example.org/pricing"})
        tools.set_image_context(None, 42, bot=object())
        tools.set_voice_context(None, 42, bot=object())
        try:
            with (
                patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}),
                patch.dict("os.environ", {"MODEL_TTS": "google:test-tts"}),
                patch.object(run, "remaining", return_value=40),
                patch.object(adapter, "_create_completion", AsyncMock(return_value=make_response(
                    FakeToolMessage([], content=plan)
                ))) as create,
            ):
                response = await adapter.chat([models.Message("user", "价格是多少")], tools_enabled=True)
                execute = AsyncMock()
                run.planning_active = True
                rejected = await run.execute("search_web", '{"query":"price"}', execute)
                run.planning_active = False
            self.assertEqual(str(response), "价格为2美元")
            self.assertIn("https://example.org/pricing", run.answer_draft)
            request = create.await_args.args[1]
            self.assertLessEqual(request["timeout"], 20)
            self.assertFalse(READ_TOOLS & {t["function"]["name"] for t in request.get("tools", [])})
            self.assertFalse({"generate_image", "send_voice"} & {t["function"]["name"] for t in request.get("tools", [])})
            self.assertEqual(json.loads(rejected)["error"]["code"], "research_budget_exhausted")
            execute.assert_not_awaited()
            self.assertFalse(run.planning_active)
        finally:
            tools.clear_image_context()
            tools.clear_voice_context()
            current_run.reset(token)
            await adapter.client.close()

    async def test_reserved_time_allows_the_chat_stage_to_deliver_the_completed_plan(self):
        from test_tool_rounds import FakeToolMessage, make_response

        adapter = models.OpenAICompatibleAdapter("openai:one", "dummy", "https://example.test/v1", "one")
        manager = models.ModelManager()
        plan = json.dumps({"response_mode": "factual", "verify_group_identity": False,
                           "content": "已有来源支持价格2美元", "answer_draft": "价格2美元"})
        create = AsyncMock(side_effect=[
            make_response(FakeToolMessage([], content=plan)),
            make_response(FakeToolMessage([], content="价格2美元")),
        ])
        try:
            with (
                patch.object(AgentRunner, "remaining", return_value=40),
                patch.object(manager, "_resolve_candidates", return_value=("openai:one", ["openai:one"])),
                patch.object(manager, "get_or_build_adapter", return_value=adapter),
                patch.object(adapter, "_create_completion", create),
                patch.object(tools, "execute_tool", AsyncMock()) as execute,
            ):
                response = await manager.run_turn([models.Message("user", "价格是多少")], role="chat")
            self.assertEqual(str(response), "价格2美元")
            self.assertEqual(create.await_count, 2)
            self.assertEqual(response.agent_run.status, "complete")
            self.assertLessEqual(create.await_args_list[0].args[1]["timeout"], 20)
            self.assertEqual(create.await_args_list[1].args[1]["timeout"], 40)
            execute.assert_not_awaited()
        finally:
            await adapter.client.close()

    async def test_recent_user_links_and_steering_register_without_trusting_bot_sources(self):
        from types import SimpleNamespace
        from src.plugins.llm_chat.web_search import register_url

        run = AgentRunner("group_7")
        token = current_run.set(run)
        try:
            await run.start([
                models.Message("assistant", "https://bot.example/answer"),
                models.Message("user", "历史摘要 https://summary.example/", is_reference=True),
                models.Message("user", "https://user.example/article。", sender_id=101),
                models.Message("user", "读上面这篇", sender_id=101),
            ])
            self.assertEqual({s["url"] for s in run.sources.values()}, {"https://user.example/article"})
            # A source already discovered by search becomes readable when the
            # requester explicitly supplies that same URL in a follow-up.
            source = register_url(run, "https://user.example/new")
            self.assertTrue(agent_runner.steer("group_7", 101, "还有 https://user.example/new"))
            self.assertTrue(run.sources[source]["input_url"])
            backend = SimpleNamespace(model_name="test:fast", _create_completion=AsyncMock(return_value="ok"))
            kwargs = {"messages": [], "tools": [tools.RECALL_HISTORY_TOOL]}
            await run.step(backend, None, kwargs)
            reader = next(t for t in backend._create_completion.await_args.args[1]["tools"]
                          if t["function"]["name"] == "read_web")
            self.assertIn(source, reader["function"]["description"])
        finally:
            await run.finish("complete")
            current_run.reset(token)

    async def test_incomplete_nonstream_answer_and_tool_calls_use_fallback(self):
        from test_tool_rounds import FakeToolMessage, make_response, make_tool_call

        one = models.OpenAICompatibleAdapter("openai:one", "dummy", "https://example.test/v1", "one")
        two = models.OpenAICompatibleAdapter("openai:two", "dummy", "https://example.test/v1", "two")
        manager = models.ModelManager()
        try:
            for reason in ("length", "content_filter"):
                broken = make_response(FakeToolMessage(
                    [make_tool_call("partial", "send_poke", {"target_id": "sender"})], content="不完整的"
                ))
                broken.choices[0].finish_reason = reason
                with (
                    patch.object(manager, "_resolve_candidates", return_value=("openai:one", ["openai:one", "openai:two"])),
                    patch.object(manager, "get_or_build_adapter", side_effect=[one, two]),
                    patch.object(one, "_create_completion", AsyncMock(return_value=broken)),
                    patch.object(two, "_create_completion", AsyncMock(return_value=make_response(
                        FakeToolMessage([], content="完整答案")
                    ))),
                    patch.object(tools, "execute_tool", AsyncMock()) as execute,
                ):
                    response = await manager.chat([models.Message("user", "问题")], role="chat")
                self.assertEqual(str(response), "完整答案")
                execute.assert_not_awaited()
        finally:
            await one.client.close()
            await two.client.close()

    async def test_post_search_model_timeout_preserves_observations(self):
        from types import SimpleNamespace

        async def stalled(*_):
            await asyncio.sleep(1)

        run = AgentRunner()
        run.transcript = [{"role": "tool", "name": "query_history", "content": '{"rows":[["candidate"]]}'}]
        backend = SimpleNamespace(model_name="test:slow", _create_completion=AsyncMock(side_effect=stalled))
        with patch.object(models.config, "agent_search_step_timeout", 0.02), self.assertRaises(TimeoutError):
            await run.step(backend, None, {"messages": list(run.transcript)})
        self.assertIn("candidate", run.transcript[0]["content"])
        # A locally enforced step deadline is labelled separately so the retry
        # policy cannot confuse it with a transient upstream timeout.
        self.assertEqual(run.step_metrics[-1]["status"], "StepTimeout")

    async def test_search_grounding_notice_and_timeout_do_not_affect_initial_or_media_step(self):
        from types import SimpleNamespace

        for tool in (None, "generate_image", "query_history"):
            run = AgentRunner()
            if tool:
                run.transcript = [{"role": "tool", "name": tool, "content": "result"}]
            backend = SimpleNamespace(model_name="test:fast", _create_completion=AsyncMock(return_value="ok"))
            await run.step(backend, None, {"messages": list(run.transcript)})
            sent = backend._create_completion.await_args.args[1]
            if tool == "query_history":
                self.assertLessEqual(sent["timeout"], models.config.agent_search_step_timeout)
                self.assertIn("不能把猜测说成事实", sent["messages"][-1]["content"])
            else:
                self.assertEqual(sent["timeout"], models.config.llm_request_timeout)
                self.assertNotIn("检索核验提示", json.dumps(sent["messages"], ensure_ascii=False))
            self.assertEqual(run.step_metrics[-1]["status"], "ok")

    async def test_reasoning_only_completion_is_never_sent_as_the_answer(self):
        from test_tool_rounds import FakeToolMessage, make_response

        adapter = models.OpenAICompatibleAdapter("openai:one", "dummy", "https://example.test/v1", "one")
        message = FakeToolMessage([], content=None)
        message.reasoning_content = "internal planning text"
        try:
            with (
                patch.object(adapter, "_create_completion", AsyncMock(return_value=make_response(message))),
                self.assertRaisesRegex(RuntimeError, "最终答复"),
            ):
                await adapter.chat([models.Message(role="user", content="测试")], tools_enabled=True)
        finally:
            await adapter.client.close()

    async def test_known_delivery_receipt_survives_failure_after_send(self):
        run = AgentRunner("group_7")
        token = current_run.set(run)

        async def action(*_):
            await agent_runner.record_tool_delivery("sent", {"message_id": 987})
            raise RuntimeError("post_processing_failed")

        try:
            with self.assertRaises(RuntimeError):
                await run.execute("send_poke", '{"target_id":"sender"}', action)
            record = next(iter(run.actions.values()))
            self.assertEqual(record["status"], "sent")
            self.assertEqual(record["receipt"], {"message_id": 987})
            replay = AsyncMock()
            await run.execute("send_poke", '{"target_id":"sender"}', replay)
            replay.assert_not_awaited()
        finally:
            current_run.reset(token)

    async def test_budget_rejects_extra_calls_without_counting_them_as_executed(self):
        run = AgentRunner()
        run.tool_calls = memory_module.config.agent_max_tool_calls
        execute = AsyncMock()
        for _ in range(3):
            result = await run.execute("recall_history", "{}", execute)
            self.assertEqual(json.loads(result)["error"]["code"], "tool_budget_exhausted")
        self.assertEqual(run.tool_calls, memory_module.config.agent_max_tool_calls)
        self.assertEqual(run.rejected_tool_calls, 3)
        execute.assert_not_awaited()




    async def test_ambiguous_poke_does_not_try_alternate_actions(self):
        from types import SimpleNamespace

        bot = SimpleNamespace(call_api=AsyncMock(side_effect=TimeoutError()), send_group_msg=AsyncMock())
        token = tools._poke_context.set({"bot": bot, "group_id": 7, "user_id": 101})
        try:
            answer = await tools.send_poke("sender")
            self.assertIn("结果未知", answer)
            self.assertEqual(bot.call_api.await_count, 1)
            bot.send_group_msg.assert_not_awaited()
        finally:
            tools._poke_context.reset(token)

    async def test_provider_fallback_receives_completed_tool_observation(self):
        from test_tool_rounds import FakeToolMessage, make_response, make_tool_call

        one = models.OpenAICompatibleAdapter("openai:one", "dummy", "https://example.test/v1", "one")
        two = models.OpenAICompatibleAdapter("openai:two", "dummy", "https://example.test/v1", "two")
        first = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([make_tool_call("a", "recall_history", {"query": "型号"})])),
                RuntimeError("provider_failed"),
            ]
        )
        second = AsyncMock(return_value=make_response(FakeToolMessage([], content="已核对")))
        manager = models.ModelManager()
        with (
            patch.object(manager, "_resolve_candidates", return_value=("openai:one", ["openai:one", "openai:two"])),
            patch.object(manager, "get_or_build_adapter", side_effect=[one, two, two]),
            patch.object(one, "_create_completion", first),
            patch.object(two, "_create_completion", second),
            patch.object(
                tools, "execute_tool", AsyncMock(return_value='{"status":"ok","items":[],"fact":"确认为 RTX 4090"}')
            ),
        ):
            response = await manager.run_turn([models.Message(role="user", content="查一下型号")], role="chat")
        self.assertEqual(str(response), "已核对")
        sent = second.await_args_list[0].args[1]["messages"]
        self.assertIn("RTX 4090", json.dumps(sent, ensure_ascii=False))
        self.assertEqual(response.agent_run.tool_round, 1)
        await one.client.close()
        await two.client.close()

    async def test_only_current_requester_can_steer_active_run(self):
        runner = AgentRunner("group_7")
        runner.requester_id = 101
        agent_runner.active_runs["group_7"] = runner
        try:
            self.assertFalse(agent_runner.steer("group_7", 202, "改成别人的问题"))
            self.assertTrue(agent_runner.steer("group_7", 101, "我说的是去年"))
            self.assertEqual(runner.steering, ["我说的是去年"])
        finally:
            agent_runner.active_runs.pop("group_7", None)

    async def test_malformed_arguments_never_execute(self):
        runner = AgentRunner()
        execute = AsyncMock()
        for raw in ('{"query":', "[]", "null"):
            response = await runner.execute("recall_history", raw, execute)
            self.assertEqual(json.loads(response)["status"], "invalid_arguments")
        execute.assert_not_awaited()

    async def test_read_cache_and_changed_focus(self):
        runner = AgentRunner()
        execute = AsyncMock(return_value='{"status":"ok","items":[]}')
        await runner.execute("read_web", '{"source_id":"w:a","focus":"甲"}', execute)
        await runner.execute("read_web", '{"source_id":"w:a","focus":"甲"}', execute)
        await runner.execute("read_web", '{"source_id":"w:a","focus":"乙"}', execute)
        self.assertEqual(execute.await_count, 2)

    async def test_tool_timeout_preserves_failure_type(self):
        runner = AgentRunner(budget=1.01)

        async def slow(*_):
            await asyncio.sleep(1)

        found = await runner.execute("recall_history", "{}", slow)
        self.assertEqual(json.loads(found)["error"]["code"], "tool_timeout")

    async def test_web_read_cannot_open_another_run_source(self):
        from src.plugins.llm_chat.web_search import read, register_url

        first = AgentRunner("group_7")
        source_id = register_url(first, "https://example.com")
        token = current_run.set(AgentRunner("group_8"))
        try:
            self.assertEqual((await read(source_id))["status"], "invalid_arguments")
        finally:
            current_run.reset(token)


if __name__ == "__main__":
    unittest.main()
