import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import nonebot

nonebot.init()

from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.memory_changes import change, inspect
from src.plugins.llm_chat.memory_index import connect, MemoryIndex
from src.plugins.llm_chat.history_sql import query_history_sql
from src.plugins.llm_chat.retrieval import HistorySearch
from src.plugins.llm_chat.agent_runner import AgentRunner, current_run
from src.plugins.llm_chat.models import Message
from src.plugins.llm_chat.memory_trace import attach_refs
from src.plugins.llm_chat.working_context import restore_context
from src.plugins.llm_chat import tools


class MemoryChangesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._closing = True
        self.path = self.manager._db_path = Path(self.tmp.name) / "memory.db"
        db = await self.manager._get_db()
        await db.executemany(
            "INSERT INTO messages_detail(session_id,role,sender_id,content,timestamp) VALUES(?,?,?,?,'2026-10-17T12:00:00')",
            [("group_7", "user", 101, "他说喜欢咖啡"), ("group_7", "user", 202, "大家好")],
        )
        await db.execute(
            "INSERT INTO important_info(session_id,user_id,info_type,content,provenance_json,created_at) VALUES(?,?,?,?,?,'2026-10-17T12:00:00')",
            (
                "group_7",
                101,
                "user_preference",
                "喜欢咖啡",
                json.dumps({"source_ids": [1], "assertion": "asserted", "provenance": "derived"}),
            ),
        )
        await db.execute(
            "INSERT INTO summaries(session_id,content,created_at) VALUES('group_7','甲喜欢咖啡','2026-10-17T12:00:00')"
        )
        await db.commit()

    async def asyncTearDown(self):
        await self.manager.close()
        MemoryManager._instance = None
        tools.clear_poke_context()
        self.tmp.cleanup()

    def preview(self, **kwargs):
        return change(
            self.path, "group_7", 101, action="preview", ref="h:0:important_info:1", reason="原来是转述", **kwargs
        )

    def apply(self, preview):
        return change(
            self.path,
            "group_7",
            101,
            action="apply",
            operation_id=preview["operation_id"],
            confirmation=preview["confirm_text"],
        )

    async def test_attribution_preview_apply_and_undo_preserve_original(self):
        preview = self.preview(owner_id=202)
        before = await HistorySearch(self.path).profile("group_7", [101])
        self.assertIn("咖啡", before)
        self.assertEqual(json.loads(preview["after"]["provenance_json"])["assertion"], "reported")
        applied = self.apply(preview)
        self.assertNotIn("咖啡", await HistorySearch(self.path).profile("group_7", [101]))
        self.assertIn("转述", await HistorySearch(self.path).profile("group_7", [202]))
        self.assertEqual((await self.manager.get_fixed_context_memory("group_7", [101]))["summaries"], [])
        sql = await query_history_sql(self.path, "group_7", "SELECT user_id,content FROM facts")
        self.assertEqual(sql["rows"], [["202", "喜欢咖啡"]])
        with connect(self.path) as db:
            self.assertEqual(
                tuple(db.execute("SELECT sender_id,content FROM messages_detail WHERE id=1").fetchone()),
                (101, "他说喜欢咖啡"),
            )
        undone = change(
            self.path,
            "group_7",
            101,
            action="undo",
            operation_id=preview["operation_id"],
            confirmation=applied["undo_text"],
        )
        self.assertEqual(undone["status"], "undone")
        self.assertIn("咖啡", await HistorySearch(self.path).profile("group_7", [101]))
        self.assertEqual(self.apply_error(preview), "operation_state_conflict")

    def apply_error(self, preview):
        try:
            self.apply(preview)
        except ValueError as error:
            return str(error)
        self.fail("expected conflict")

    async def test_confirmation_scope_permissions_and_conflicting_undo(self):
        p = self.preview(content="喜欢茶")
        with self.assertRaisesRegex(ValueError, "user_must_send"):
            change(self.path, "group_7", 101, action="apply", operation_id=p["operation_id"], confirmation="确认")
        with self.assertRaises(ValueError):
            change(
                self.path,
                "group_8",
                101,
                action="apply",
                operation_id=p["operation_id"],
                confirmation=p["confirm_text"],
            )
        with self.assertRaises(ValueError):
            change(self.path, "group_7", 202, action="preview", ref="h:0:important_info:1", content="x", reason="x")
        self.apply(p)
        self.assertTrue(self.apply(p)["already_done"])
        p2 = self.preview(content="喜欢牛奶")
        self.apply(p2)
        with self.assertRaisesRegex(ValueError, "memory_changed"):
            change(
                self.path,
                "group_7",
                101,
                action="undo",
                operation_id=p["operation_id"],
                confirmation="撤销记忆 " + p["operation_id"],
            )

    async def test_withdrawal_hides_all_search_paths_and_stale_projection(self):
        index = MemoryIndex(self.path)
        while await asyncio.to_thread(index._project_one):
            pass
        self.apply(self.preview(assertion="withdrawn"))
        search = HistorySearch(self.path)
        found = await search.search("group_7", query="咖啡", mode="keyword", sources=["important_info", "summaries"])
        self.assertEqual(found["items"], [])
        self.assertNotEqual((await search.read("group_7", ids=["h:0:important_info:1"]))["status"], "ok")
        for table in ("facts", "summaries"):
            self.assertEqual((await query_history_sql(self.path, "group_7", f"SELECT * FROM {table}"))["rows"], [])
        self.assertEqual((await self.manager.get_fixed_context_memory("group_7", [101]))["important_info"], [])
        self.assertTrue(inspect(self.path, "group_7", owner_id=101)["memories"])

    async def test_trace_survives_checkpoint_and_correction_blocks_old_working_context(self):
        messages = [
            Message("user", "喜欢咖啡 [h:0:important_info:1]", is_reference=True),
            Message("user", "你觉得我喜欢啥", sender_id=101),
        ]
        await attach_refs(self.path, "group_7", messages)
        run = AgentRunner("group_7")
        run.memory_path = self.path
        await run.start(messages)
        run.planning_result = "他喜欢咖啡"
        await run.finish("complete", "咖啡")
        records = inspect(self.path, "group_7", run_id=run.id)
        self.assertEqual(records["injected_memories"][0]["id"], "h:0:important_info:1")
        self.assertTrue(records["injected_memories"][0]["version"])
        self.apply(self.preview(assertion="withdrawn"))
        following = AgentRunner("group_7")
        following.memory_path, following.memory_generation, following.requester_id = self.path, 0, 101
        await restore_context(following)
        self.assertEqual(following.transcript, [])
        from src.plugins.llm_chat.conversation import TurnSuperseded

        with self.assertRaises(TurnSuperseded):
            await run.validate_scope()

    async def test_tool_confirmation_cannot_be_fabricated_by_model(self):
        run = AgentRunner("group_7")
        run.memory_path, run.memory_generation, run.requester_id = self.path, 0, 101
        token = current_run.set(run)
        tools.set_poke_context(7, 101, current_user_text="还没确认")
        try:
            p = json.loads(
                await tools.revise_memory("preview", ref="h:0:important_info:1", content="喜欢茶", reason="本人纠正")
            )
            rejected = json.loads(await tools.revise_memory("apply", operation_id=p["operation_id"]))
            self.assertEqual(rejected["status"], "invalid_arguments")
            tools.set_poke_context(7, 101, current_user_text=p["confirm_text"])
            applied = json.loads(await tools.revise_memory("apply", operation_id=p["operation_id"]))
            self.assertEqual(applied["status"], "applied")
            await run.validate_scope()
        finally:
            current_run.reset(token)

    async def test_clear_history_erases_change_snapshots(self):
        self.apply(self.preview(content="喜欢茶"))
        await self.manager.clear_history(101, 7, full=True)
        with connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM memory_changes").fetchone()[0], 0)
