import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import memory as memory_module
from src.plugins.llm_chat import models, tools
from src.plugins.llm_chat.memory import MemoryManager


class MemoryRecallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.temp_dir.name) / "history.db"
        self.manager._closing = True
        await self.manager._get_db()

        with patch.object(memory_module.config, "enable_important_info", False):
            await self.manager.add_message(
                0,
                "user",
                "我喜欢蓝色汉服",
                7,
                name="甲",
                sender_id=101,
            )
            await self.manager.add_message(
                0,
                "assistant",
                "【生成图片 model=test】提示词：蓝色汉服自拍；实际画面：窗边的磷叶石。",
                7,
            )
            await self.manager.add_message(
                0,
                "user",
                "我喜欢红色外套",
                7,
                name="乙",
                sender_id=202,
            )

        db = await self.manager._get_db()
        await db.execute(
            """
            INSERT INTO summaries(session_id, content, time_range, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("group_7", "上周讨论过适合自拍的服装。", "2026-07-01~2026-07-07", "2026-07-08T00:00:00"),
        )
        await db.executemany(
            """
            INSERT INTO important_info(session_id, user_id, info_type, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("group_7", 0, "key_event", "群里计划制作角色合影。", "2026-07-08T00:00:00"),
                ("group_7", 101, "user_preference", "甲偏好蓝色汉服。", "2026-07-08T00:00:01"),
                ("group_7", 202, "user_preference", "乙偏好红色外套。", "2026-07-08T00:00:02"),
            ],
        )
        await db.commit()

    async def asyncTearDown(self):
        await self.manager.close()
        MemoryManager._instance = None
        self.temp_dir.cleanup()

    async def test_sender_id_is_persisted_and_filters_message_recall(self):
        db = await self.manager._get_db()
        async with db.execute("PRAGMA table_info(messages_detail)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        self.assertIn("sender_id", columns)

        result = await self.manager.recall_memory(
            session_id="group_7",
            query="蓝色 汉服",
            sources=["messages"],
            sender_ids=[101],
            limit=10,
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["hits"][0]["sender_id"], 101)
        self.assertEqual(result["hits"][0]["source"], "messages")

    async def test_generated_media_is_a_separate_recall_source(self):
        result = await self.manager.recall_memory(
            session_id="group_7",
            query="自拍",
            sources=["generated_media"],
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["hits"][0]["source"], "generated_media")
        self.assertIn("实际画面", result["hits"][0]["content"])

    async def test_fixed_context_contains_only_active_participant_facts(self):
        with (
            patch.object(memory_module.config, "fixed_context_max_participants", 5),
            patch.object(memory_module.config, "fixed_context_info_per_type", 1),
            patch.object(memory_module.config, "fixed_context_max_info", 20),
            patch.object(memory_module.config, "recent_summaries", 2),
        ):
            result = await self.manager.get_fixed_context_memory("group_7", [101])

        contents = [item["content"] for item in result["important_info"]]
        self.assertIn("群里计划制作角色合影。", contents)
        self.assertIn("甲偏好蓝色汉服。", contents)
        self.assertNotIn("乙偏好红色外套。", contents)
        self.assertEqual(len(result["summaries"]), 1)

    async def test_recall_tool_returns_structured_results_for_iterative_search(self):
        token = models.current_session_id.set("group_7")
        try:
            with patch.object(memory_module, "memory_manager", self.manager):
                raw = await tools.recall_history(
                    query="汉服",
                    sources=["important_info", "summaries"],
                    sender_ids=["101"],
                    match_mode="any",
                )
        finally:
            models.current_session_id.reset(token)

        result = json.loads(raw)
        self.assertEqual(result["session_id"], "group_7")
        self.assertTrue(result["hits"])
        self.assertTrue(
            all(hit["source"] in {"important_info", "summaries"} for hit in result["hits"])
        )
        self.assertTrue(all("id" in hit for hit in result["hits"]))


if __name__ == "__main__":
    unittest.main()
