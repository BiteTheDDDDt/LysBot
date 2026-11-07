import asyncio
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import nonebot

nonebot.init()

from src.plugins.llm_chat import memory as memory_module
from src.plugins.llm_chat.conversation import session_scope
from src.plugins.llm_chat.memory import MemoryManager


class IncomingMemoryTests(unittest.IsolatedAsyncioTestCase):
    group_id = 17017
    session_id = "group_17017"

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous_instance = MemoryManager._instance
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.tmp.name) / "history.db"
        self.manager._closing = True
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(memory_module.config, "enable_summary", False))
        self.patches.enter_context(patch.object(memory_module.config, "enable_important_info", False))
        self.db = await self.manager._get_db()

    async def asyncTearDown(self):
        await self.manager.close()
        self.patches.close()
        MemoryManager._instance = self.previous_instance
        self.tmp.cleanup()

    async def add(self, content, timestamp="2026-10-17T10:00:00"):
        return await self.manager.add_message(
            101, "user", content, self.group_id, sender_id=101, name="测试成员", timestamp=timestamp,
        )

    async def source_rows(self):
        async with self.db.execute(
            "SELECT id,content,timestamp FROM messages_detail WHERE session_id=? ORDER BY id", (self.session_id,)
        ) as cursor:
            return await cursor.fetchall()

    async def stored_context(self):
        async with self.db.execute(
            "SELECT messages FROM sessions WHERE session_id=?", (self.session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return json.loads(row[0]) if row else []

    async def revision(self):
        async with self.db.execute(
            "SELECT value FROM kv_meta WHERE key=?", ("memory_revision:" + self.session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def jobs(self):
        async with self.db.execute(
            "SELECT source_id,revision,attempts,retry_at,priority FROM memory_index_jobs "
            "WHERE session_id=? AND kind='messages' ORDER BY id", (self.session_id,)
        ) as cursor:
            return await cursor.fetchall()

    async def test_enrichment_preserves_original_order_ids_and_timestamps(self):
        first = await self.add("A：图片待识别")
        second = await self.add("B：稍后到达的消息", "2026-10-17T10:00:01")
        identities = [(row[0], row[2]) for row in await self.source_rows()]

        changed = await self.manager.enrich_message(101, first, "A：图片是一只猫", self.group_id)

        self.assertTrue(changed)
        rows = await self.source_rows()
        self.assertEqual([(row[0], row[2]) for row in rows], identities)
        self.assertEqual([row[1] for row in rows], ["A：图片是一只猫", "B：稍后到达的消息"])
        self.assertEqual([row[0] for row in rows], [first.source_id, second.source_id])
        history = await self.manager.get_history(101, self.group_id)
        self.assertEqual([message.content for message in history], [row[1] for row in rows])
        self.assertEqual([message.source_id for message in history], [first.source_id, second.source_id])
        stored = await self.stored_context()
        self.assertEqual([(item["source_id"], item["timestamp"]) for item in stored], identities)
        self.assertEqual([item["content"] for item in stored], [row[1] for row in rows])
        self.assertEqual(first.content, "A：图片是一只猫")

    async def test_enrichment_advances_revision_and_requeues_the_existing_index_block(self):
        message = await self.add("图片待识别")
        await self.db.execute(
            "UPDATE memory_index_jobs SET attempts=7,retry_at=9999999999,priority=10 WHERE session_id=?",
            (self.session_id,),
        )
        await self.db.commit()
        old_revision = await self.revision()
        old_jobs = await self.jobs()

        self.assertTrue(await self.manager.enrich_message(101, message, "图片是一只猫", self.group_id))

        self.assertEqual(await self.revision(), old_revision + 1)
        new_jobs = await self.jobs()
        self.assertEqual(len(new_jobs), 1)
        self.assertEqual(new_jobs[0], (old_jobs[0][0], old_jobs[0][1] + 1, 0, 0.0, 0))

    async def test_full_clear_rejects_late_enrichment_and_add_from_the_old_scope(self):
        captured, resume = asyncio.Event(), asyncio.Event()

        async def delayed_image_analysis():
            with session_scope(self.session_id):
                message = await self.add("旧图片待识别")
                captured.set()
                await resume.wait()
                # Nested work must retain the obsolete generation.
                with session_scope(self.session_id):
                    enriched = await self.manager.enrich_message(101, message, "旧图片描述", self.group_id)
                    added = await self.add("旧任务尝试重新保存")
                return enriched, added

        task = asyncio.create_task(delayed_image_analysis())
        try:
            await asyncio.wait_for(captured.wait(), timeout=5)
            await self.manager.clear_history(101, self.group_id, full=True)
            resume.set()
            enriched, added = await asyncio.wait_for(task, timeout=5)
        finally:
            resume.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.assertFalse(enriched)
        self.assertIsNone(added)
        self.assertEqual(await self.source_rows(), [])
        self.assertEqual(await self.stored_context(), [])
        self.assertEqual(await self.manager.get_history(101, self.group_id), [])
        self.assertEqual(await self.jobs(), [])
        with session_scope(self.session_id):
            fresh = await self.add("清空后的新消息")
        self.assertIsNotNone(fresh)
        self.assertEqual([row[1] for row in await self.source_rows()], ["清空后的新消息"])

    async def test_failed_enrichment_rolls_back_source_context_and_index_changes(self):
        message = await self.add("原图占位")
        old_rows, old_context = await self.source_rows(), await self.stored_context()
        old_revision, old_jobs = await self.revision(), await self.jobs()
        execute = self.db.execute

        for statement in ("UPDATE messages_detail", "UPDATE sessions"):
            with self.subTest(failed_statement=statement):
                def fail_update(sql, parameters=(), *, failing_statement=statement):
                    if sql.lstrip().startswith(failing_statement):
                        raise aiosqlite.OperationalError("synthetic update failure")
                    return execute(sql, parameters)

                with (
                    patch.object(self.db, "execute", side_effect=fail_update),
                    self.assertRaisesRegex(aiosqlite.OperationalError, "synthetic update failure"),
                ):
                    await self.manager.enrich_message(101, message, "不应提交的图片描述", self.group_id)

                self.assertEqual(await self.source_rows(), old_rows)
                self.assertEqual(await self.stored_context(), old_context)
                self.assertEqual(await self.revision(), old_revision)
                self.assertEqual(await self.jobs(), old_jobs)
                self.assertEqual(message.content, "原图占位")
                history = await self.manager.get_history(101, self.group_id)
                self.assertEqual([item.content for item in history], ["原图占位"])

    async def test_enriching_a_compressed_message_does_not_append_it_to_recent_context(self):
        first = await self.add("A：图片待识别")
        second = await self.add("B：保留在近期上下文", "2026-10-17T10:00:01")
        with (
            patch.object(memory_module.config, "enable_summary", True),
            patch.object(memory_module.config, "summary_threshold", 1),
            patch.object(memory_module.config, "summary_messages", 1),
            patch.object(memory_module.config, "max_history_messages", 1),
            patch.object(memory_module.model_manager, "chat", AsyncMock(
                return_value='{"summary":"收到图片","memory_candidates":[]}'
            )),
        ):
            await self.manager._compress_history(self.session_id)
        self.assertEqual([message.source_id for message in self.manager._cache[self.session_id]], [second.source_id])

        self.assertTrue(await self.manager.enrich_message(101, first, "A：图片是一只猫", self.group_id))

        self.assertEqual([message.source_id for message in self.manager._cache[self.session_id]], [second.source_id])
        self.assertEqual([item["source_id"] for item in await self.stored_context()], [second.source_id])
        self.assertEqual(
            [(row[0], row[1]) for row in await self.source_rows()],
            [(first.source_id, "A：图片是一只猫"), (second.source_id, "B：保留在近期上下文")],
        )


if __name__ == "__main__":
    unittest.main()
