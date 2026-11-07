import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import models as models_module
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.memory import MemoryManager, _parse_important_info_entries
from src.plugins.llm_chat.models import Message, ModelManager


class ImportantInfoQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.temp_dir.name) / "history.db"
        self.manager._closing = True
        await self.manager._get_db()

    async def asyncTearDown(self):
        await self.manager.close()
        MemoryManager._instance = None
        self.temp_dir.cleanup()

    async def _enqueue_and_load(self) -> dict:
        messages = [
            Message(role="user", content="我喜欢吃火锅", timestamp="2026-07-10T20:00:00"),
            Message(role="assistant", content="记住了", timestamp="2026-07-10T20:00:01"),
        ]
        await self.manager._enqueue_important_info_job(
            "group_1",
            messages,
            user_id=0,
            start_idx=1,
            end_idx=2,
        )
        job, _ = await self.manager._load_next_important_info_job()
        self.assertIsNotNone(job)
        return job

    async def test_summary_selects_original_candidates_or_skips_extraction(self):
        for group, candidates, expected in [(71, [], []), (72, [3], ["原文1", "原文2", "原文3"])]:
            for i in range(6):
                await self.manager.add_message(0, "user", f"原文{i}", group_id=group, sender_id=42)
            with (
                patch.object(config, "summary_threshold", 4),
                patch.object(config, "summary_messages", 5),
                patch.object(config, "max_history_messages", 1),
                patch.object(config, "enable_important_info", True),
                patch.object(models_module.model_manager, "chat", AsyncMock(return_value=json.dumps(
                    {"summary":"摘要", "memory_candidates":candidates}))),
            ):
                await self.manager._compress_history(f"group_{group}")
            db = await self.manager._get_db()
            async with db.execute("SELECT messages_json FROM important_info_jobs WHERE session_id=?", (f"group_{group}",)) as cursor:
                rows = await cursor.fetchall()
            actual = [m["content"] for m in json.loads(rows[0][0])] if rows else []
            self.assertEqual(actual, expected)
            async with db.execute("SELECT count(*) FROM messages_detail WHERE session_id=?", (f"group_{group}",)) as cursor:
                self.assertEqual((await cursor.fetchone())[0], 6)

    def test_summary_retry_backs_off_with_a_ceiling(self):
        with patch("src.plugins.llm_chat.memory.time.monotonic", return_value=100):
            for expected in [60, 120, 240, 480, 960, 1920, 3600, 3600]:
                self.manager._defer_summary("group_1")
                self.assertEqual(self.manager._summary_retry_at["group_1"], 100 + expected)

    async def test_failure_keeps_persisted_job_for_retry(self):
        job = await self._enqueue_and_load()
        fake_chat = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(models_module.model_manager, "chat", fake_chat):
            await self.manager._process_important_info_job(job)

        db = await self.manager._get_db()
        async with db.execute(
            "SELECT attempts, last_error FROM important_info_jobs WHERE id = ?",
            (job["id"],),
        ) as cursor:
            row = await cursor.fetchone()

        self.assertEqual(row[0], 1)
        self.assertIn("TimeoutError", row[1])
        self.assertTrue(fake_chat.await_args.kwargs["allow_fallback"])

    async def test_max_attempts_moves_job_to_dead_letter(self):
        job = await self._enqueue_and_load()
        job["attempts"] = config.important_info_max_attempts - 1

        await self.manager._defer_important_info_job(job, RuntimeError("still broken"))

        db = await self.manager._get_db()
        async with db.execute(
            "SELECT attempts, dead_letter_at FROM important_info_jobs WHERE id = ?",
            (job["id"],),
        ) as cursor:
            row = await cursor.fetchone()
        next_job, _ = await self.manager._load_next_important_info_job()

        self.assertEqual(row[0], config.important_info_max_attempts)
        self.assertIsNotNone(row[1])
        self.assertIsNone(next_job)

    async def test_success_writes_entries_and_deletes_job_atomically(self):
        job = await self._enqueue_and_load()
        fake_chat = AsyncMock(return_value="user_preference:用户喜欢吃火锅")

        with patch.object(models_module.model_manager, "chat", fake_chat):
            await self.manager._process_important_info_job(job)

        db = await self.manager._get_db()
        async with db.execute("SELECT COUNT(*) FROM important_info_jobs") as cursor:
            pending_count = (await cursor.fetchone())[0]
        async with db.execute(
            "SELECT info_type, content FROM important_info WHERE session_id = 'group_1'"
        ) as cursor:
            info_row = await cursor.fetchone()

        self.assertEqual(pending_count, 0)
        self.assertEqual(info_row, ("user_preference", "用户喜欢吃火锅"))

    async def test_pending_job_survives_manager_restart(self):
        original_job = await self._enqueue_and_load()
        db_path = self.manager._db_path
        await self.manager.close()

        MemoryManager._instance = None
        restarted_manager = MemoryManager()
        restarted_manager._db_path = db_path
        restarted_manager._closing = True
        await restarted_manager._get_db()
        self.manager = restarted_manager

        recovered_job, _ = await restarted_manager._load_next_important_info_job()
        self.assertIsNotNone(recovered_job)
        self.assertEqual(recovered_job["job_key"], original_job["job_key"])

    async def test_existing_job_table_is_migrated_for_dead_letters(self):
        legacy_db_path = Path(self.temp_dir.name) / "legacy-history.db"
        db = sqlite3.connect(legacy_db_path)
        try:
            db.execute(
                """
                CREATE TABLE important_info_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_key TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    messages_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT NOT NULL,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            db.commit()
        finally:
            db.close()

        await self.manager.close()
        MemoryManager._instance = None
        migrated_manager = MemoryManager()
        migrated_manager._db_path = legacy_db_path
        migrated_manager._closing = True
        await migrated_manager._get_db()
        self.manager = migrated_manager

        db = await migrated_manager._get_db()
        async with db.execute("PRAGMA table_info(important_info_jobs)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}

        self.assertIn("dead_letter_at", columns)


class ModelFallbackTests(unittest.TestCase):
    def test_fallback_can_be_disabled_per_call(self):
        manager = ModelManager()
        with (
            patch.object(
                models_module,
                "resolve_role",
                return_value=("nvidia:primary", ["google:fallback"]),
            ),
            patch.object(models_module, "provider_configured", return_value=True),
        ):
            primary, candidates = manager._resolve_candidates(
                role="important_info",
                allow_fallback=False,
            )

        self.assertEqual(primary, "nvidia:primary")
        self.assertEqual(candidates, ["nvidia:primary"])


class ImportantInfoParsingTests(unittest.TestCase):
    def test_accepts_subject_suffix_and_multiple_entries_on_one_line(self):
        result = (
            "user_preference（群员-Wqr_）：认为 Fable 效果更好。   "
            "key_event（群员-爱做梦的crud）：有一位表姐是 T。"
        )

        entries = _parse_important_info_entries(result, config.important_info_types)

        self.assertEqual(
            entries,
            [
                ("user_preference", "群员-Wqr_：认为 Fable 效果更好。"),
                ("key_event", "群员-爱做梦的crud：有一位表姐是 T。"),
            ],
        )

    def test_keeps_original_strict_format_compatible(self):
        entries = _parse_important_info_entries(
            "- user_identity:用户是数据库工程师\n- emotion_state：用户最近有些焦虑",
            config.important_info_types,
        )

        self.assertEqual(
            entries,
            [
                ("user_identity", "用户是数据库工程师"),
                ("emotion_state", "用户最近有些焦虑"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
