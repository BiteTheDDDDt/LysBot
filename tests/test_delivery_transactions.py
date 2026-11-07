import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import nonebot

nonebot.init()

from src.plugins.llm_chat import conversation
from src.plugins.llm_chat.memory import MemoryManager


class DeliveryTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = object.__new__(MemoryManager)
        self.manager._init()
        self.path = Path(self.temp.name) / "history.db"
        self.manager._db_path = self.path
        self.db = await self.manager._get_db()
        conversation.start()

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    def visible_rows(self, table):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def save(self, table):
        if table == "daily_briefings":
            return self.manager.save_published_briefing(
                71, {"summary": "old content"}, sent_as="image", briefing_date="2026-10-17",
            )
        return self.manager.save_news_broadcast(1, 71, {"items": ["old content"]}, sent_as="image")

    async def prepare_news(self):
        await self.manager.save_published_briefing(
            71, {"summary": "fixture"}, sent_as="image", briefing_date="2026-10-16",
        )

    async def assert_cancel_rolls_back(self, table, cancel_during):
        entered = asyncio.Event()
        execute = self.db.execute

        async def pause_after_insert(sql, parameters=()):
            cursor = await execute(sql, parameters)
            if f"INSERT INTO {table}" in sql:
                entered.set()
                await asyncio.Event().wait()
            return cursor

        async def pause_before_commit():
            entered.set()
            await asyncio.Event().wait()

        attribute, replacement = ("execute", pause_after_insert) if cancel_during == "execute" else ("commit", pause_before_commit)
        with patch.object(self.db, attribute, side_effect=replacement):
            task = asyncio.create_task(self.save(table))
            try:
                await asyncio.wait_for(entered.wait(), timeout=3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=3)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertFalse(self.db.in_transaction)
        self.assertFalse(self.manager._write_lock.locked())
        self.assertEqual(self.visible_rows(table), 0)
        # An unrelated transaction on the shared connection must not publish the cancelled write.
        await self.db.execute("INSERT OR REPLACE INTO kv_meta(key,value) VALUES('unrelated', '1')")
        await self.db.commit()
        self.assertEqual(self.visible_rows(table), 0)

    async def test_cancelled_briefing_insert_and_commit_roll_back(self):
        for point in ("execute", "commit"):
            with self.subTest(point=point):
                await self.assert_cancel_rolls_back("daily_briefings", point)

    async def test_cancelled_news_insert_and_commit_roll_back(self):
        await self.prepare_news()
        for point in ("execute", "commit"):
            with self.subTest(point=point):
                await self.assert_cancel_rolls_back("news_broadcasts", point)

    async def test_generation_is_checked_after_acquiring_write_lock(self):
        await self.prepare_news()
        for table in ("daily_briefings", "news_broadcasts"):
            with self.subTest(table=table):
                before = self.visible_rows(table)
                entered = asyncio.Event()
                async def old_task():
                    with conversation.session_scope("group_71"):
                        entered.set()
                        await self.save(table)
                await self.manager._write_lock.acquire()
                task = asyncio.create_task(old_task())
                try:
                    await asyncio.wait_for(entered.wait(), timeout=3)
                    conversation.invalidate("group_71", forget=True)
                finally:
                    self.manager._write_lock.release()
                await asyncio.wait_for(task, timeout=3)
                self.assertEqual(self.visible_rows(table), before)

    async def test_database_lock_errors_still_retry(self):
        execute = self.db.execute
        for table in ("daily_briefings", "news_broadcasts"):
            attempts = 0
            async def locked_once(sql, parameters=()):
                nonlocal attempts
                if f"INSERT INTO {table}" in sql:
                    attempts += 1
                    if attempts == 1:
                        raise aiosqlite.OperationalError("database is locked")
                return await execute(sql, parameters)
            with self.subTest(table=table), patch.object(self.db, "execute", side_effect=locked_once):
                await self.save(table)
            self.assertEqual(attempts, 2)
            self.assertEqual(self.visible_rows(table), 1)


if __name__ == "__main__":
    unittest.main()
