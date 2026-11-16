import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import aiosqlite
import nonebot

nonebot.init()

from src.plugins.llm_chat import conversation
from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.memory_schema import initialize_schema


class MemoryRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = object.__new__(MemoryManager)
        self.manager._init()
        self.manager._db_path = Path(self.temp.name) / "history.db"
        conversation.start()

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    async def test_repositories_follow_connection_reopening(self):
        # Construction must not open SQLite or start another connection owner.
        self.assertIsNone(self.manager._db)
        when = datetime(2026, 1, 2, 12)
        await self.manager.save_published_briefing(
            71, {"topics": ["fixture"]}, sent_as="image", published_at=when,
        )
        db = await self.manager._get_db()
        await db.execute(
            "INSERT INTO messages_detail(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("group_71", "user", "fixture", when.isoformat()),
        )
        await db.commit()
        await self.manager.close()
        # Repositories must resolve the current connection, not retain the closed one.
        hits = await self.manager.search_messages_detail("group_71", keyword="fixture")
        self.assertEqual(len(hits), 1)
        self.assertIsNot(await self.manager._get_db(), db)
        card = await self.manager.get_latest_published_briefing(71, published_after=datetime(2026, 1, 1))
        self.assertEqual(card["card_data"], {"topics": ["fixture"]})

    async def test_query_context_and_digest_reads_remain_group_scoped(self):
        db = await self.manager._get_db()
        await db.executemany(
            "INSERT INTO messages_detail(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [("group_71", "user", "before", "2026-01-02T11:00:00"),
             ("group_72", "user", "private fixture", "2026-01-02T11:30:00"),
             ("group_71", "user", "fixture", "2026-01-02T12:00:00"),
             ("group_72", "user", "private after", "2026-01-02T12:30:00"),
             ("group_71", "user", "after", "2026-01-02T13:00:00")],
        )
        await db.commit()
        recalled = await self.manager.recall_memory(
            session_id="group_71", query="fixture", sources=["messages"],
            context_before=2, context_after=2,
        )
        self.assertEqual(recalled["count"], 1)
        self.assertEqual([row["content"] for row in recalled["hits"][0]["context"]], ["before", "after"])
        messages, count = await self.manager.get_digest_messages(
            "group_71", datetime(2026, 1, 2), datetime(2026, 1, 3),
        )
        self.assertEqual(count, 3)
        self.assertEqual([message.content for message in messages], ["before", "fixture", "after"])

    async def test_briefing_activity_preserves_zero_days_and_group_scope(self):
        db = await self.manager._get_db()
        def row(ts, sender, *, group="group_71", role="user"):
            return group, role, sender, "fixture", f"2026-11-{ts}"

        await db.executemany(
            "INSERT INTO messages_detail(session_id,role,sender_id,content,timestamp) VALUES(?,?,?,?,?)",
            [row("09T23:59:59", 12), row("10T00:00:00", 12), row("10T12:00:00", 10),
             row("10T13:00:00", 10), row("10T15:00:00", 10, role="assistant"),
             row("10T16:00:00", 99, group="group_72"), row("16T23:59:59", 12),
             row("17T00:00:00", 12), row("17T23:30:00", 12)],
        )
        await db.commit()
        days = await self.manager.briefing_activity("group_71", datetime(2026, 11, 17))
        self.assertEqual(len(days), 7)
        self.assertEqual(days[0], {"date": "2026-11-10", "total": 3, "leader_id": 10, "leader_count": 2})
        self.assertEqual(days[1], {"date": "2026-11-11", "total": 0, "leader_id": None, "leader_count": 0})
        self.assertEqual(days[-1], {"date": "2026-11-16", "total": 1, "leader_id": 12, "leader_count": 1})
        evening = await self.manager.briefing_activity("group_71", datetime(2026, 11, 17, 23, 30))
        self.assertEqual(evening[0]["date"], "2026-11-11")
        self.assertEqual(evening[-1], {"date": "2026-11-17", "total": 1, "leader_id": 12, "leader_count": 1})

    async def test_schema_migrates_legacy_rows_and_is_repeatable(self):
        async with aiosqlite.connect(self.manager._db_path) as db:
            await db.execute("""CREATE TABLE messages_detail (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, name TEXT,
                content TEXT, is_thinking INTEGER DEFAULT 0, timestamp TEXT)""")
            await db.execute(
                "INSERT INTO messages_detail(id,session_id,role,content,timestamp) VALUES (1,?,?,?,?)",
                ("group_71", "user", "retained", "2026-01-02T12:00:00"),
            )
            await db.commit()
            await initialize_schema(db)
            await initialize_schema(db)
            async with db.execute("SELECT content,sender_id,model_spec FROM messages_detail WHERE id=1") as cursor:
                self.assertEqual(await cursor.fetchone(), ("retained", None, None))


if __name__ == "__main__":
    unittest.main()
