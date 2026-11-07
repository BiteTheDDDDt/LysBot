import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import usage_db


class UsageDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await usage_db.close_db()
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "history.db"
        self.patcher = patch.object(usage_db, "DB_PATH", self.path)
        self.patcher.start()

    async def asyncTearDown(self):
        await usage_db.close_db()
        self.patcher.stop()
        self.directory.cleanup()

    async def test_fixed_window_is_half_open_and_session_scoped(self):
        db = await usage_db._get_db()
        for session, ts, tokens in (
            ("group_123", "2026-10-19T23:59:59", 100),
            ("group_123", "2026-10-20T00:00:00", 1),
            ("group_123", "2026-10-20T23:59:59", 2),
            ("group_123", "2026-10-21T00:00:00", 100),
            ("group_456", "2026-10-20T12:00:00", 100),
        ):
            await db.execute(
                "INSERT INTO llm_usage(ts,session_id,role,spec,provider,model_id,total_tokens,usage_known) "
                "VALUES(?,?,'chat','test:model','test','model',?,1)", (ts, session, tokens),
            )
        await db.commit()
        result = await usage_db.query_session(
            "group_123", start=datetime(2026, 10, 20), end=datetime(2026, 10, 21),
        )
        self.assertEqual(result["totals"]["calls"], 2)
        self.assertEqual(result["totals"]["total_tokens"], 3)
        self.assertEqual(result["by_spec"][0]["total_tokens"], 3)
        with self.assertRaises(ValueError):
            await usage_db.query_session("group_123", start=datetime(2026, 10, 20))

    async def test_missing_usage_is_distinct_from_measured_zero_in_all_aggregates(self):
        for usage, success in (
            (None, True), (None, False),
            ({"completion_tokens": 2}, True),
            ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, True),
            ({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, True),
        ):
            await usage_db.record_usage(
                session_id="group_123", role="image_gen", spec="test:image",
                caller="test", usage=usage, latency_ms=1, success=success,
            )
        result = await usage_db.query_session("group_123", "all")
        self.assertEqual(result["totals"]["calls"], 5)
        self.assertEqual(result["totals"]["unknown_usage_calls"], 3)
        self.assertEqual(result["totals"]["total_tokens"], 5)
        self.assertEqual(result["totals"]["completion_tokens"], 4)
        self.assertEqual(result["totals"]["failures"], 1)
        for key in ("by_provider", "by_model", "by_spec", "by_role"):
            self.assertEqual(result[key][0]["unknown_usage_calls"], 3)

    async def test_migration_preserves_old_rows_without_claiming_old_zeroes_are_known(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE llm_usage(id INTEGER PRIMARY KEY, ts TEXT, session_id TEXT, role TEXT, "
                "spec TEXT, provider TEXT, model_id TEXT, caller TEXT, prompt_tokens INTEGER, "
                "completion_tokens INTEGER, total_tokens INTEGER, latency_ms INTEGER, "
                "success INTEGER, fallback_attempt INTEGER)"
            )
            for tokens in (0, 4):
                db.execute(
                    "INSERT INTO llm_usage VALUES(NULL,'2026-10-20','group_123','chat','test:model',"
                    "'test','model','test',?,0,?,1,1,0)", (tokens, tokens),
                )
        result = await usage_db.query_session("group_123", "all")
        self.assertEqual(result["totals"]["calls"], 2)
        self.assertEqual(result["totals"]["total_tokens"], 4)
        self.assertEqual(result["totals"]["unknown_usage_calls"], 1)
        await usage_db.close_db()
        result = await usage_db.query_session("group_123", "all")
        self.assertEqual(result["totals"]["unknown_usage_calls"], 1)

    async def test_cancelled_usage_commit_rolls_back_and_releases_writer(self):
        db = await usage_db._get_db()
        with patch.object(db, 'commit', AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await usage_db.record_usage(session_id='group_123', role='chat', spec='test:model',
                                            caller='test', usage=None, latency_ms=1, success=True)
        self.assertFalse(db.in_transaction)
        with closing(sqlite3.connect(self.path, timeout=0.1)) as other, other:
            other.execute("INSERT INTO llm_usage(ts,session_id,role,spec,provider,model_id) VALUES('now','other','chat','test:model','test','model')")
        result = await usage_db.query_session('group_123', 'all')
        self.assertEqual(result['totals']['calls'], 0)
