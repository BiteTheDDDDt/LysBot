import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import history_sql, tools
from src.plugins.llm_chat.agent_runner import READ_TOOLS, AgentRunner
from src.plugins.llm_chat.config import config


class HistorySQLTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "history.db"
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
                CREATE TABLE kv_meta(key TEXT PRIMARY KEY,value TEXT);
                INSERT INTO kv_meta VALUES('memory_generation:group_7','3');
                CREATE TABLE messages_detail(id INTEGER PRIMARY KEY,session_id TEXT,sender_id INTEGER,
                    name TEXT,role TEXT,content TEXT,timestamp TEXT,is_thinking INTEGER DEFAULT 0);
                INSERT INTO messages_detail VALUES
                    (1,'group_7',101,'甲','user','旧显卡','2026-09-01T10:00:00',0),
                    (2,'group_7',101,'甲','user','新显卡','2026-09-10T10:00:00',0),
                    (3,'group_7',202,'乙','user','模型建议','2026-09-10T11:00:00',0),
                    (4,'group_7',101,'甲','assistant','内部思考','2026-09-10T12:00:00',1),
                    (5,'group_8',101,'甲','user','外群秘密','2026-09-10T12:00:00',0);
                CREATE TABLE summaries(id INTEGER PRIMARY KEY,session_id TEXT,content TEXT,
                    time_range TEXT,created_at TEXT,provenance_json TEXT);
                INSERT INTO summaries VALUES(1,'group_7','摘要','九月','2026-09-10','{}');
                CREATE TABLE important_info(id INTEGER PRIMARY KEY,session_id TEXT,user_id INTEGER,
                    info_type TEXT,content TEXT,created_at TEXT,provenance_json TEXT);
                INSERT INTO important_info VALUES(1,'group_7',101,'hardware','已购买','2026-09-10','{"source_ids":[1,2]}');
                INSERT INTO important_info VALUES(2,'group_8',101,'hardware','秘密配置','2026-09-10','{}');
                CREATE TABLE agent_runs(state_json TEXT);
                INSERT INTO agent_runs VALUES('private transcript');
            """)

    def tearDown(self):
        self.tmp.cleanup()

    async def query(self, sql="", **kwargs):
        return await history_sql.query_history_sql(self.path, "group_7", sql, **kwargs)

    async def test_schema_and_scoped_count_without_template(self):
        self.assertEqual(set((await self.query())["schema"]), {"messages", "summaries", "facts"})
        found = await self.query("SELECT count(*),count(DISTINCT sender_id) FROM messages")
        self.assertEqual(found["rows"], [[3, 2]])
        self.assertEqual(found["effective_scope"]["generation"], 3)

    async def test_cte_window_join_json_and_original_source_ids(self):
        found = await self.query(
            """
            WITH ranked AS (
              SELECT *,row_number() OVER(PARTITION BY sender_id ORDER BY timestamp DESC) AS rn FROM messages
            )
            SELECT m.source_id,m.content,json_extract(f.provenance_json,'$.source_ids[1]')
            FROM ranked m JOIN facts f ON f.user_id=m.sender_id
            WHERE rn=1 AND f.info_type=:kind
        """,
            parameters={"kind": "hardware"},
        )
        self.assertEqual(found["status"], "ok", found)
        self.assertEqual(found["rows"], [["h:3:messages:2", "新显卡", 2]])

    async def test_json_table_function_and_union(self):
        found = await self.query(
            "SELECT j.value FROM facts f,json_each(f.provenance_json,'$.source_ids') j UNION SELECT id FROM summaries ORDER BY 1"
        )
        self.assertEqual(found["rows"], [[1], [2]], found)

    async def test_access_control_cannot_be_bypassed_by_alias_or_cte(self):
        for sql in (
            "SELECT * FROM main.messages_detail",
            "SELECT count(*) FROM main.messages_detail",
            "SELECT * FROM agent_runs",
            "SELECT sql FROM sqlite_master",
            "SELECT sql FROM temp.sqlite_schema",
            "WITH messages AS(SELECT * FROM main.messages_detail) SELECT * FROM messages",
            "WITH facts AS(SELECT * FROM main.important_info) SELECT * FROM facts",
            "WITH summaries AS(SELECT count(*) n FROM main.summaries) SELECT * FROM summaries",
            "SELECT (SELECT content FROM main.messages_detail WHERE id=5) FROM messages",
        ):
            with self.subTest(sql=sql):
                found = await self.query(sql)
                self.assertEqual(found["error"]["code"], "sql_access_denied", found)
        self.assertEqual((await self.query("SELECT content FROM messages WHERE id=5"))["status"], "empty")

    async def test_write_metadata_file_and_extension_operations_denied(self):
        for sql in (
            "DELETE FROM messages_detail",
            "DROP TABLE messages_detail",
            "CREATE TABLE x(a)",
            "CREATE TEMP VIEW messages2 AS SELECT 1",
            "PRAGMA query_only=OFF",
            "PRAGMA table_info(messages)",
            "ATTACH DATABASE ':memory:' AS other",
            "SELECT load_extension('x')",
            "SELECT readfile('/etc/passwd')",
            "SELECT randomblob(1000000000)",
            "SELECT printf('%1000000000s','x')",
            "SELECT * FROM pragma_table_info('messages')",
            "SELECT 1; SELECT 2",
            "EXPLAIN DELETE FROM main.messages_detail",
        ):
            with self.subTest(sql=sql):
                found = await self.query(sql)
                self.assertEqual(found["status"], "invalid_arguments", found)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM messages_detail").fetchone()[0], 5)
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='x'").fetchone())

    async def test_bindings_errors_and_output_limits(self):
        encoded = await self.query("SELECT content FROM messages WHERE id=:id", parameters='{"id":2}')
        self.assertEqual(encoded["rows"], [["新显卡"]])
        self.assertEqual((await self.query("SELECT 1", parameters="{"))["error"]["code"], "invalid_parameters_json")
        self.assertEqual(
            (await self.query("SELECT content FROM messages WHERE content=?", parameters=["' OR 1=1 --"]))["rows"], []
        )
        found = await self.query("SELECT id FROM messages ORDER BY id", limit=1)
        self.assertEqual(found["rows"], [[1]])
        self.assertTrue(found["truncated"])
        bad = await self.query("SELECT imaginary_column FROM messages")
        self.assertIn("imaginary_column", bad["error"]["message"])
        self.assertEqual((await self.query("SELECT :x", parameters={"x": [1]}))["status"], "invalid_arguments")
        large = await self.query("SELECT replace('xx','x',:long)", parameters={"long": "a" * 5000})
        self.assertEqual(len(large["rows"][0][0]), 4000)
        self.assertIn("cell_text_truncated_read_original_or_refine_sql", large["warnings"])

    async def test_recursive_query_deadline_and_cancellation(self):
        sql = "WITH RECURSIVE x(n) AS(SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT max(n) FROM x"
        with patch.object(history_sql, "SQL_TIMEOUT", 0.03):
            found = await self.query(sql)
        self.assertEqual(found["error"]["code"], "query_budget_exhausted_refine_sql")
        task = asyncio.create_task(self.query(sql))
        await asyncio.sleep(0.03)
        started = time.monotonic()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertLess(time.monotonic() - started, 1)

    async def test_concurrent_scope_rewrite_invalidates_result(self):
        with patch.object(history_sql, "_scope", side_effect=[(3, "a"), (4, "b")]):
            found = await self.query("SELECT * FROM messages")
        self.assertEqual(found["error"]["code"], "scope_changed_retry_query")
        self.assertEqual(found["items"], [])

    async def test_sql_tool_is_read_only_and_disabled_with_agentic_switch(self):
        self.assertIn("query_history", READ_TOOLS)
        with patch.object(config, "agentic_search_enabled", False):
            self.assertNotIn("query_history", {t["function"]["name"] for t in tools.get_chat_tools()})
            self.assertEqual(
                json.loads(await tools.query_history("SELECT 1"))["error"]["code"], "agentic_search_disabled"
            )
        runner = AgentRunner("group_7", 10)
        runner.memory_path = None
        execute = AsyncMock(return_value=json.dumps({"status": "ok", "items": [], "columns": ["n"], "rows": [[3]]}))
        await runner.execute("query_history", '{"sql":"SELECT count(*) FROM messages"}', execute)
        self.assertEqual(runner.actions, {})
        self.assertEqual(runner.no_new_evidence_calls, 0)

    async def test_sql_source_evidence_is_reread_not_trusted_from_computed_rows(self):
        from src.plugins.llm_chat.memory import memory_manager
        from src.plugins.llm_chat.models import current_session_id

        token = current_session_id.set("group_7")
        try:
            with patch.object(memory_manager, "_get_db", AsyncMock()), patch.object(memory_manager, "_db_path", self.path):
                found = json.loads(await tools.query_history(
                    "SELECT source_id, '模型伪造的文本' AS content FROM messages WHERE id=1"
                ))
                self.assertIn("模型伪造的文本", str(found["rows"]))
                self.assertEqual(found["items"][0]["content"], "旧显卡")
                self.assertEqual(found["items"][0]["read_state"], "original")
                self.assertEqual(found["items"][0]["evidence_kind"], "user_message")
                blocked = json.loads(await tools.query_history("SELECT 'h:3:messages:5' AS source_id"))
                self.assertEqual(blocked["items"], [])
                self.assertNotIn("外群秘密", json.dumps(blocked, ensure_ascii=False))
                self.assertIn("sql_source_verification_failed_read_history_required", blocked["warnings"])
        finally:
            current_session_id.reset(token)

    async def test_sql_cache_invalidates_after_memory_revision_changes(self):
        runner = AgentRunner("group_7", 10)
        runner.memory_path = self.path

        async def execute_sql(name, args):
            return json.dumps(await self.query(args["sql"]))

        execute = AsyncMock(side_effect=execute_sql)
        arguments = '{"sql":"SELECT count(*) FROM messages"}'
        first = json.loads(await runner.execute("query_history", arguments, execute))
        await runner.execute("query_history", arguments, execute)
        self.assertEqual(execute.await_count, 1)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT INTO messages_detail VALUES(6,'group_7',101,'甲','user','新增','2026-09-15',0)")
            db.execute("INSERT INTO kv_meta VALUES('memory_revision:group_7','new')")
            db.commit()
        second = json.loads(await runner.execute("query_history", arguments, execute))
        self.assertEqual(first["rows"], [[3]])
        self.assertEqual(second["rows"], [[4]])
        self.assertEqual(execute.await_count, 2)
