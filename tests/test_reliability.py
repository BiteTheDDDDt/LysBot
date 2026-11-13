from contextlib import closing
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime
import unittest
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import memory as memory_module, conversation
from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.models import Message, ModelManager
from src.plugins.llm_chat import models as models_module
from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
from src.plugins.llm_chat.turns import send_response_turn
from src.plugins.llm_chat.reply_policy import DETAILED_REPLY
from src.plugins.llm_chat.news_sources import extract_publication_date
from src.plugins.llm_chat import news_broadcast
from scripts.healthcheck import check
from scripts.database_maintenance import reindex_copy, snapshot


class MemoryReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manager = object.__new__(MemoryManager)
        self.manager._init()
        self.manager._db_path = Path(self.directory.name) / "history.db"
        self.manager._closing = True
        self.db = await self.manager._get_db()

    async def asyncTearDown(self):
        await self.manager.close()
        self.directory.cleanup()

    async def scalar(self, sql, params=()):
        async with self.db.execute(sql, params) as cursor:
            return (await cursor.fetchone())[0]

    async def restart_manager(self):
        await self.manager.close()
        self.manager = object.__new__(MemoryManager)
        self.manager._init()
        self.manager._db_path = Path(self.directory.name) / "history.db"
        self.manager._closing = True
        self.db = await self.manager._get_db()

    async def test_cancelled_background_memory_write_releases_database_lock(self):
        for write in (
            lambda: self.manager._save_summary('group_71', 'summary', 'range'),
            lambda: self.manager._enqueue_important_info_job(
                'group_71', [Message(role='user', content='hello')], user_id=0, start_idx=1, end_idx=1),
        ):
            with patch.object(self.db, 'commit', AsyncMock(side_effect=asyncio.CancelledError)):
                with self.assertRaises(asyncio.CancelledError):
                    await write()
            self.assertFalse(self.db.in_transaction)
            def independent_write():
                with closing(sqlite3.connect(self.manager._db_path, timeout=0.1)) as other, other:
                    other.execute("INSERT OR REPLACE INTO kv_meta(key,value) VALUES('cancel_test','ok')")
            await asyncio.to_thread(independent_write)

    async def test_enrichment_retries_locked_write_without_duplicating_message(self):
        message = await self.manager.add_message(0, 'user', '待识别', 71)
        original = self.db.execute
        failures = 0
        async def execute(sql, *args, **kwargs):
            nonlocal failures
            if sql.startswith('UPDATE messages_detail') and failures == 0:
                failures += 1
                raise sqlite3.OperationalError('database is locked')
            return await original(sql, *args, **kwargs)
        with patch.object(self.db, 'execute', side_effect=execute):
            self.assertTrue(await self.manager.enrich_message(0, message, '识别完成', 71))
        self.assertEqual(failures, 1)
        self.assertEqual(await self.scalar('SELECT count(*) FROM messages_detail'), 1)
        self.assertEqual(await self.scalar('SELECT content FROM messages_detail'), '识别完成')
        self.assertEqual((await self.manager.get_history(0, 71))[0].content, '识别完成')

    async def test_summary_commit_retry_reuses_generation_and_remains_atomic(self):
        for text in ('first', 'second', 'third'):
            await self.manager.add_message(0, 'user', text, 71)
        original = self.db.commit
        attempts = 0
        async def commit():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError('database is locked')
            await original()
        with patch.object(memory_module.config, 'enable_summary', True), \
             patch.object(memory_module.config, 'summary_threshold', 2), \
             patch.object(memory_module.config, 'summary_messages', 2), \
             patch.object(memory_module.config, 'max_history_messages', 1), \
             patch.object(memory_module.model_manager, 'chat', AsyncMock(return_value='{"summary":"摘要","memory_candidates":[]}')) as generate, \
             patch.object(self.db, 'commit', side_effect=commit):
            await self.manager._compress_history('group_71')
        generate.assert_awaited_once()
        self.assertEqual(await self.scalar('SELECT count(*) FROM summaries'), 1)
        self.assertEqual(await self.scalar('SELECT count(*) FROM messages_detail'), 3)
        self.assertEqual(len(await self.manager.get_history(0, 71)), 1)
        self.assertFalse(self.db.in_transaction)

    async def test_jrrp_context_sources_use_current_generation_and_preserve_group_context(self):
        from src.plugins.llm_chat import commands
        await self.manager.add_message(101, 'user', '甲最近准备买车', 71, name='甲', sender_id=101)
        await self.manager.add_message(202, 'user', '乙刚加薪', 71, name='乙', sender_id=202)
        await self.db.execute("INSERT OR REPLACE INTO kv_meta(key,value) VALUES('memory_generation:group_71','4')")
        await self.db.execute("INSERT INTO important_info(session_id,user_id,info_type,content,created_at) VALUES('group_71',101,'user_preference','喜欢车','2026-11-06')")
        await self.db.commit()
        with patch.object(commands, 'memory_manager', self.manager), patch.object(memory_module.config, 'enable_important_info', True):
            text = await commands._build_jrrp_memory_context(101, 71, '甲')
        self.assertIn('h:4:messages:', text)
        self.assertNotIn('h:4:important_info:', text)
        self.assertIn('乙刚加薪', text)
        self.assertIn('别人的经历不能归给当前用户', text)

    async def test_valid_empty_fact_result_completes_job(self):
        await self.manager._enqueue_important_info_job(
            "group_71", [Message(role="user", content="hello")], user_id=0, start_idx=1, end_idx=1,
        )
        job, _ = await self.manager._load_next_important_info_job()
        with patch.object(memory_module.model_manager, "chat", AsyncMock(return_value="[]")):
            await self.manager._process_important_info_job(job)
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM important_info_jobs"), 0)

    async def test_restart_resets_model_but_preserves_history_and_compression_cursor(self):
        await self.manager.add_message(0, "user", "first", 71)
        self.manager._compressed_count["group_71"] = 150
        await self.manager.set_session_model(0, "openai:test", 71)
        await self.manager.add_message(0, "user", "second", 71)
        self.assertEqual(await self.scalar("SELECT compressed_count FROM sessions"), 150)
        self.assertEqual(self.manager.get_session_model(0, 71), "openai:test")
        self.assertFalse(await self.scalar("SELECT model FROM sessions"))
        await self.restart_manager()
        self.assertEqual(self.manager.get_session_model(0, 71), "")
        history = await self.manager.get_history(0, 71)
        self.assertEqual([message.content for message in history], ["first", "second"])
        self.assertEqual(self.manager.get_session_model(0, 71), "")
        self.assertEqual(self.manager._compressed_count["group_71"], 150)

    async def test_legacy_model_is_ignored_in_group_and_private_sessions_after_restart(self):
        for group_id in (None, 71):
            await self.manager.add_message(42, "user", "history", group_id)
        await self.db.execute("UPDATE sessions SET model='openai:legacy'")
        await self.db.commit()
        await self.restart_manager()
        for group_id in (None, 71):
            await self.manager.get_history(42, group_id)
            override = self.manager.get_session_model(42, group_id)
            self.assertEqual(override, "")
            with patch.object(models_module, "resolve_role", return_value=("openai:configured-default", [])):
                primary, _ = ModelManager()._resolve_candidates(role="chat", model=override or None)
            self.assertEqual(primary, "openai:configured-default")
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM messages_detail"), 2)

    async def test_runtime_overrides_survive_history_reload_and_stay_scoped(self):
        await self.manager.add_message(42, "user", "private history")
        await self.manager.add_message(42, "user", "group history", 71)
        await self.db.execute("UPDATE sessions SET model='openai:legacy'")
        await self.db.commit()
        self.manager._cache.clear()
        with patch.object(self.manager, "_get_db", AsyncMock(side_effect=AssertionError("Model switching must not access the database"))):
            await self.manager.set_session_model(42, "openai:private")
            await self.manager.set_session_model(42, "openai:group", 71)
        for group_id in (None, 71):
            await self.manager.get_history(42, group_id)
            await self.manager.add_message(42, "user", "next message", group_id)
        self.assertEqual(self.manager.get_session_model(42), "openai:private")
        self.assertEqual(self.manager.get_session_model(84, 71), "openai:group")
        self.assertEqual(self.manager.get_session_model(84), "")
        self.assertEqual(self.manager.get_session_model(42, 72), "")
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM sessions WHERE model='openai:legacy'"), 2)
        await self.restart_manager()
        for group_id in (None, 71):
            await self.manager.get_history(42, group_id)
            self.assertEqual(self.manager.get_session_model(42, group_id), "")

    async def test_detail_failure_rolls_back_session_and_retries_both(self):
        await self.manager.add_message(0, "user", "first", 71)
        await self.db.execute("CREATE TRIGGER reject_detail BEFORE INSERT ON messages_detail BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
        await self.db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            await self.manager.add_message(0, "user", "second", 71)
        stored = json.loads(await self.scalar("SELECT messages FROM sessions"))
        self.assertEqual([m["content"] for m in stored], ["first"])
        await self.db.execute("DROP TRIGGER reject_detail")
        await self.db.commit()
        await self.manager.add_message(0, "user", "third", 71)
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM messages_detail"), 3)
        self.assertEqual(len(json.loads(await self.scalar("SELECT messages FROM sessions"))), 3)

    async def test_clear_all_does_not_allow_inflight_fact_job_to_resurrect_memory(self):
        await self.manager.add_message(0, "user", "first", 71)
        await self.manager._save_summary("group_71", "summary", "range")
        await self.manager._enqueue_important_info_job(
            "group_71", [Message(role="user", content="first")], user_id=0, start_idx=1, end_idx=1,
        )
        job, _ = await self.manager._load_next_important_info_job()
        await self.manager.clear_history(0, 71, full=True)
        await self.manager._complete_important_info_job(job, [(0, "key_event", "old fact")])
        for table in ("summaries", "important_info", "messages_detail", "important_info_jobs"):
            self.assertEqual(await self.scalar(f"SELECT COUNT(*) FROM {table}"), 0)

    async def test_short_reset_preserves_recall_and_model(self):
        await self.manager.add_message(0, "user", "first", 71)
        await self.manager.set_session_model(0, "openai:test", 71)
        self.manager._cache.clear()
        await self.manager.clear_history(0, 71)
        self.assertEqual(await self.manager.get_history(0, 71), [])
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM messages_detail"), 1)
        self.assertEqual(self.manager.get_session_model(0, 71), "openai:test")

    async def test_old_turn_cannot_restore_messages_after_full_clear(self):
        entered, release = asyncio.Event(), asyncio.Event()
        @conversation.serialized_turn(lambda: "group_71")
        async def old_turn():
            entered.set()
            await release.wait()
            await self.manager.add_message(0, "assistant", "sent before reset", 71)
        task = asyncio.create_task(old_turn())
        await entered.wait()
        await self.manager.clear_history(0, 71, full=True)
        release.set()
        await task
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM messages_detail"), 0)

    async def test_concurrent_append_keeps_every_detail_once(self):
        await asyncio.gather(*(self.manager.add_message(0, "user", str(i), 71) for i in range(20)))
        history = await self.manager.get_history(0, 71)
        self.assertEqual(len(history), 20)
        self.assertEqual(await self.scalar("SELECT COUNT(DISTINCT content) FROM messages_detail"), 20)
        history.clear()
        self.assertEqual(len(await self.manager.get_history(0, 71)), 20)

    async def test_failed_summary_keeps_originals_and_cursor(self):
        for i in range(4):
            await self.manager.add_message(0, "user", str(i), 71)
        with patch.object(memory_module.config, "summary_threshold", 2), patch.object(memory_module.config, "max_history_messages", 1), patch.object(memory_module.model_manager, "chat", AsyncMock(side_effect=TimeoutError())):
            await self.manager._compress_history("group_71")
        self.assertEqual(len(await self.manager.get_history(0, 71)), 4)
        self.assertEqual(await self.scalar("SELECT compressed_count FROM sessions"), 0)
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM summaries"), 0)

    async def test_summary_commit_preserves_messages_arriving_during_generation(self):
        for i in range(4):
            await self.manager.add_message(0, "user", str(i), 71)
        async def generate(*args, **kwargs):
            await self.manager.add_message(0, "user", "arrived while summarizing", 71)
            return "summary"
        with patch.object(memory_module.config, "summary_threshold", 4), patch.object(memory_module.config, "max_history_messages", 2), patch.object(memory_module.config, "summary_messages", 3), patch.object(memory_module.model_manager, "chat", AsyncMock(side_effect=generate)):
            await self.manager.add_message(0, "user", "trigger", 71)
            await self.manager._compress_history("group_71")
        contents = [m.content for m in await self.manager.get_history(0, 71)]
        self.assertIn("arrived while summarizing", contents)
        self.assertEqual(await self.scalar("SELECT compressed_count FROM sessions"), 3)
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM messages_detail"), 6)
        self.assertEqual(await self.scalar("SELECT COUNT(*) FROM important_info_jobs"), 1)

    async def test_digest_uses_event_time_not_summary_creation_time(self):
        await self.db.executemany(
            "INSERT INTO messages_detail(session_id,role,content,timestamp,is_thinking) VALUES ('group_71','user',?,?,0)",
            [("old", "2026-09-03T23:29:59"), ("included", "2026-09-04T10:00:00"), ("later", "2026-09-04T23:30:00")],
        )
        await self.db.commit()
        messages, count = await self.manager.get_digest_messages("group_71", datetime(2026,9,3,23,30), datetime(2026,9,4,23,30))
        self.assertEqual(count, 1)
        self.assertEqual([m.content for m in messages], ["included"])
        expected_id = await self.scalar("SELECT id FROM messages_detail WHERE content='included'")
        self.assertEqual(messages[0].source_id, expected_id)


class TurnReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_session_serializes_but_other_sessions_progress(self):
        entered, release = asyncio.Event(), asyncio.Event()
        order = []
        @conversation.serialized_turn(lambda sid, name: sid)
        async def work(sid, name):
            order.append(name)
            if name == "first":
                entered.set()
                await release.wait()
        first = asyncio.create_task(work("user_900", "first"))
        await entered.wait()
        second = asyncio.create_task(work("user_900", "second"))
        await work("user_901", "other")
        self.assertEqual(order, ["first", "other"])
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(order, ["first", "other", "second"])

    async def test_reset_invalidates_active_and_queued_sends(self):
        entered, release = asyncio.Event(), asyncio.Event()
        sent = []
        @conversation.serialized_turn(lambda: "user_902")
        async def work():
            entered.set()
            await release.wait()
            conversation.ensure_send_allowed()
            sent.append(True)
        first = asyncio.create_task(work())
        await entered.wait()
        second = asyncio.create_task(work())
        await asyncio.sleep(0)
        conversation.invalidate("user_902")
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(sent, [])

    async def test_second_send_failure_retains_first_message(self):
        sender = AsyncMock(side_effect=[None, RuntimeError("send rejected")])
        turn = await send_response_turn(text="第一句。第二句。", decision=PlannerDecision(action="reply", parts=[ReplyPart("text"), ReplyPart("text")]), planned_meme=None, send_text=sender, delay_range=(0,0), reply_policy=DETAILED_REPLY)
        self.assertEqual(turn.sent_texts, ["第一句"])
        self.assertEqual(sender.await_count, 2)

    async def test_cancellation_between_sends_retains_first_message(self):
        sender = AsyncMock()
        with patch("src.plugins.llm_chat.turns.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)):
            turn = await send_response_turn(text="第一句。第二句。", decision=PlannerDecision(action="reply", parts=[ReplyPart("text"), ReplyPart("text")]), planned_meme=None, send_text=sender, reply_policy=DETAILED_REPLY)
        self.assertEqual(turn.sent_texts, ["第一句"])
        sender.assert_awaited_once()

    async def test_shutdown_drains_scheduled_producers(self):
        started, stopped = asyncio.Event(), asyncio.Event()
        @conversation.tracked_job
        async def produce():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        task = asyncio.create_task(produce())
        await started.wait()
        try:
            await conversation.close()
            self.assertTrue(task.done())
            self.assertTrue(stopped.is_set())
        finally:
            conversation.start()


class ModelReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_media_turn_can_outlive_normal_chat_budget(self):
        manager = ModelManager()
        async def finish_media(*args, **kwargs):
            await asyncio.sleep(0.2)
            return "media completed"
        with (
            patch.object(models_module.config, "chat_turn_timeout", 0.1),
            patch.object(models_module.config, "chat_media_turn_timeout", 1.0),
            patch.object(manager, "_chat_impl", AsyncMock(side_effect=finish_media)),
        ):
            for role, required_tool in [("chat", "generate_image"), (None, "generate_image"), ("chat", "send_voice")]:
                result = await manager.chat([], role=role, required_tool=required_tool)
                self.assertEqual(result, "media completed")

    async def test_stuck_media_is_still_cancelled_with_clear_timeout(self):
        stopped = asyncio.Event()
        manager = ModelManager()
        async def stuck(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        with (
            patch.object(models_module.config, "chat_turn_timeout", 0.1),
            patch.object(models_module.config, "chat_media_turn_timeout", 0.15),
            patch.object(manager, "_chat_impl", AsyncMock(side_effect=stuck)) as call,
        ):
            with self.assertRaises(models_module.ModelTurnTimeout) as caught:
                await manager.chat([], role="chat", required_tool="generate_image")
        self.assertTrue(stopped.is_set())
        self.assertIn("generate_image", str(caught.exception))
        self.assertEqual(caught.exception.timeout, 0.15)
        self.assertIn("图片生成", caught.exception.user_message)
        call.assert_awaited_once()

    async def test_media_budget_does_not_extend_planner_or_background_calls(self):
        manager = ModelManager()
        async def stuck(*args, **kwargs):
            await asyncio.Event().wait()
        with (
            patch.object(models_module.config, "planner_total_timeout", 0.1),
            patch.object(models_module.config, "background_llm_total_timeout", 0.1),
            patch.object(models_module.config, "chat_media_turn_timeout", 1.0),
            patch.object(manager, "_chat_impl", AsyncMock(side_effect=stuck)),
        ):
            for role in ["planner", "vision"]:
                with self.assertRaises(models_module.ModelTurnTimeout) as caught:
                    await manager.chat([], role=role, required_tool="generate_image")
                self.assertEqual(caught.exception.timeout, 0.1)

    async def test_nested_request_timeout_is_not_misreported_as_turn_timeout(self):
        manager = ModelManager()
        original = asyncio.TimeoutError("nested request timeout")
        with patch.object(manager, "_chat_impl", AsyncMock(side_effect=original)):
            with self.assertRaises(asyncio.TimeoutError) as caught:
                await manager.chat([], required_tool="generate_image")
        self.assertIs(caught.exception, original)

    async def test_external_cancellation_propagates_without_becoming_timeout(self):
        started, stopped = asyncio.Event(), asyncio.Event()
        manager = ModelManager()
        async def stuck(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        with patch.object(manager, "_chat_impl", AsyncMock(side_effect=stuck)):
            task = asyncio.create_task(manager.chat([], required_tool="generate_image"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())

    async def test_timeout_restores_parent_session_context(self):
        manager = ModelManager()
        outer = models_module.current_session_id.set("user_parent")
        async def stuck(*args, **kwargs):
            self.assertEqual(models_module.current_session_id.get(), "user_child")
            await asyncio.Event().wait()
        try:
            with (
                patch.object(models_module.config, "chat_turn_timeout", 0.1),
                patch.object(manager, "_chat_impl", AsyncMock(side_effect=stuck)),
            ):
                with self.assertRaises(models_module.ModelTurnTimeout):
                    await manager.chat([], session_id="user_child")
            self.assertEqual(models_module.current_session_id.get(), "user_parent")
        finally:
            models_module.current_session_id.reset(outer)

    async def test_total_chat_budget_cancels_the_candidate_loop(self):
        cancelled = asyncio.Event()
        async def slow(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        manager = ModelManager()
        with patch.object(models_module.config, "chat_turn_timeout", 0.1), patch.object(manager, "_chat_impl", AsyncMock(side_effect=slow)):
            with self.assertRaises(TimeoutError):
                await manager.chat([Message(role="user", content="hello")])
        self.assertTrue(cancelled.is_set())

    async def test_one_vision_caller_cancelling_does_not_cancel_shared_recognition(self):
        manager = ModelManager()
        entered, release = asyncio.Event(), asyncio.Event()
        expected = {"description":"a cat", "is_meme":False, "topic":None}
        async def recognize(*args, **kwargs):
            entered.set()
            await release.wait()
            return expected
        manager._vision_analysis_cache.clear()
        with patch.object(manager,"_vision_cache_limit",return_value=5), patch.object(manager,"_analyze_image_for_meme_uncached",AsyncMock(side_effect=recognize)) as generate:
            first = asyncio.create_task(manager._get_image_analysis("test-image",session_id="group_71",caller="test"))
            await entered.wait()
            second = asyncio.create_task(manager._get_image_analysis("test-image",session_id="group_71",caller="test"))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual(await second, expected)
            self.assertEqual(await manager._get_image_analysis("test-image",session_id="group_71",caller="test"), expected)
            generate.assert_awaited_once()
            self.assertEqual(manager._vision_pending, {})
        manager._vision_analysis_cache.clear()


class OperationalTests(unittest.TestCase):
    def test_config_loads_process_environment_without_dotenv_file(self):
        import nonebot
        from src.plugins.llm_chat.config import Config, RuntimeConfig
        driver_config = nonebot.get_driver().config
        with patch.object(driver_config, "_env_file", None), patch.dict(os.environ, {
            "OPENAI_API_KEY": "isolated-key", "CONTEXT_MAX_CHARS": "7654", "CHAT_TURN_TIMEOUT": "123",
        }):
            # The global NoneBot config does not declare these plugin fields.
            with patch("nonebot.plugin.model_dump", return_value={}):
                value = nonebot.get_plugin_config(Config)
        self.assertEqual(value.openai_api_key, "isolated-key")
        self.assertEqual(value.chat_turn_timeout, 123)
        runtime = RuntimeConfig(**value.model_dump(exclude_unset=True))
        self.assertEqual(runtime.context_max_chars, 18000)

    def test_source_date_overrides_model_date_and_missing_evidence_is_rejected(self):
        raw = json.dumps({"items":[{"source_index":1,"published_at":"2026-09-05","summary":"A factual update"}]})
        source = {"index":1,"title":"New release","site_name":"Publisher","url":"https://example.com/release"}
        parse = lambda: news_broadcast.parse_native_news_selection(raw,[source],interest_label="AI",earliest_date=datetime(2026,9,3).date(),latest_date=datetime(2026,9,5).date())
        self.assertEqual(parse(), [])
        source["published_at"] = "2020-01-01"
        self.assertEqual(parse(), [])
        source["published_at"] = "2026-09-04"
        self.assertEqual(parse()[0]["published_at"], "2026-09-04")

    def test_article_metadata_conflicts_fail_closed(self):
        html = '<script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-09-04T10:00:00+08:00"}</script>'
        self.assertEqual(extract_publication_date(html), "2026-09-04")
        self.assertEqual(extract_publication_date(html + '<meta property="article:published_time" content="2026-09-05">'), "")

    def test_health_checks_database_and_onebot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {"updated_at":100,"plugin_ready":True,"database_ok":False,"onebot_connected":True}
            (root/"health.json").write_text(json.dumps(state))
            self.assertEqual(check(root,now=101), (False,"database unhealthy"))
            state.update(database_ok=True,onebot_connected=False)
            (root/"health.json").write_text(json.dumps(state))
            self.assertEqual(check(root,now=101), (False,"OneBot disconnected"))

    def test_maintenance_only_repairs_a_new_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory)/"source.db", Path(directory)/"copy.db"
            with sqlite3.connect(source) as db:
                db.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY, content TEXT)")
                db.execute("CREATE INDEX content_index ON sample(content)")
                db.execute("INSERT INTO sample(content) VALUES ('preserve this')")
            db.close()
            original = source.read_bytes()
            result = reindex_copy(source, destination)
            self.assertTrue(result["valid"])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                snapshot(source,destination)
            with self.assertRaises(ValueError):
                snapshot(source,source)
