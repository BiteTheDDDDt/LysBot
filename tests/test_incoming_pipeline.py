import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from nonebot.adapters.onebot.v11 import Message, MessageSegment

nonebot.init()

from src.plugins.llm_chat import chat, conversation, followups
from src.plugins.llm_chat.memory import MemoryManager


class FakeGroupEvent(SimpleNamespace):
    def get_plaintext(self):
        return self.message.extract_plain_text()


class IncomingPipelineTests(unittest.IsolatedAsyncioTestCase):
    group_id = 17018
    session_id = "group_17018"
    image_url = "https://example.com/synthetic.png"

    async def asyncSetUp(self):
        await chat.close_incoming_tasks()
        self.tmp = tempfile.TemporaryDirectory()
        self.previous_instance = MemoryManager._instance
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.tmp.name) / 'history.db'
        self.manager._closing = True
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(chat, 'memory_manager', self.manager))
        self.patches.enter_context(patch.object(chat, 'GroupMessageEvent', FakeGroupEvent))
        self.patches.enter_context(patch.object(conversation, '_closing', False))
        for setting in ('enable_summary', 'enable_important_info', 'enable_group_repeat', 'enable_sentiment_analysis'):
            self.patches.enter_context(patch.object(chat.config, setting, False))
        for method in ('add_recent_message', 'increment_counter'):
            self.patches.enter_context(patch.object(chat.proactive_manager, method))
        self.patches.enter_context(patch.object(chat.proactive_manager, 'try_proactive', AsyncMock(return_value=False)))
        self.patches.enter_context(patch.object(chat, 'download_images_as_base64', AsyncMock(side_effect=lambda urls: [(url, 'c3ludGhldGljLWltYWdl') for url in urls])))
        self.patches.enter_context(patch.object(chat.model_manager, 'analyze_image_for_meme', AsyncMock(return_value={'description': '一只猫', 'is_meme': False})))
        self.bot = SimpleNamespace(call_api=AsyncMock(return_value={'messages': [{'sender': {'nickname': '甲', 'user_id': 1}, 'content': '转发中的实际内容'}]}), send=AsyncMock())
        self.tasks, self.gates = ([], [])
        followups.windows.pop(self.group_id, None)
        self.db = await self.manager._get_db()

    async def asyncTearDown(self):
        for gate in self.gates:
            gate.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await chat.close_incoming_tasks()
        await self.manager.close()
        self.patches.close()
        MemoryManager._instance = self.previous_instance
        chat.image_reference_store.clear()
        chat._mention_names.clear()
        chat.group_message_orders.pop(self.group_id, None)
        chat.aggregation_buffers.pop(self.group_id, None)
        followups.windows.pop(self.group_id, None)
        self.tmp.cleanup()

    def event(self, content, *, message_id=1, direct=True):
        message = Message(content)
        return FakeGroupEvent(
            self_id=999, user_id=123, group_id=self.group_id, message_id=message_id,
            message=message, original_message=Message(message), reply=None, to_me=direct,
            sender=SimpleNamespace(card="测试成员", nickname="测试成员", user_id=123),
            time=1792202400 + message_id,
        )

    def spawn_record(self, event):
        task = asyncio.create_task(chat.record_group_message(self.bot, event))
        self.tasks.append(task)
        return task

    async def rows(self):
        async with self.db.execute(
            "SELECT id,content,timestamp FROM messages_detail WHERE session_id=? ORDER BY id", (self.session_id,)
        ) as cursor:
            return await cursor.fetchall()

    def blocked_vision(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.gates.append(release)

        async def analyze(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"description": "一只猫", "is_meme": False}

        self.patches.enter_context(patch.object(chat.model_manager, "analyze_image_for_meme", AsyncMock(side_effect=analyze)))
        return entered, release

    async def finish_background(self):
        await asyncio.wait_for(asyncio.gather(*tuple(chat._incoming_tasks)), timeout=5)

    async def test_raw_message_is_saved_before_forward_expansion_and_keeps_arrival_order(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.gates.append(release)
        original_resolve = chat.resolve_incoming_message

        async def delayed_resolve(bot, event):
            if event.message_id == 1:
                entered.set()
                await release.wait()
            return await original_resolve(bot, event)

        with patch.object(chat, "resolve_incoming_message", AsyncMock(side_effect=delayed_resolve)):
            first = self.spawn_record(self.event(MessageSegment.forward("synthetic-forward")))
            await asyncio.wait_for(entered.wait(), timeout=5)
            raw = await self.rows()
            self.assertEqual(len(raw), 1)
            self.assertIn("等待展开", raw[0][1])
            self.assertFalse(first.done())
            await asyncio.wait_for(chat.record_group_message(self.bot, self.event("后到的文本", message_id=2)), timeout=5)
            before = await self.rows()
            self.assertEqual(len(before), 2)
            self.assertIn("后到的文本", before[1][1])
            release.set()
            await asyncio.wait_for(first, timeout=5)
        after = await self.rows()
        self.assertEqual([(r[0], r[2]) for r in after], [(r[0], r[2]) for r in before])
        self.assertIn("转发中的实际内容", after[0][1])
        self.assertIn("后到的文本", after[1][1])

    async def test_non_direct_image_returns_before_vision_and_enriches_its_original_row(self):
        entered, release = self.blocked_vision()
        event = self.event(MessageSegment.image(self.image_url), direct=False)
        await asyncio.wait_for(chat.record_group_message(self.bot, event), timeout=5)
        await asyncio.wait_for(entered.wait(), timeout=5)
        raw = await self.rows()
        self.assertIn("等待识别", raw[0][1])
        self.assertFalse(await chat.should_respond(event))
        await asyncio.wait_for(chat.record_group_message(self.bot, self.event("后到的文本", message_id=2)), timeout=5)
        before = await self.rows()
        self.assertEqual(len(before), 2)
        release.set()
        await self.finish_background()
        after = await self.rows()
        self.assertEqual([(r[0], r[2]) for r in after], [(r[0], r[2]) for r in before])
        self.assertIn("一只猫", after[0][1])
        self.assertIn("后到的文本", after[1][1])

    async def test_direct_image_waits_for_vision_after_raw_record_is_persisted(self):
        entered, release = self.blocked_vision()
        event = self.event(Message("这张是什么") + MessageSegment.image(self.image_url))
        recording = self.spawn_record(event)
        await asyncio.wait_for(entered.wait(), timeout=5)
        raw = await self.rows()
        self.assertEqual(len(raw), 1)
        self.assertIn("等待识别", raw[0][1])
        self.assertFalse(recording.done())
        release.set()
        await asyncio.wait_for(recording, timeout=5)
        enriched = await self.rows()
        self.assertEqual([(r[0], r[2]) for r in enriched], [(r[0], r[2]) for r in raw])
        self.assertIn("一只猫", enriched[0][1])
        self.assertTrue(await chat.should_respond(event))

    async def test_clear_during_vision_discards_old_enrichment_and_passive_reply(self):
        entered, release = self.blocked_vision()
        event = self.event(Message("这张是什么") + MessageSegment.image(self.image_url))
        recording = self.spawn_record(event)
        await asyncio.wait_for(entered.wait(), timeout=5)
        self.assertEqual(len(await self.rows()), 1)
        await self.manager.clear_history(123, self.group_id, full=True)
        release.set()
        await asyncio.wait_for(recording, timeout=5)
        self.assertEqual(await self.rows(), [])
        self.assertTrue(event._llm_record_obsolete)
        self.assertFalse(await chat.should_respond(event))
        self.bot.send.assert_not_awaited()

    async def test_slow_sentiment_does_not_block_direct_record_or_reply_eligibility(self):
        entered, release = asyncio.Event(), asyncio.Event()
        self.gates.append(release)

        async def analyze(*args, **kwargs):
            entered.set()
            await release.wait()

        event = self.event("请帮我看一下这个问题")
        with (
            patch.object(chat.config, "enable_sentiment_analysis", True),
            patch.object(chat.sentiment_analyzer, "analyze", AsyncMock(side_effect=analyze)),
        ):
            await asyncio.wait_for(chat.record_group_message(self.bot, event), timeout=5)
            await asyncio.wait_for(entered.wait(), timeout=5)
            self.assertEqual(len(await self.rows()), 1)
            self.assertTrue(await chat.should_respond(event))
            release.set()
            await self.finish_background()

    async def test_reset_after_recorder_returns_rejects_the_same_event_at_chat_matcher(self):
        event = self.event("请回答这个问题")
        await chat.record_group_message(self.bot, event)
        self.assertTrue(await chat.should_respond(event))
        await self.manager.clear_history(123, self.group_id, full=True)
        self.assertFalse(await chat.should_respond(event))
        with (
            patch.object(chat, "resolve_incoming_message", AsyncMock()) as resolve,
            patch.object(chat, "add_to_aggregation_buffer", AsyncMock()) as aggregate,
            patch.object(chat, "handle_single_chat", AsyncMock()) as single,
        ):
            await chat.handle_chat(self.bot, event)
        resolve.assert_not_awaited()
        aggregate.assert_not_awaited()
        single.assert_not_awaited()
        self.assertTrue(event._llm_record_obsolete)
        self.assertEqual(await self.rows(), [])

    async def test_reset_during_chat_download_prevents_image_cache_and_aggregation(self):
        event = self.event(Message("看看这张图") + MessageSegment.image(self.image_url))
        await chat.record_group_message(self.bot, event)
        entered, release = asyncio.Event(), asyncio.Event()
        self.gates.append(release)

        async def download(urls):
            entered.set()
            await release.wait()
            return [(url, "c3ludGhldGljLWltYWdl") for url in urls]

        with (
            patch.object(chat, "download_images_as_base64", AsyncMock(side_effect=download)),
            patch.object(chat, "_capture_incoming_images") as capture,
            patch.object(chat.config, "enable_message_aggregation", True),
            patch.object(chat, "add_to_aggregation_buffer", AsyncMock()) as aggregate,
            patch.object(chat, "handle_single_chat", AsyncMock()) as single,
        ):
            task = asyncio.create_task(chat.handle_chat(self.bot, event))
            self.tasks.append(task)
            await asyncio.wait_for(entered.wait(), timeout=5)
            await self.manager.clear_history(123, self.group_id, full=True)
            release.set()
            await asyncio.wait_for(task, timeout=5)
        capture.assert_not_called()
        aggregate.assert_not_awaited()
        single.assert_not_awaited()
        self.assertFalse(await chat.should_respond(event))
        self.assertEqual(await self.rows(), [])

    async def test_background_image_capacity_keeps_busy_placeholder_without_extra_vision(self):
        entered, release = self.blocked_vision()
        with patch.object(chat.config, "incoming_image_max_pending", 1):
            await chat.record_group_message(self.bot, self.event(MessageSegment.image(self.image_url), direct=False))
            await asyncio.wait_for(entered.wait(), timeout=5)
            image_tasks = set(chat._incoming_image_tasks)
            all_tasks = set(chat._incoming_tasks)
            self.assertEqual(len(image_tasks), 1)
            self.assertEqual(chat.model_manager.analyze_image_for_meme.await_count, 1)
            second = self.event(MessageSegment.image("https://example.com/second.png"), message_id=2, direct=False)
            await asyncio.wait_for(chat.record_group_message(self.bot, second), timeout=5)
            rows = await self.rows()
            self.assertEqual(len(rows), 2)
            self.assertIn("后台繁忙", rows[1][1])
            self.assertEqual(second._llm_memory_content, rows[1][1])
            self.assertEqual(chat._incoming_image_tasks, image_tasks)
            self.assertEqual(chat._incoming_tasks, all_tasks)
            self.assertEqual(chat.model_manager.analyze_image_for_meme.await_count, 1)
            self.assertEqual(chat.download_images_as_base64.await_count, 1)
            release.set()
            await self.finish_background()
        self.assertIn("后台繁忙", (await self.rows())[1][1])

    async def test_rejected_enrichment_does_not_publish_to_proactive_cache(self):
        event = self.event(MessageSegment.forward("synthetic-forward"), direct=False)
        with patch.object(self.manager, "enrich_message", AsyncMock(return_value=False)) as enrich:
            await chat.record_group_message(self.bot, event)
        enrich.assert_awaited_once()
        self.assertIn("转发中的实际内容", enrich.await_args.args[2])
        chat.proactive_manager.add_recent_message.assert_not_called()
        chat.proactive_manager.increment_counter.assert_not_called()
        chat.proactive_manager.try_proactive.assert_not_awaited()
        rows = await self.rows()
        self.assertIn("等待展开", rows[0][1])
        self.assertEqual(event._llm_memory_content, rows[0][1])

    async def test_existing_aggregation_worker_accepts_a_new_generation_after_reset(self):
        accepted = []

        async def merged(_bot, _group_id, batch):
            conversation.ensure_current_session()
            accepted.extend(item.message for item in batch)
            await self.manager.add_message(0, "assistant", "已处理新批次", self.group_id)

        with (
            patch.object(chat.config, "aggregation_window", 60),
            patch.object(chat.config, "aggregation_max_wait_seconds", 60),
            patch.object(chat, "handle_merged_chat", AsyncMock(side_effect=merged)),
        ):
            with conversation.session_scope(self.session_id):
                await chat.add_to_aggregation_buffer(self.bot, self.event("旧批次"), "旧批次", [], [])
            worker = chat.aggregation_buffers[self.group_id].task
            self.tasks.append(worker)
            await self.manager.clear_history(123, self.group_id, full=True)
            self.assertFalse(worker.done())
            with conversation.session_scope(self.session_id):
                await chat.add_to_aggregation_buffer(
                    self.bot, self.event("新批次", message_id=2), "新批次", [], [],
                )
            buffer = chat.aggregation_buffers[self.group_id]
            self.assertIs(buffer.task, worker)
            buffer.first_received_at = buffer.last_received_at = asyncio.get_running_loop().time() - 100
            buffer.wakeup.set()
            await asyncio.wait_for(worker, timeout=5)
        self.assertEqual(accepted, ["新批次"])
        self.assertEqual([row[1] for row in await self.rows()], ["已处理新批次"])


if __name__ == "__main__":
    unittest.main()
