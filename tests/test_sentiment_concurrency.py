import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import conversation, models
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.sentiment import SentimentAnalyzer


class SentimentConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.analyzer = SentimentAnalyzer()

    async def asyncTearDown(self):
        await self.analyzer.close()

    async def test_cancelled_caller_does_not_cancel_shared_analysis_or_other_caller(self):
        entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def upstream(*args, **kwargs):
            entered.set()
            try:
                await release.wait()
                return "positive"
            finally:
                stopped.set()

        with patch.object(models.model_manager, "chat", side_effect=upstream) as call:
            first = asyncio.create_task(self.analyzer.analyze("机器人真可爱", 101, 9811, is_to_bot=True))
            second = asyncio.create_task(self.analyzer.analyze("机器人真可爱", 202, 9811, is_to_bot=True))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertFalse(stopped.is_set())
                self.assertFalse(second.done())
                release.set()
                result = await asyncio.wait_for(second, 1)
            finally:
                release.set()
                await asyncio.gather(first, second, return_exceptions=True)
        self.assertEqual((result.sentiment_type, result.confidence), ("positive", 0.6))
        self.assertEqual(result.user_id, 202)
        self.assertEqual(call.await_count, 1)
        self.assertEqual(call.await_args.kwargs["session_id"], "group_9811")
        self.assertEqual([r.user_id for r in self.analyzer._group_states[9811].records], [202])
        self.assertFalse(self.analyzer._pending_tasks)

    async def test_shared_upstream_timeout_falls_back_once_for_both_callers(self):
        stopped = asyncio.Event()

        async def upstream(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with (
            patch.object(config, "sentiment_total_timeout", 0.1),
            patch.object(models.model_manager, "chat", side_effect=upstream) as call,
        ):
            results = await asyncio.wait_for(asyncio.gather(
                self.analyzer.analyze("机器人别吵", 101, 9812, is_to_bot=True),
                self.analyzer.analyze("机器人别吵", 202, 9812, is_to_bot=True),
            ), 1)
            cached = await self.analyzer.analyze("机器人别吵", 303, 9812, is_to_bot=True)
        expected = ("neutral", 0.0)
        self.assertEqual([(r.sentiment_type, r.confidence) for r in [*results, cached]], [expected] * 3)
        self.assertTrue(stopped.is_set())
        self.assertEqual(call.await_count, 1)
        self.assertFalse(self.analyzer._pending_tasks)

    async def test_same_text_is_cached_independently_for_each_session(self):
        async def upstream(*args, **kwargs):
            return "positive" if kwargs["session_id"] == "group_9813" else "negative:高"

        with patch.object(models.model_manager, "chat", side_effect=upstream) as call:
            first, second = await asyncio.gather(
                self.analyzer.analyze("这机器人怎么样", 101, 9813, is_to_bot=True),
                self.analyzer.analyze("这机器人怎么样", 101, 9814, is_to_bot=True),
            )
            cached_first = await self.analyzer.analyze("这机器人怎么样", 202, 9813, is_to_bot=True)
            cached_second = await self.analyzer.analyze("这机器人怎么样", 202, 9814, is_to_bot=True)
        self.assertEqual((first.sentiment_type, cached_first.sentiment_type), ("positive", "positive"))
        self.assertEqual((second.sentiment_type, cached_second.sentiment_type), ("negative", "negative"))
        self.assertEqual(call.await_count, 2)
        self.assertEqual({c.kwargs["session_id"] for c in call.await_args_list}, {"group_9813", "group_9814"})

    async def test_close_cancels_shared_upstream_and_releases_all_callers(self):
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def upstream(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch.object(models.model_manager, "chat", side_effect=upstream) as call:
            callers = [asyncio.create_task(self.analyzer.analyze(
                "机器人还在吗", user, 9815, is_to_bot=True,
            )) for user in (101, 202)]
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(self.analyzer.close(), 1)
                results = await asyncio.wait_for(asyncio.gather(*callers, return_exceptions=True), 1)
            finally:
                for task in callers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*callers, return_exceptions=True)
        self.assertTrue(stopped.is_set())
        self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))
        self.assertEqual(call.await_count, 1)
        self.assertFalse(self.analyzer._pending_tasks)
        self.assertFalse(self.analyzer._llm_cache)
        self.assertFalse(self.analyzer._group_states[9815].records)

    async def test_new_generation_does_not_join_an_obsolete_shared_task(self):
        entered, release = asyncio.Event(), asyncio.Event()
        attempts = 0

        async def upstream(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                entered.set()
                await release.wait()
                # The real provider runner performs this check before fallback.
                conversation.ensure_send_allowed(9816)
            return "positive"

        async def analyze(user):
            with conversation.session_scope("group_9816"):
                return await self.analyzer.analyze("机器人测试消息", user, 9816, is_to_bot=True)

        with patch.object(models.model_manager, "chat", side_effect=upstream) as call:
            old = asyncio.create_task(analyze(101))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                conversation.invalidate("group_9816")
                fresh = await asyncio.wait_for(analyze(202), 1)
                self.assertEqual(fresh.sentiment_type, "positive")
                self.assertFalse(old.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await old
            finally:
                release.set()
                await asyncio.gather(old, return_exceptions=True)
        self.assertEqual(call.await_count, 2)
        self.assertEqual([r.user_id for r in self.analyzer._group_states[9816].records], [202])
        self.assertFalse(self.analyzer._pending_tasks)

    async def test_reset_isolates_cached_results_for_the_same_session_and_text(self):
        with patch.object(models.model_manager, "chat", AsyncMock(side_effect=["positive", "negative:高"])) as call:
            with conversation.session_scope("group_9817"):
                original = await self.analyzer.analyze("机器人测试缓存", 101, 9817, is_to_bot=True)
                cached = await self.analyzer.analyze("机器人测试缓存", 202, 9817, is_to_bot=True)
            conversation.invalidate("group_9817")
            with conversation.session_scope("group_9817"):
                fresh = await self.analyzer.analyze("机器人测试缓存", 303, 9817, is_to_bot=True)
        self.assertEqual((original.sentiment_type, cached.sentiment_type, fresh.sentiment_type),
                         ("positive", "positive", "negative"))
        self.assertEqual(call.await_count, 2)

    async def test_obsolete_caller_cannot_start_analysis_or_update_fast_path_state(self):
        with patch.object(models.model_manager, "chat", AsyncMock()) as call:
            with conversation.session_scope("group_9818"):
                conversation.invalidate("group_9818")
                with self.assertRaises(conversation.TurnSuperseded):
                    await self.analyzer._analyze_with_llm("机器人测试", session_id="group_9818")
                with self.assertRaises(conversation.TurnSuperseded):
                    await self.analyzer.analyze("闭嘴", 101, 9818, is_to_bot=True)
        call.assert_not_awaited()
        self.assertNotIn(9818, self.analyzer._group_states)
        self.assertFalse(self.analyzer._llm_cache)
        self.assertFalse(self.analyzer._pending_tasks)


if __name__ == "__main__":
    unittest.main()
