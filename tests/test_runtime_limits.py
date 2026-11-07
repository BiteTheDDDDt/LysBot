import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import models
from src.plugins.llm_chat.config import config


class RuntimeLimitsTests(unittest.IsolatedAsyncioTestCase):
    async def test_vision_concurrency_limit_is_shared_across_sessions(self):
        manager = models.ModelManager()
        entered, release = asyncio.Event(), asyncio.Event()
        active = peak = 0

        async def candidates(*args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
            try:
                await release.wait()
                return models.ChatResponse(kwargs["session_id"])
            finally:
                active -= 1

        with (
            patch.object(manager, "_vision_slots", None, create=True),
            patch.object(config, "vision_max_concurrency", 2),
            patch.object(config, "background_llm_total_timeout", 3),
            patch.object(manager, "_chat_with_images_candidates", side_effect=candidates) as call,
        ):
            tasks = [asyncio.create_task(manager._chat_with_images_impl(
                [], [str(index)], session_id=f"group_{9820 + index}",
            )) for index in range(6)]
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.sleep(0)
                self.assertEqual(call.await_count, 2)
                release.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
            finally:
                release.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)
        self.assertEqual([str(result) for result in results], [f"group_{9820 + index}" for index in range(6)])

    async def test_vision_timeout_includes_waiting_for_a_slot_and_does_not_leak_it(self):
        manager = models.ModelManager()
        entered, release = asyncio.Event(), asyncio.Event()

        async def candidates(*args, **kwargs):
            if kwargs["session_id"] == "group_holder":
                entered.set()
                await release.wait()
            return models.ChatResponse("识别完成")

        with (
            patch.object(manager, "_vision_slots", None, create=True),
            patch.object(config, "vision_max_concurrency", 1),
            patch.object(config, "background_llm_total_timeout", 3),
            patch.object(manager, "_chat_with_images_candidates", side_effect=candidates) as call,
        ):
            holder = asyncio.create_task(manager._chat_with_images_impl([], ["first"], session_id="group_holder"))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                with patch.object(config, "background_llm_total_timeout", 0.1):
                    with self.assertRaises(asyncio.TimeoutError):
                        await manager._chat_with_images_impl([], ["queued"], session_id="group_queued")
                self.assertEqual(call.await_count, 1)
                self.assertFalse(holder.done())
                release.set()
                await asyncio.wait_for(holder, 1)
                result = await asyncio.wait_for(manager._chat_with_images_impl(
                    [], ["third"], session_id="group_next",
                ), 1)
            finally:
                release.set()
                if not holder.done():
                    holder.cancel()
                await asyncio.gather(holder, return_exceptions=True)
        self.assertEqual(str(result), "识别完成")
        self.assertEqual([call.kwargs["session_id"] for call in call.await_args_list], ["group_holder", "group_next"])

    async def _check_fallback_metrics(self, *, vision, cancelled):
        manager = models.ModelManager()
        candidates = ["openai:runtime-primary", "openai:runtime-second", "openai:runtime-third"]
        first_error = RuntimeError("primary unavailable")
        second_error = asyncio.CancelledError() if cancelled else RuntimeError("secondary unavailable")
        method = "chat_with_images" if vision else "chat"
        adapters = {
            candidates[0]: SimpleNamespace(**{method: AsyncMock(side_effect=first_error)}),
            candidates[1]: SimpleNamespace(**{method: AsyncMock(side_effect=second_error)}),
            candidates[2]: SimpleNamespace(**{method: AsyncMock(return_value=models.ChatResponse("最终成功"))}),
        }
        with (
            patch.object(manager, "_vision_slots", None, create=True),
            patch.object(manager, "_model_cooldowns", {}),
            patch.object(manager, "_resolve_candidates", return_value=(candidates[0], candidates)),
            patch.object(manager, "_extra_for_role", return_value=None),
            patch.object(manager, "get_or_build_adapter", side_effect=adapters.__getitem__),
            patch.object(models, "start_call", side_effect=lambda **kwargs: kwargs["model"]),
            patch.object(models, "finish_call") as finish,
        ):
            request = (manager._chat_with_images_impl([], ["offline-image"], session_id="group_metrics")
                       if vision else manager.chat([], role="chat", session_id="group_metrics"))
            if cancelled:
                with self.assertRaises(asyncio.CancelledError):
                    await request
            else:
                result = await request
                self.assertEqual(str(result), "最终成功")
                self.assertEqual(result.fallback_attempt, 2)
        expected_count = 2 if cancelled else 3
        self.assertEqual([call.args[0] for call in finish.call_args_list], candidates[:expected_count])
        self.assertEqual([call.kwargs["fallback_attempt"] for call in finish.call_args_list], list(range(expected_count)))
        self.assertIs(finish.call_args_list[0].kwargs["error"], first_error)
        self.assertIs(finish.call_args_list[1].kwargs["error"], second_error)
        if cancelled:
            getattr(adapters[candidates[2]], method).assert_not_awaited()

    async def test_failed_chat_and_vision_fallbacks_keep_their_actual_attempt_index(self):
        for vision in (False, True):
            with self.subTest(vision=vision):
                await self._check_fallback_metrics(vision=vision, cancelled=False)

    async def test_cancelled_chat_and_vision_fallbacks_keep_their_actual_attempt_index(self):
        for vision in (False, True):
            with self.subTest(vision=vision):
                await self._check_fallback_metrics(vision=vision, cancelled=True)


if __name__ == "__main__":
    unittest.main()
