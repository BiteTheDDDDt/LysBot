import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import chat, planner, proactive
from src.plugins.llm_chat.planner import PlannerDecision


class PlannerWaitTests(unittest.IsolatedAsyncioTestCase):
    def test_wait_delay_is_clamped_to_120_seconds(self):
        with (
            patch.object(planner.config, "planner_wait_default_seconds", 8),
            patch.object(planner.config, "planner_wait_max_seconds", 120),
        ):
            explicit = planner._normalize_decision(
                {"action": "wait", "wait_seconds": 999},
                "reply",
            )
            defaulted = planner._normalize_decision(
                {"action": "wait", "wait_seconds": 0},
                "reply",
            )

        self.assertEqual(explicit.wait_seconds, 120)
        self.assertEqual(explicit.wait_delay_seconds, 120)
        self.assertEqual(defaulted.wait_delay_seconds, 8)

    async def test_passive_wait_replans_with_latest_context_then_replies(self):
        event = SimpleNamespace(group_id=77, user_id=42, to_me=True, get_plaintext=lambda: "先等等")
        final_decision = PlannerDecision(action="reply", reply_hint="现在接话")
        plan = AsyncMock(return_value=final_decision)
        handle = AsyncMock()

        with (
            patch.object(chat.config, "planner_wait_max_seconds", 120),
            patch.object(chat.config, "planner_wait_max_replans", 3),
            patch.object(chat.asyncio, "sleep", AsyncMock()),
            patch.object(
                chat.proactive_manager,
                "get_recent_messages_text",
                return_value="用户2: 后续补充",
            ),
            patch.object(chat, "plan_chat_action", plan),
            patch.object(chat, "handle_single_chat", handle),
        ):
            await chat._resume_after_planner_wait(
                bot=object(),
                event=event,
                message="先等等",
                images_base64=[],
                faces=[],
                initial_decision=PlannerDecision(action="wait", wait_seconds=15),
            )

        self.assertEqual(plan.await_count, 1)
        self.assertEqual(plan.await_args.kwargs["recent_context"], "用户2: 后续补充")
        handle.assert_awaited_once()
        self.assertIs(handle.await_args.args[-1], final_decision)

    async def test_repeated_wait_stops_at_replan_limit(self):
        event = SimpleNamespace(group_id=78, user_id=42, to_me=True, get_plaintext=lambda: "等等")
        plan = AsyncMock(return_value=PlannerDecision(action="wait", wait_seconds=1))
        handle = AsyncMock()

        with (
            patch.object(chat.config, "planner_wait_max_seconds", 120),
            patch.object(chat.config, "planner_wait_max_replans", 2),
            patch.object(chat.asyncio, "sleep", AsyncMock()),
            patch.object(
                chat.proactive_manager,
                "get_recent_messages_text",
                return_value="仍在连续输入",
            ),
            patch.object(chat, "plan_chat_action", plan),
            patch.object(chat, "handle_single_chat", handle),
        ):
            await chat._resume_after_planner_wait(
                bot=object(),
                event=event,
                message="等等",
                images_base64=[],
                faces=[],
                initial_decision=PlannerDecision(action="wait", wait_seconds=1),
            )

        self.assertEqual(plan.await_count, 2)
        handle.assert_not_awaited()

    async def test_proactive_wait_schedules_a_fresh_planner_pass(self):
        manager = proactive.ProactiveManager()
        gate = asyncio.Event()

        async def gated_sleep(_delay):
            await gate.wait()
        plan = AsyncMock(side_effect=[PlannerDecision(action='wait', wait_seconds=10), PlannerDecision(action='silent')])
        with patch.object(proactive.config, 'enable_proactive', True), patch.object(proactive.config, 'enable_sentiment_analysis', False), patch.object(proactive.config, 'proactive_msg_threshold', 1), patch.object(proactive.config, 'planner_wait_max_seconds', 120), patch.object(proactive.config, 'planner_wait_max_replans', 3), patch.object(manager, 'is_on_cooldown', return_value=False), patch.object(manager, 'is_on_planner_cooldown', return_value=False), patch.object(proactive, 'plan_chat_action', plan), patch.object(proactive.asyncio, 'sleep', side_effect=gated_sleep):
            result = await manager.try_proactive(SimpleNamespace(self_id='999'), group_id=79, message='群友还在分段输入', user_id=101)
            self.assertFalse(result)
            wait_task = manager._wait_tasks[79]
            gate.set()
            await wait_task
        self.assertEqual(plan.await_count, 2)
        self.assertEqual(plan.await_args_list[1].kwargs['recent_context'], '用户101: 群友还在分段输入')
        self.assertNotIn(79, manager._wait_tasks)


if __name__ == "__main__":
    unittest.main()
