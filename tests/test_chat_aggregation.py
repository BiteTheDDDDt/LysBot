import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import chat, tools
from src.plugins.llm_chat.planner import PlannerDecision


class AggregatedChatTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.object(chat, "plan_chat_action", AsyncMock(return_value=PlannerDecision(action="reply")))
        self.planner = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        tools.clear_poke_context()
        tools.clear_meme_context()
        tools.clear_voice_context()
        tools.clear_image_context()

    async def asyncTearDown(self):
        await chat.close_chat_tasks()

    async def test_new_message_does_not_cancel_active_batch(self):
        started, release = asyncio.Event(), asyncio.Event()
        batches = []
        async def process(bot, group_id, messages):
            batches.append([message.message for message in messages])
            if len(batches) == 1:
                started.set()
                await release.wait()
        event = SimpleNamespace(group_id=919,user_id=42,sender=SimpleNamespace(card="甲",nickname="甲"))
        with patch.object(chat.config,"aggregation_window",0), patch.object(chat,"handle_merged_chat",AsyncMock(side_effect=process)):
            await chat.add_to_aggregation_buffer(object(),event,"first",[],[])
            await started.wait()
            worker = chat.aggregation_buffers[919].task
            await chat.add_to_aggregation_buffer(object(),event,"second",[],[])
            self.assertIs(chat.aggregation_buffers[919].task, worker)
            self.assertFalse(worker.cancelled())
            release.set()
            await worker
        self.assertEqual(batches, [["first"],["second"]])

    async def test_hard_deadline_flushes_while_quiet_window_is_still_open(self):
        now = asyncio.get_running_loop().time()
        chat.aggregation_buffers[918] = chat.AggregationBuffer(
            messages=[self._pending_message(42,"甲","still typing")],
            first_received_at=now-30,last_received_at=now,
        )
        handle = AsyncMock()
        with patch.object(chat.config,"aggregation_window",60), patch.object(chat.config,"aggregation_max_wait_seconds",15), patch.object(chat,"handle_merged_chat",handle):
            await asyncio.wait_for(chat.delayed_process(object(),918), timeout=0.5)
        handle.assert_awaited_once()

    @staticmethod
    def _pending_message(user_id: int, sender_name: str, message: str):
        return chat.PendingMessage(
            user_id=user_id,
            sender_name=sender_name,
            message=message,
            event=SimpleNamespace(to_me=True),
        )

    async def test_merged_text_reply_uses_first_message_user_context(self):
        first = self._pending_message(42, "甲", "第一条")
        second = self._pending_message(84, "乙", "第二条")
        bot = SimpleNamespace(send=AsyncMock())
        finalize_turn = AsyncMock(return_value=False)
        add_message = AsyncMock()

        with (
            patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])),
            patch.object(chat.memory_manager, "get_session_model", return_value="test-model"),
            patch.object(chat.memory_manager, "add_message", add_message),
            patch.object(chat, "execute_planned_side_effect", AsyncMock()),
            patch.object(chat, "prepare_planned_meme", AsyncMock(return_value=None)),
            patch.object(chat, "build_context_messages", AsyncMock(return_value=[])),
            patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="聚合回复")),
            patch("src.plugins.llm_chat.model_identity.update_group_card_for_response", AsyncMock()),
            patch.object(chat, "_finalize_immediate_tool_turn", finalize_turn),
            patch.object(chat.proactive_manager, "mark_just_spoke") as mark_spoke,
        ):
            await chat.handle_merged_chat(bot, 99, [first, second])

        self.assertEqual(finalize_turn.await_count, 1)
        self.assertEqual(finalize_turn.await_args.args, (42, 99))
        self.assertEqual(finalize_turn.await_args.kwargs["followup_text"], "聚合回复")
        self.assertTrue(callable(finalize_turn.await_args.kwargs["send_followup"]))
        bot.send.assert_awaited_once_with(first.event, "聚合回复")
        add_message.assert_awaited_once_with(0, "assistant", "聚合回复", 99, model_spec=None)
        mark_spoke.assert_called_once_with(99)
        self.planner.assert_awaited_once()
        self.assertIn("第一条", self.planner.await_args.kwargs["current_message"])
        self.assertIn("第二条", self.planner.await_args.kwargs["current_message"])

    async def test_merged_model_error_can_finalize_media_for_primary_user(self):
        pending = self._pending_message(42, "甲", "请发语音")
        bot = SimpleNamespace(send=AsyncMock())
        finalize_turn = AsyncMock(return_value=True)

        with (
            patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])),
            patch.object(chat.memory_manager, "get_session_model", return_value="test-model"),
            patch.object(chat, "execute_planned_side_effect", AsyncMock()),
            patch.object(chat, "prepare_planned_meme", AsyncMock(return_value=None)),
            patch.object(chat, "build_context_messages", AsyncMock(return_value=[])),
            patch.object(
                chat.model_manager,
                "chat",
                AsyncMock(side_effect=RuntimeError("provider failed after tool call")),
            ),
            patch.object(chat, "_finalize_immediate_tool_turn", finalize_turn),
        ):
            await chat.handle_merged_chat(bot, 99, [pending])

        finalize_turn.assert_awaited_once_with(42, 99)
        bot.send.assert_not_awaited()

    async def test_media_timeout_reports_cause_without_large_exception_trace(self):
        pending = self._pending_message(42, "甲", "改这张图")
        bot = SimpleNamespace(send=AsyncMock())
        timeout = chat.ModelTurnTimeout(role="chat", required_tool="generate_image", timeout=600)
        with (
            patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])),
            patch.object(chat.memory_manager, "get_session_model", return_value="test"),
            patch.object(chat, "build_context_messages", AsyncMock(return_value=[])),
            patch.object(chat.model_manager, "run_turn", AsyncMock(side_effect=timeout)),
            patch.object(chat, "_finalize_immediate_tool_turn", AsyncMock(return_value=False)),
            patch.object(chat.logger, "exception") as exception_log,
        ):
            await chat.handle_merged_chat(bot, 99, [pending])
        bot.send.assert_awaited_once_with(pending.event, timeout.user_message)
        exception_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
