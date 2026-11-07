import asyncio
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import nonebot
from nonebot.adapters.onebot.v11 import Message as QQMessage

nonebot.init()

from src.plugins.llm_chat import agent_runner, chat, conversation, models, tools
from src.plugins.llm_chat.planner import PlannerDecision


def event(text, *, user_id=42, group_id=9701, message_id=1):
    return SimpleNamespace(
        group_id=group_id, user_id=user_id, self_id=999, message_id=message_id,
        sender=SimpleNamespace(card=str(user_id), nickname=str(user_id)),
        message=QQMessage(text), to_me=True, get_plaintext=lambda: text,
    )


@contextmanager
def chat_dependencies():

    async def context(**kwargs):
        return [models.Message('user', kwargs['user_content'])]
    with ExitStack() as stack:
        for name, value in (('get_history', AsyncMock(return_value=[])), ('get_session_model', lambda *args: None), ('add_message', AsyncMock())):
            stack.enter_context(patch.object(chat.memory_manager, name, value))
        stack.enter_context(patch.object(chat, 'build_context_messages', side_effect=context))
        stack.enter_context(patch.object(chat, 'prepare_planned_meme', AsyncMock(return_value=None)))
        stack.enter_context(patch('src.plugins.llm_chat.model_identity.update_group_card_for_response', AsyncMock()))
        stack.enter_context(patch.object(agent_runner, '_runtime_path', None))
        yield


class TurnLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await chat.close_chat_tasks()
        conversation.start()
        for clear in (tools.clear_poke_context, tools.clear_meme_context,
                      tools.clear_image_context, tools.clear_voice_context):
            clear()

    async def test_correction_arriving_during_render_becomes_the_next_batch(self):
        entered, release = asyncio.Event(), asyncio.Event()
        rendered_questions = []
        bot = SimpleNamespace(send=AsyncMock())

        async def model_call(messages, **kwargs):
            if kwargs["role"] == "planner":
                return models.ChatResponse("已整理回答依据")
            question = next(m.content for m in messages if m.content.startswith("用户本轮问题："))
            rendered_questions.append(question)
            if len(rendered_questions) == 1:
                entered.set()
                await release.wait()
                return models.ChatResponse("第一轮回答")
            return models.ChatResponse("更正后的回答")

        async def resolve(_bot, incoming):
            return chat.ResolvedIncomingMessage(plain_text=incoming.get_plaintext())

        with (
            chat_dependencies(),
            patch.object(chat, "GroupMessageEvent", SimpleNamespace),
            patch.object(chat.config, "enable_message_aggregation", True),
            patch.object(chat.config, "aggregation_window", 0),
            patch.object(chat, "resolve_incoming_message", side_effect=resolve),
            patch.object(chat, "download_images_as_base64", AsyncMock(return_value=[])),
            patch.object(chat, "plan_chat_action", AsyncMock(return_value=PlannerDecision())),
            patch.object(chat.model_manager, "_chat_impl", side_effect=model_call),
        ):
            await chat.handle_chat(bot, event("查询今年的结果"))
            worker = chat.aggregation_buffers[9701].task
            try:
                await asyncio.wait_for(entered.wait(), 2)
                self.assertIn("group_9701", agent_runner.active_runs)
                await chat.handle_chat(bot, event("我说的是去年", message_id=2))
                self.assertFalse(worker.done())
                release.set()
                await asyncio.wait_for(worker, 2)
            finally:
                release.set()
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
        self.assertEqual(len(rendered_questions), 2)
        self.assertIn("查询今年的结果", rendered_questions[0])
        self.assertIn("我说的是去年", rendered_questions[1])
        self.assertEqual([call.args[1] for call in bot.send.await_args_list],
                         ["第一轮回答", "更正后的回答"])

    async def test_started_wait_turns_survive_new_messages_and_remain_shutdown_tracked(self):
        for merged in (False, True):
            with self.subTest(merged=merged):
                entered = asyncio.Event()

                async def executing(*args, **kwargs):
                    entered.set()
                    await asyncio.Event().wait()

                incoming = event("继续这个请求", group_id=9702)
                decisions = [PlannerDecision(action="wait"), PlannerDecision()] if merged else [PlannerDecision()]
                target = "_do_handle_merged_chat" if merged else "_do_handle_single_chat"
                with (
                    patch.object(chat, "GroupMessageEvent", SimpleNamespace),
                    patch.object(chat, "plan_chat_action", AsyncMock(side_effect=decisions)),
                    patch.object(chat.config, "enable_reply_judge", True),
                    patch.object(PlannerDecision, "wait_delay_seconds", new_callable=PropertyMock, return_value=0),
                    patch.object(chat, target, side_effect=executing),
                ):
                    if merged:
                        pending = chat.PendingMessage(42, "甲", "继续这个请求", incoming)
                        await chat.handle_merged_chat(object(), 9702, [pending])
                    else:
                        chat._schedule_planner_wait(bot=object(), event=incoming,
                            message="继续这个请求", images_base64=[], faces=[],
                            decision=PlannerDecision(action="wait"))
                    worker = chat.planner_wait_tasks[9702]
                    try:
                        await asyncio.wait_for(entered.wait(), 2)
                        self.assertNotIn(9702, chat.planner_wait_tasks)
                        self.assertIn(worker, conversation._active_tasks)
                        chat._cancel_planner_wait(9702)
                        self.assertFalse(worker.cancelling())
                        await conversation.close()
                        self.assertTrue(worker.done())
                    finally:
                        conversation.start()
                        if not worker.done():
                            worker.cancel()
                        await asyncio.gather(worker, return_exceptions=True)

    async def test_multiuser_poke_is_replanned_for_the_actual_request_author(self):
        first = chat.PendingMessage(101, "甲", "普通问题", event("普通问题", user_id=101, group_id=9703))
        second = chat.PendingMessage(202, "乙", "戳我", event("戳我", user_id=202, group_id=9703, message_id=2))
        bot = SimpleNamespace(call_api=AsyncMock(return_value={"ok": True}), send=AsyncMock())
        plan = AsyncMock(side_effect=[
            PlannerDecision(action="poke", poke_target_id="sender", parts=[]),
            PlannerDecision(),
            PlannerDecision(action="poke", poke_target_id="sender", parts=[]),
        ])
        with (
            chat_dependencies(),
            patch.object(chat, "plan_chat_action", plan),
            patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="")),
        ):
            await chat.handle_merged_chat(bot, 9703, [first, second])
        bot.call_api.assert_awaited_once_with("send_poke", group_id=9703, user_id=202)
        self.assertEqual(plan.await_count, 3)
        self.assertNotIn("戳我", plan.await_args_list[1].kwargs["current_message"])
        self.assertNotIn("普通问题", plan.await_args_list[2].kwargs["current_message"])

    async def test_reset_during_delivery_checkpoint_prevents_transport_send(self):
        entered, release = asyncio.Event(), asyncio.Event()
        run = agent_runner.AgentRunner("user_9704")
        send = AsyncMock()

        async def checkpoint(*, strict=False):
            if strict:
                entered.set()
                await release.wait()

        @conversation.serialized_turn(lambda: "user_9704")
        async def deliver():
            conversation.ensure_send_allowed()
            await run.deliver("旧回复", send)

        with patch.object(run, "checkpoint", side_effect=checkpoint), \
                patch.object(run, "validate_scope", AsyncMock()):
            task = asyncio.create_task(deliver())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                conversation.invalidate("user_9704")
                release.set()
                await asyncio.wait_for(task, 2)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        send.assert_not_awaited()

    async def test_multiuser_dynamic_poke_rejects_ambiguous_sender_without_api_call(self):
        bot = SimpleNamespace(call_api=AsyncMock(), send_group_msg=AsyncMock())
        tools.set_poke_context(9705, 101, bot, allow_sender_alias=False)
        for alias in ("sender", "SENDER"):
            result = await tools.send_poke(alias)
            self.assertIn("明确指定目标QQ号", result)
        bot.call_api.assert_not_awaited()
        bot.send_group_msg.assert_not_awaited()
        self.assertEqual(tools._poke_context.get()["poke_results"], {})

    async def test_multiuser_dynamic_poke_allows_an_explicit_qq_target(self):
        bot = SimpleNamespace(call_api=AsyncMock(return_value={"ok": True}))
        tools.set_poke_context(9706, 101, bot, allow_sender_alias=False)
        result = await tools.send_poke("202")
        self.assertIn("已发送给 202", result)
        bot.call_api.assert_awaited_once_with("send_poke", group_id=9706, user_id=202)


if __name__ == "__main__":
    unittest.main()
