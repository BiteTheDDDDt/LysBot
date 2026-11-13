import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from nonebot.adapters.onebot.v11 import Message

nonebot.init()

from src.plugins.llm_chat import chat, followups, planner
from src.plugins.llm_chat.models import Message as LLMMessage
from src.plugins.llm_chat.planner import PlannerDecision


class FollowupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        followups.windows.clear()
        self.addCleanup(followups.windows.clear)
        self.event = SimpleNamespace(group_id=7, user_id=42, self_id=999, to_me=False,
                                     message=Message("那为什么呢"), get_plaintext=lambda: "那为什么呢")

    def open(self):
        followups.answered(7, [42], "因为输入格式不同。", direct=True)

    def test_three_replies_do_not_extend_deadline(self):
        self.open()
        deadline = followups.windows[7].expires
        quiet_deadline = followups.windows[7].quiet_expires
        for i in range(3):
            self.assertTrue(followups.eligible(7, 42))
            followups.answered(7, [42], "补充说明", direct=False)
            self.assertEqual(followups.windows[7].expires, deadline)
            self.assertEqual(followups.windows[7].quiet_expires, quiet_deadline)
            if i < 2:
                self.assertEqual(followups.windows[7].remaining, 2 - i)
        self.assertFalse(followups.eligible(7, 42))

    def test_expiry_and_group_isolation(self):
        with patch.object(followups.time, "monotonic", return_value=100):
            self.open()
            self.assertFalse(followups.eligible(8, 42))
        with patch.object(followups.time, "monotonic", return_value=221):
            self.assertTrue(followups.eligible(7, 42))
        with patch.object(followups.time, "monotonic", return_value=701):
            self.assertFalse(followups.eligible(7, 42))

    def test_interruption_removes_quiet_grace_without_extending_normal_window(self):
        with patch.object(followups.time, "monotonic", return_value=100):
            self.open()
        with patch.object(followups.time, "monotonic", return_value=150):
            followups.observe(7, 43)
            self.assertTrue(followups.eligible(7, 42))
        with patch.object(followups.time, "monotonic", return_value=221):
            self.assertFalse(followups.eligible(7, 42))
        with patch.object(followups.time, "monotonic", return_value=300):
            self.open()
        with patch.object(followups.time, "monotonic", return_value=523):
            followups.observe(7, 43)
            self.assertFalse(followups.eligible(7, 43))
            self.assertFalse(followups.eligible(7, 42))

    def test_zero_window_still_disables_implicit_followups(self):
        with patch.object(followups.config, "followup_window_seconds", 0):
            self.open()
        self.assertFalse(followups.eligible(7, 42))

    def test_other_members_do_not_inherit_window_and_busy_group_closes_it(self):
        self.open()
        followups.observe(7, 43)
        self.assertFalse(followups.eligible(7, 43))
        followups.observe(7, 44)
        self.assertFalse(followups.eligible(7, 44))
        followups.observe(7, 45)
        self.assertFalse(followups.eligible(7, 42))

    def test_explicit_other_target_and_close_stop_continuation(self):
        self.open()
        self.assertFalse(followups.eligible(7, 42, directed_elsewhere=True))
        followups.observe(7, 42, directed_elsewhere=True)
        self.assertIsNone(followups.active(7))
        self.open()
        followups.answered(7, [42], "那先这样", direct=False, closed=True)
        self.assertIsNone(followups.active(7))

    async def test_unmentioned_followup_enters_passive_path(self):
        self.open()
        with patch.object(chat, 'GroupMessageEvent', SimpleNamespace):
            self.assertTrue(await chat.should_respond(self.event))
        followups.windows.clear()
        with patch.object(chat, 'GroupMessageEvent', SimpleNamespace):
            self.assertFalse(await chat.should_respond(self.event))

    async def test_quiet_delayed_answer_reaches_passive_planner_and_delivery(self):
        for message, target, evidence, should_send in [('你来改', 'bot', '回答你刚才问的怎么参与，让你来改 vllm/sglang', True), ('今晚吃面', 'unclear', '没有回应 bot 上条的具体内容', False)]:
            with self.subTest(message=message):
                followups.windows.clear()
                with patch.object(followups.time, 'monotonic', return_value=100):
                    followups.answered(7, [42], '哦vllm sglang这个我有点感兴趣，怎么参与？', direct=True)
                window = followups.windows[7]
                event = SimpleNamespace(group_id=7, user_id=42, self_id=999, to_me=False, message=Message(message), get_plaintext=lambda: message, sender=SimpleNamespace(card='甲', nickname='甲'))
                resolved = SimpleNamespace(plain_text=message, compose=lambda: message, image_urls=[], direct_image_urls=[], faces=[])
                bot = SimpleNamespace(send=AsyncMock())

                def stored_message(_user_id, role, content, *args, **kwargs):
                    return LLMMessage(role=role, content=content, source_id=1)
                proactive = AsyncMock()
                generate = AsyncMock(return_value='把代码发来，我看看')
                plan_call = AsyncMock(return_value=json.dumps({'action': 'reply', 'parts': [{'type': 'text'}], 'followup_target': target, 'followup_evidence': evidence}))
                with ExitStack() as stack:
                    for patcher in (patch.object(followups.time, 'monotonic', return_value=323), patch.object(chat, 'GroupMessageEvent', SimpleNamespace), patch.object(chat, 'resolve_incoming_message', AsyncMock(return_value=resolved)), patch.object(chat, 'download_images_as_base64', AsyncMock(return_value=[])), patch.object(chat, '_capture_incoming_images', return_value=[]), patch.object(chat.config, 'enable_group_repeat', False), patch.object(chat.memory_manager, 'add_message', AsyncMock(side_effect=stored_message)), patch.object(chat.memory_manager, 'enrich_message', AsyncMock(return_value=True)), patch.object(chat.memory_manager, 'get_history', AsyncMock(return_value=[])), patch.object(chat.memory_manager, 'get_session_model', return_value='test-model'), patch.object(chat.proactive_manager, 'add_recent_message'), patch.object(chat.proactive_manager, 'increment_counter'), patch.object(chat.proactive_manager, 'is_explicitly_muted', return_value=False, create=True), patch.object(chat.proactive_manager, 'is_on_cooldown', return_value=True), patch.object(chat.proactive_manager, 'try_proactive', proactive), patch.object(chat.proactive_manager, 'mark_just_spoke'), patch.object(planner, '_resolve_planner_context', AsyncMock(return_value='')), patch.object(planner.model_manager, 'chat', plan_call), patch.object(chat, 'execute_planned_side_effect', AsyncMock()), patch.object(chat, 'prepare_planned_meme', AsyncMock(return_value=None)), patch.object(chat, 'build_context_messages', AsyncMock(return_value=[])), patch.object(chat.model_manager, 'run_turn', generate), patch.object(chat, '_finalize_immediate_tool_turn', AsyncMock(return_value=False)), patch('src.plugins.llm_chat.model_identity.update_group_card_for_response', AsyncMock())):
                        stack.enter_context(patcher)
                    await chat.record_group_message(bot, event)
                    self.assertIs(event._llm_followup_window, window)
                    self.assertTrue(await chat.should_respond(event))
                    await chat.handle_single_chat(bot, event, message, [], [])
                    proactive.assert_not_awaited()
                    plan_call.assert_awaited_once()
                    self.assertEqual(plan_call.await_args.kwargs['caller'], 'planner:passive')
                    prompt = plan_call.await_args.args[0][0].content
                    self.assertIn('怎么参与', prompt)
                    self.assertIn(message, prompt)
                    if should_send:
                        bot.send.assert_awaited_once_with(event, '把代码发来，我看看')
                        generate.assert_awaited_once()
                        self.assertEqual(window.remaining, 2)
                        self.assertEqual(window.quiet_expires, 700)
                    else:
                        bot.send.assert_not_awaited()
                        generate.assert_not_awaited()
                        self.assertIsNone(followups.active(7))

    async def test_implicit_single_can_stay_silent_even_when_reply_judge_disabled(self):
        self.open()
        with (
            patch.object(chat, "GroupMessageEvent", SimpleNamespace),
            patch.object(chat.config, "enable_reply_judge", False),
            patch.object(chat, "plan_chat_action", AsyncMock(return_value=PlannerDecision(action="silent"))) as plan,
        ):
            await chat.handle_single_chat(object(), self.event, "那为什么呢", [], [])
        self.assertFalse(plan.await_args.kwargs["is_direct"])
        self.assertEqual(plan.await_args.kwargs["default_action"], "silent")
        self.assertIsNone(followups.active(7))

    async def test_implicit_batch_can_stay_silent_without_generating(self):
        self.open()
        pending = SimpleNamespace(event=self.event, message="那为什么呢", sender_name="甲", user_id=42,
                                  images_base64=[], faces=[])
        with (
            patch.object(chat.config, "enable_reply_judge", False),
            patch.object(chat, "plan_chat_action", AsyncMock(return_value=PlannerDecision(action="silent"))) as plan,
            patch.object(chat.model_manager, "run_turn", AsyncMock()) as generate,
        ):
            await chat.handle_merged_chat(object(), 7, [pending])
        self.assertFalse(plan.await_args.kwargs["is_direct"])
        generate.assert_not_awaited()

    async def test_implicit_planner_requires_explicit_addressee_evidence(self):
        self.open()
        for fields, expected in [
            ({}, "silent"),
            ({"followup_target": "other", "followup_evidence": "延续同一话题"}, "silent"),
            ({"followup_target": "unclear", "followup_evidence": "时间相邻"}, "silent"),
            ({"followup_target": "bot", "followup_evidence": ""}, "silent"),
            ({"followup_target": "bot", "followup_evidence": "追问你所说的输入格式差异为什么存在"}, "reply"),
        ]:
            with (
                self.subTest(fields=fields),
                patch.object(planner, "_resolve_planner_context", AsyncMock(return_value="")),
                patch.object(planner.model_manager, "chat", AsyncMock(return_value=json.dumps(
                    {"action": "reply", "parts": [{"type": "text"}], **fields}))),
            ):
                result = await planner.plan_chat_action(session_id="group_7", group_id=7,
                    current_message="那为什么呢", is_direct=False, default_action="silent",
                    followup_context=followups.context(7))
                self.assertEqual(result.action, expected)

    async def test_closed_window_drops_queued_implicit_messages(self):
        pending = SimpleNamespace(event=self.event, message="那为什么呢", sender_name="甲", user_id=42,
                                  images_base64=[], faces=[])
        with (
            patch.object(chat, "GroupMessageEvent", SimpleNamespace),
            patch.object(chat, "plan_chat_action", AsyncMock()) as plan,
        ):
            await chat.handle_single_chat(object(), self.event, "那为什么呢", [], [])
            await chat.handle_merged_chat(object(), 7, [pending])
        plan.assert_not_awaited()

    async def test_proactive_reply_cannot_turn_its_trigger_into_a_followup(self):
        self.event.sender = SimpleNamespace(card='甲', nickname='甲')
        resolved = SimpleNamespace(plain_text='xp是自由的', compose=lambda: 'xp是自由的', image_urls=[], direct_image_urls=[], faces=[])

        async def proactive_reply(*args, **kwargs):
            followups.answered(7, [42], 'XP自由，钱包可不自由。', direct=True)
            return True

        def stored_message(_user_id, role, content, *args, **kwargs):
            return LLMMessage(role=role, content=content, source_id=1, timestamp=kwargs.get('timestamp'))
        with patch.object(chat, 'GroupMessageEvent', SimpleNamespace), patch.object(chat, 'resolve_incoming_message', AsyncMock(return_value=resolved)), patch.object(chat, 'download_images_as_base64', AsyncMock(return_value=[])), patch.object(chat, '_capture_incoming_images', return_value=[]), patch.object(chat.config, 'enable_group_repeat', False), patch.object(chat.memory_manager, 'add_message', AsyncMock(side_effect=stored_message)), patch.object(chat.memory_manager, 'enrich_message', AsyncMock(return_value=True)) as enriched, patch.object(chat.proactive_manager, 'add_recent_message'), patch.object(chat.proactive_manager, 'increment_counter'), patch.object(chat.proactive_manager, 'try_proactive', AsyncMock(side_effect=proactive_reply)) as proactive, patch.object(chat, 'plan_chat_action', AsyncMock()) as plan:
            await chat.record_group_message(object(), self.event)
            self.assertIn('xp是自由的', enriched.await_args.args[2])
            proactive.assert_awaited_once()
            self.assertTrue(followups.eligible(7, 42))
            self.assertFalse(await chat.should_respond(self.event))
            pending = SimpleNamespace(event=self.event, message='xp是自由的', sender_name='甲', user_id=42, images_base64=[], faces=[])
            await chat.handle_single_chat(object(), self.event, 'xp是自由的', [], [])
            await chat.handle_merged_chat(object(), 7, [pending])
            plan.assert_not_awaited()
            next_event = SimpleNamespace(**vars(self.event))
            next_event._llm_followup_window = followups.active(7)
            self.assertTrue(await chat.should_respond(next_event))
            self.event.to_me = True
            self.assertTrue(await chat.should_respond(self.event))

    def test_replacement_window_does_not_revive_old_queued_followup(self):
        self.open()
        self.event._llm_followup_window = followups.active(7)
        self.assertTrue(chat._eligible_event_followup(self.event))
        followups.answered(7, [42], "另一次对话的回答", direct=True)
        self.assertFalse(chat._eligible_event_followup(self.event))
