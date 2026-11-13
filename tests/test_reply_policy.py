import asyncio
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import chat, context_builder, events, proactive, tools
from src.plugins.llm_chat.config import PERSONA_PROMPT, SYSTEM_PROMPT
from src.plugins.llm_chat.models import Message
from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
from src.plugins.llm_chat.reply_policy import (
    DETAILED_REPLY, ORIGINAL_SHORT_RULE, POKE_REPLY, PROACTIVE_REPLY,
    SHORT_REPLY, constrain_plan, current_request_text, enforce_reply_budget,
    select_reply_policy,
)
from src.plugins.llm_chat.turns import build_turn_items, send_response_turn
from src.plugins.llm_chat.utils import clean_response


class ReplyPolicyTests(unittest.TestCase):
    def test_tiny_mode_survives_planning_and_does_not_cut_output(self):
        from src.plugins.llm_chat.planner import _normalize_decision
        from src.plugins.llm_chat.conversation_style import unpack_plan
        decision = _normalize_decision({"action": "reply", "reply_mode": "tiny"}, "reply")
        self.assertEqual(decision.reply_mode, "tiny")
        for scene in ("chat", "proactive", "poke"):
            policy = select_reply_policy([], decision=decision, scene=scene)
            self.assertTrue(policy.tiny)
            self.assertEqual(policy.max_text_parts, 1)
            self.assertIn("一个词或两个短词组合", policy.system_hint())
            self.assertIn("不带标点符号", policy.system_hint())
            self.assertEqual(enforce_reply_budget("这句完整的话不会因为极短模式被截断。", policy),
                             "这句完整的话不会因为极短模式被截断")
        self.assertEqual(unpack_plan('{"response_mode":"social","verify_group_identity":false,"content":"接梗","reply_mode":"tiny"}')["reply_mode"], "tiny")

    def test_short_reply_keeps_qualifying_sentence(self):
        self.assertEqual(enforce_reply_budget("第二赛段已经开播。但不是所有集数都上线了。"),
                         "第二赛段已经开播。但不是所有集数都上线了")
    def test_planner_mode_is_normalized_and_missing_or_invalid_modes_are_short(self):
        from src.plugins.llm_chat.planner import _normalize_decision
        for value in (None, "research", "long", 500, "sentence", "detailed"):
            decision = _normalize_decision({"action": "reply", "reply_mode": value}, "reply")
            self.assertEqual(select_reply_policy(["介绍一下这个项目"], decision=decision).detailed,
                             value == "detailed")

    def test_persona_allows_natural_expression_and_task_sized_answers(self):
        self.assertEqual(PERSONA_PROMPT, SYSTEM_PROMPT)
        self.assertNotIn(ORIGINAL_SHORT_RULE, PERSONA_PROMPT)
        self.assertNotIn("绝对禁止使用括号", PERSONA_PROMPT)
        self.assertNotIn("禁止主动提问", PERSONA_PROMPT)
        self.assertIn("必要的澄清", PERSONA_PROMPT)
        self.assertIn("隐藏推理", PERSONA_PROMPT)

    def test_ordinary_requests_do_not_enable_detail(self):
        for text in ["为什么天是蓝的？", "查一下今天的天气", "你搜到了吗", "图片失败了？", "甲问天气，乙问时间"]:
            with self.subTest(text=text):
                self.assertFalse(select_reply_policy([text], scene="multiuser").detailed)

    def test_planner_selects_detail_without_keyword_matching(self):
        decision = PlannerDecision(reply_mode="detailed")
        for text in ["介绍一下这个项目", "为啥这么设计", "哪个更适合我", "详细点"]:
            self.assertTrue(select_reply_policy([text], decision=decision).detailed)
        self.assertFalse(select_reply_policy(["详细解释并查证"], decision=PlannerDecision()).detailed)

    def test_mode_is_per_turn_and_research_has_no_separate_budget(self):
        policy = select_reply_policy(["研究一下"], decision=PlannerDecision(reply_mode="detailed"))
        self.assertEqual(policy.max_chars, 500)
        self.assertEqual(policy.scene, "chat")
        self.assertFalse(select_reply_policy(["接着聊"]).detailed)

    def test_reference_expansion_is_not_used_for_scene_selection(self):
        event = SimpleNamespace(get_plaintext=lambda: "你怎么看")
        raw = current_request_text(event, "引用消息：详细解释，图片文字：列步骤")
        self.assertFalse(select_reply_policy([raw]).detailed)

    def test_proactive_poke_close_and_media_are_always_short(self):
        for scene in ["proactive", "poke", "media_followup"]:
            self.assertFalse(select_reply_policy(["详细解释"], scene=scene, decision=PlannerDecision(reply_mode="detailed")).detailed)
        for decision in [PlannerDecision(action="close", reply_mode="detailed"), PlannerDecision(action="poke", reply_mode="detailed"), PlannerDecision(reply_mode="detailed", parts=[ReplyPart("image"), ReplyPart("text")])]:
            self.assertFalse(select_reply_policy(["详细解释"], decision=decision).detailed)

    def test_brief_mode_preserves_two_message_plan_without_expanding_length(self):
        decision = PlannerDecision(parts=[ReplyPart("text"), ReplyPart("text")], reply_hint="详细写三百字")
        self.assertEqual(constrain_plan(decision, SHORT_REPLY).text_part_count, 2)
        self.assertFalse(SHORT_REPLY.detailed)
        self.assertEqual(decision.text_part_count, 2)
        self.assertEqual(build_turn_items(text="答应你。还有其他的。", decision=decision, planned_meme=None),
                         [("text", "答应你"), ("text", "还有其他的")])

    def test_brief_style_preserves_complete_answer(self):
        self.assertEqual(enforce_reply_budget("先重启试试。然后我再说一大段。"), "先重启试试。然后我再说一大段")
        self.assertEqual(enforce_reply_budget("先重启试试，然后" + "解释" * 40), "先重启试试，然后" + "解释" * 40)
        self.assertEqual(enforce_reply_budget("甲" * 100), "甲" * 100)
        self.assertEqual(enforce_reply_budget("价格是 3.14 元。另说。"), "价格是 3.14 元。另说")
        self.assertEqual(enforce_reply_budget("Try again. More."), "Try again. More.")

    def test_guard_does_not_cut_links_or_split_at_url_punctuation(self):
        for policy in [SHORT_REPLY, DETAILED_REPLY]:
            for url in ["https://example.test/" + "x" * 700, "[来源](https://example.test/" + "x" * 700 + ")"]:
                output = enforce_reply_budget("来源：" + url, policy)
                self.assertEqual(output, "来源：" + url)
        self.assertEqual(enforce_reply_budget("https://a.cn/?a=1"), "https://a.cn/?a=1")

    def test_explicit_steps_and_conclusion_are_retained(self):
        text = "1. 检查电源。\n2. 重启设备。\n结论：先排查供电。"
        cleaned = clean_response(text, preserve_paragraphs=True)
        self.assertEqual(cleaned, text)
        for count in [1, 2]:
            items = build_turn_items(text=cleaned, decision=PlannerDecision(parts=[ReplyPart("text")] * count),
                                     planned_meme=None, reply_policy=DETAILED_REPLY)
            expected = "1. 检查电源。\n2. 重启设备。\n结论：先排查供电" if count == 1 else "1. 检查电源\n2. 重启设备。\n结论：先排查供电"
            self.assertEqual("\n".join(payload for kind, payload in items), expected)

    def test_detailed_delivery_retains_all_text_parts(self):
        items = build_turn_items(text=("这是一段解释。\n" * 80),
                                 decision=PlannerDecision(parts=[ReplyPart("text"), ReplyPart("text")]),
                                 planned_meme=None, reply_policy=DETAILED_REPLY)
        self.assertEqual(sum(payload.count("这是一段解释") for kind, payload in items), 80)
        self.assertEqual(len(items), 2)

    def test_long_first_step_does_not_collapse_to_number_marker(self):
        result = enforce_reply_budget("1. " + "请检查电源" * 100, DETAILED_REPLY)
        self.assertIn("请检查电源", result)
        self.assertEqual(result, "1. " + "请检查电源" * 100)

    def test_guard_is_idempotent_at_boundaries(self):
        for policy in [SHORT_REPLY, DETAILED_REPLY]:
            for text in ["", "嗯。好。", "🙂" * 400, "甲" * 19 + "，" + "乙" * 400,
                         "先看这里： https://a.example/" + "b" * 400]:
                with self.subTest(policy=policy, text=text[:30]):
                    bounded = enforce_reply_budget(text, policy)
                    self.assertEqual(bounded, text.strip().rstrip("。"))
                    self.assertEqual(enforce_reply_budget(bounded, policy), bounded)


class ReplyContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_multi_message_request_is_delivered_as_two_messages(self):
        from src.plugins.llm_chat.planner import _normalize_decision

        decision = _normalize_decision({
            "action": "reply", "reply_mode": "sentence",
            "parts": [{"type": "text"}, {"type": "text"}],
        }, "reply")
        policy = select_reply_policy(["你来试试，一次发多句"], decision=decision)
        send = AsyncMock()
        delivered = await send_response_turn(
            text=clean_response("第一条。\n第二条。", preserve_paragraphs=policy.detailed),
            decision=decision, planned_meme=None, reply_policy=policy,
            send_text=send, delay_range=(0, 0),
        )
        self.assertFalse(policy.detailed)
        self.assertEqual([call.args for call in send.await_args_list],
                         [("第一条", True), ("第二条", False)])
        self.assertEqual(delivered.text_for_memory, "第一条\n第二条")

    async def test_ordinary_short_reply_does_not_split_on_newline(self):
        send = AsyncMock()
        await send_response_turn(
            text="第一句\n第二句", decision=PlannerDecision(parts=[ReplyPart("text")]),
            planned_meme=None, send_text=send, delay_range=(0, 0),
        )
        send.assert_awaited_once_with("第一句\n第二句", True)

    async def test_context_pressure_preserves_question_after_long_quote(self):
        current = chat.ResolvedIncomingMessage(
            plain_text="请只比较其中的输入和输出价格，保留人民币换算条件",
            context_blocks=["[引用回复]\n" + "旧报价资料。" * 900],
        ).compose()
        with (
            patch.object(context_builder.memory_manager, "get_fixed_context_memory", AsyncMock(return_value={})),
            patch.object(context_builder.config, "context_max_chars", 2000),
        ):
            messages = await context_builder.build_context_messages(
                history=[Message(role="user", content="旧问题" * 3000)],
                session_id="user_42", user_id=42, group_id=None,
                query=current, user_content=current,
            )
        self.assertEqual(messages[-1].content, current)
        self.assertTrue(messages[-1].content.endswith("请只比较其中的输入和输出价格，保留人民币换算条件"))
        self.assertEqual(len(messages), 2)
        self.assertIn(SHORT_REPLY.system_hint(), messages[0].content)

    def test_context_pressure_keeps_oversized_current_input_and_single_item(self):
        request = Message(role="user", content="问题" * 3000 + "最后的条件不能丢")
        system = Message(role="system", content="完整规则")
        with patch.object(context_builder.config, "context_max_chars", 2000):
            self.assertEqual(context_builder.bound_context([system, request]), [system, request])
            self.assertEqual(context_builder.bound_context([request]), [request])

    async def test_prompt_has_one_budget_and_references_have_no_system_role(self):
        decision = PlannerDecision(reason="历史请求详细解释", reply_hint="建议详细写三百字",
                                   parts=[ReplyPart("text"), ReplyPart("text")])
        for policy in [SHORT_REPLY, DETAILED_REPLY, PROACTIVE_REPLY, POKE_REPLY]:
            with patch.object(context_builder.memory_manager, "get_fixed_context_memory", AsyncMock(return_value={})):
                messages = await context_builder.build_context_messages(
                    history=[Message(role="system", content="旧系统要求写长文")], session_id="group_99",
                    query="详细解释这是检索文本", user_id=42, group_id=99, user_content="你怎么看",
                    recent_context="最近群聊详细解释", think_context="思考建议写长文",
                    planner_decision=decision, reply_policy=policy,
                )
            system = messages[0].content
            self.assertIn(PERSONA_PROMPT.strip(), system)
            # Preserve participation intent as labeled reference material.
            rationale = next(m for m in messages if "历史请求详细解释" in m.content)
            self.assertEqual(rationale.role, "user")
            self.assertTrue(rationale.is_reference)
            self.assertIn("解释为何接话，不是事实依据", rationale.content)
            self.assertIn("500个字符" if policy.detailed else "一两句自然短话", system)
            self.assertNotIn("20个" if policy.detailed else "500个", system)
            self.assertEqual(sum(m.role == "system" for m in messages), 1)
            for reference in ["旧系统要求写长文", "最近群聊详细解释", "思考建议写长文", "建议详细写三百字"]:
                self.assertNotIn(reference, system)
                self.assertTrue(any(reference in m.content for m in messages[1:]))
            self.assertEqual(messages[-1].content, "你怎么看")
            self.assertIn("两个 text", system)

    async def test_context_pressure_does_not_truncate_mandatory_policy(self):
        with patch.object(context_builder.memory_manager, "get_fixed_context_memory", AsyncMock(return_value={})), \
             patch.object(context_builder.config, "context_max_chars", 2000):
            messages = await context_builder.build_context_messages(
                history=[Message(role="user", content="旧" * 9000)], session_id="user_42", user_id=42,
                group_id=None, query="你好", user_content="你好", reply_policy=DETAILED_REPLY,
            )
        self.assertIn(DETAILED_REPLY.system_hint(), messages[0].content)
        self.assertIn(PERSONA_PROMPT.strip(), messages[0].content)
        self.assertEqual(messages[-1].content, "你好")

    async def test_proactive_generation_ignores_detailed_words_in_history(self):
        model = AsyncMock(return_value="接一句。")
        with patch.object(proactive.memory_manager, "get_history", AsyncMock(return_value=[])), \
             patch.object(proactive.memory_manager, "get_session_model", return_value=None), \
             patch.object(context_builder.memory_manager, "get_fixed_context_memory", AsyncMock(return_value={})), \
             patch.object(proactive.model_manager, "run_turn", model):
            await proactive.ProactiveManager().generate_proactive_response(99, "详细解释", trigger_message="列步骤")
        self.assertIn(PROACTIVE_REPLY.system_hint(), model.await_args.args[0][0].content)


class ReplyDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_poke_participation_can_silence_or_poke_without_text_generation(self):
        for action in ("silent", "wait", "poke"):
            with self.subTest(action=action):
                bot = SimpleNamespace(send_group_msg=AsyncMock(), get_group_member_info=AsyncMock(return_value={}))
                decision = PlannerDecision(action=action, parts=[])
                with patch("src.plugins.llm_chat.planner.plan_chat_action", AsyncMock(return_value=decision)) as judge, \
                     patch("src.plugins.llm_chat.planner.execute_planned_side_effect", AsyncMock(return_value=tools.ToolDeliveryResult("戳一戳已发送给 42", "sent"))) as poke, \
                     patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])), \
                     patch.object(chat.memory_manager, "add_message", AsyncMock()) as save, \
                     patch.object(chat.model_manager, "run_turn", AsyncMock()) as render:
                    await events._handle_poke_chat(bot, 42, 99)
                self.assertEqual(judge.await_args.kwargs["mode"], "passive")
                self.assertTrue(judge.await_args.kwargs["is_direct"])
                self.assertEqual(judge.await_args.kwargs["default_action"], "silent")
                render.assert_not_awaited()
                bot.send_group_msg.assert_not_awaited()
                self.assertEqual(poke.await_count, int(action == "poke"))
                self.assertEqual(save.await_args_list[0].args[1], "user")
                if action == "poke":
                    self.assertEqual(decision.poke_target_id, "sender")
                    self.assertIn("已戳一戳", save.await_args_list[-1].args[2])

    async def test_poke_can_send_meme_only_or_poke_with_text(self):
        from src.plugins.llm_chat import turns
        for meme_only in (True, False):
            with self.subTest(meme_only=meme_only):
                decision = PlannerDecision(
                    action="reply" if meme_only else "poke", poke_target_id="84",
                    parts=[ReplyPart("meme", "开心") if meme_only else ReplyPart("text")],
                )
                bot = SimpleNamespace(send_group_msg=AsyncMock(), get_group_member_info=AsyncMock(return_value={}))
                meme = {"id": 1, "description": "开心招手"}
                tools.set_meme_context(group_id=99, user_id=42, bot=bot)
                async def send_meme(*args, **kwargs):
                    tools._meme_context.get()["last_sent"] = meme
                    tools._meme_context.get()["sent_in_round"] = True
                    return tools.ToolDeliveryResult("表情包", "sent")
                try:
                    with patch("src.plugins.llm_chat.planner.plan_chat_action", AsyncMock(return_value=decision)), \
                         patch("src.plugins.llm_chat.planner.execute_planned_side_effect", AsyncMock(return_value=tools.ToolDeliveryResult("戳一戳已发送给 84", "sent"))) as poke, \
                         patch.object(context_builder, "build_context_messages", AsyncMock(return_value=[])), \
                         patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])), \
                         patch.object(chat.memory_manager, "get_session_model", return_value=None), \
                         patch.object(chat.memory_manager, "add_message", AsyncMock()) as save, \
                         patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="来了")) as render, \
                         patch.object(turns, "prepare_planned_meme", AsyncMock(return_value=meme if meme_only else None)), \
                         patch.object(tools, "send_picked_meme", AsyncMock(side_effect=send_meme)) as send, \
                         patch.object(proactive.proactive_manager, "mark_just_spoke") as mark:
                        await events._handle_poke_chat(bot, 42, 99)
                    if meme_only:
                        render.assert_not_awaited()
                        poke.assert_not_awaited()
                        send.assert_awaited_once()
                        bot.send_group_msg.assert_not_awaited()
                    else:
                        self.assertEqual(decision.poke_target_id, "84")
                        poke.assert_awaited_once()
                        bot.send_group_msg.assert_awaited_once()
                        self.assertIn("QQ:84", save.await_args_list[1].args[2])
                        self.assertEqual(save.await_args_list[-1].args[2], "来了")
                    mark.assert_called_with(99)
                finally:
                    tools.clear_meme_context()

    async def test_poke_burst_is_dropped_before_queue_without_fixed_cooldown(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def respond(*args):
            entered.set()
            await release.wait()
        first = SimpleNamespace(self_id=900, target_id=900, user_id=42, group_id=99)
        second = SimpleNamespace(self_id=900, target_id=900, user_id=84, group_id=99)
        events._poke_inflight.clear()
        try:
            with patch.object(events, "_handle_poke_chat", AsyncMock(side_effect=respond)) as reply:
                task = asyncio.create_task(events.handle_poke(SimpleNamespace(), first))
                await entered.wait()
                try:
                    await events.handle_poke(SimpleNamespace(), second)
                    self.assertEqual(reply.await_count, 1)
                finally:
                    release.set()
                    await task
                await events.handle_poke(SimpleNamespace(), second)
                self.assertEqual(reply.await_count, 2)
            self.assertFalse(events._poke_inflight)
        finally:
            events._poke_inflight.clear()

    async def test_group_poke_reply_has_no_at_segment(self):
        bot = SimpleNamespace(send_group_msg=AsyncMock(), get_group_member_info=AsyncMock(return_value={}))
        with patch("src.plugins.llm_chat.planner.plan_chat_action", AsyncMock(return_value=PlannerDecision(action="reply"))), \
             patch.object(context_builder, "build_context_messages", AsyncMock(return_value=[])), \
             patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])), \
             patch.object(chat.memory_manager, "get_session_model", return_value=None), \
             patch.object(chat.memory_manager, "add_message", AsyncMock()), \
             patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="痒啦")), \
             patch("src.plugins.llm_chat.model_identity.update_group_card_for_response", AsyncMock()):
            await events._handle_poke_chat(bot, 42, 99)
        payload = bot.send_group_msg.await_args.kwargs["message"]
        self.assertEqual(payload, "痒啦")

    def tearDown(self):
        tools.clear_poke_context()
        tools.clear_meme_context()
        tools.clear_voice_context()
        tools.clear_image_context()

    async def test_sent_turn_memory_contains_only_bounded_text(self):
        sender = AsyncMock()
        turn = await send_response_turn(text="只接这一句。后面是模型多说的。", decision=None,
                                        planned_meme=None, send_text=sender)
        sender.assert_awaited_once_with("只接这一句。后面是模型多说的", True)
        self.assertEqual(turn.text_for_memory, "只接这一句。后面是模型多说的")

    async def test_batch_uses_planner_mode_and_retains_paragraphs(self):
        for detailed, second_id in [(False, 42), (False, 84), (True, 42)]:
            first = chat.PendingMessage(user_id=42, sender_name="甲", message="怎么回事？", event=SimpleNamespace())
            second = chat.PendingMessage(user_id=second_id, sender_name="乙" if second_id == 84 else "甲",
                                         message="详细解释一下" if detailed else "有结果了吗", event=SimpleNamespace())
            first.planner_decision = PlannerDecision(reply_mode="detailed" if detailed else "sentence")
            model_text = "1. 检查电源。\n2. 重启设备。\n结论：先排查供电。" if detailed else "先重启试试。然后可以检查其他地方。"
            context = AsyncMock(return_value=[])
            saved = AsyncMock()
            bot = SimpleNamespace(send=AsyncMock())
            with patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])), \
                 patch.object(chat.memory_manager, "get_session_model", return_value="test-model"), \
                 patch.object(chat.memory_manager, "add_message", saved), \
                 patch.object(chat, "execute_planned_side_effect", AsyncMock()), \
                 patch.object(chat, "prepare_planned_meme", AsyncMock(return_value=None)), \
                 patch.object(chat, "build_context_messages", context), \
                 patch.object(chat.model_manager, "run_turn", AsyncMock(return_value=model_text)), \
                 patch("src.plugins.llm_chat.model_identity.update_group_card_for_response", AsyncMock()), \
                 patch.object(chat, "_finalize_immediate_tool_turn", AsyncMock(return_value=False)), \
                 patch.object(chat.proactive_manager, "mark_just_spoke"):
                await chat._do_handle_merged_chat(bot, 99, [first, second], primary_user_id=42)
            args = context.await_args.kwargs
            self.assertEqual(args["reply_policy"].detailed, detailed)
            self.assertEqual(args["reply_policy"].scene, "multiuser" if second_id != 42 else "chat")
            self.assertNotIn("多人提问", args["user_content"])
            self.assertNotIn("分别回应每一个人", args["extra_system"])
            if second_id == 42:
                self.assertEqual(args["extra_system"], "")
            actual = bot.send.await_args.args[1]
            self.assertEqual(actual, model_text.rstrip("。"))
            self.assertEqual(saved.await_args.args[2], actual)

    async def test_direct_meme_only_skips_generation_in_single_and_batch(self):
        from src.plugins.llm_chat.planner import ReplyPart
        from src.plugins.llm_chat.turns import SentTurn
        for batch, picked in ((False, True), (True, True), (False, False)):
            event = SimpleNamespace(get_plaintext=lambda: "笑死")
            decision = PlannerDecision(parts=[ReplyPart("meme", "funny")])
            meme = {"id": 1, "topic": "funny"}
            bot = SimpleNamespace(send=AsyncMock())
            with (
                patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])),
                patch.object(chat.memory_manager, "get_session_model", return_value="test-model"),
                patch.object(chat.memory_manager, "add_message", AsyncMock()),
                patch.object(chat, "execute_planned_side_effect", AsyncMock()),
                patch.object(chat, "prepare_planned_meme", AsyncMock(return_value=meme if picked else None)),
                patch.object(chat, "build_context_messages", AsyncMock(return_value=[])),
                patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="哈哈")) as generate,
                patch.object(chat, "send_response_turn", AsyncMock(return_value=SentTurn([], meme))) as send,
                patch("src.plugins.llm_chat.model_identity.update_group_card_for_response", AsyncMock()),
                patch.object(chat, "_finalize_immediate_tool_turn", AsyncMock(return_value=False)),
                patch.object(chat.proactive_manager, "mark_just_spoke"),
            ):
                if batch:
                    message = chat.PendingMessage(user_id=42, sender_name="甲", message="笑死", event=event)
                    message.planner_decision = decision
                    await chat._do_handle_merged_chat(bot, 99, [message], primary_user_id=42)
                else:
                    await chat._do_handle_single_chat(bot, event, "笑死", [], [], 42, planner_decision=decision)
            self.assertEqual(generate.await_count, 0 if picked else 1)
            self.assertEqual(send.await_args.kwargs["text"], "" if picked else "哈哈")
            self.assertEqual(send.await_args.kwargs["planned_meme"], meme if picked else None)

    async def test_single_delivery_uses_planner_mode_and_retains_paragraphs(self):
        event = SimpleNamespace(get_plaintext=lambda: "看看这个")
        context, saved = AsyncMock(return_value=[]), AsyncMock()
        bot = SimpleNamespace(send=AsyncMock())
        with patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])), \
             patch.object(chat.memory_manager, "get_session_model", return_value="test-model"), \
             patch.object(chat.memory_manager, "add_message", saved), \
             patch.object(chat, "execute_planned_side_effect", AsyncMock()), \
             patch.object(chat, "prepare_planned_meme", AsyncMock(return_value=None)), \
             patch.object(chat, "build_context_messages", context), \
             patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="是这样。\n再展开讲。")), \
             patch.object(chat, "_finalize_immediate_tool_turn", AsyncMock(return_value=False)):
            await chat._do_handle_single_chat(bot, event, "介绍一下", [], [], 42, planner_decision=PlannerDecision(reply_mode="detailed"))
        self.assertTrue(context.await_args.kwargs["reply_policy"].detailed)
        bot.send.assert_awaited_once_with(event, "是这样。\n再展开讲")
        self.assertEqual(saved.await_args.args[2], "是这样。\n再展开讲")

    async def test_media_followup_guard_preserves_media_memory(self):
        for kind in ["voice", "image"]:
            sender, saved = AsyncMock(), AsyncMock()
            get_tool = f"get_last_sent_{kind}"
            finalize = getattr(chat, f"_finalize_{kind}_tool_turn")
            with patch.object(tools, get_tool, return_value={"text": "语音本体不缩短", "prompt": "图片本体不缩短"}), \
                 patch.object(chat.memory_manager, "add_message", saved):
                result = await finalize(42, None, followup_text="给你啦。再解释一大段。", send_followup=sender)
            self.assertTrue(result)
            sender.assert_awaited_once_with("给你啦。再解释一大段")
            self.assertIn("给你啦", saved.await_args.args[2])
            self.assertIn("再解释", saved.await_args.args[2])

    async def test_poke_delivery_and_memory_use_short_scene(self):
        bot = SimpleNamespace(send_private_msg=AsyncMock())
        context, saved = AsyncMock(return_value=[]), AsyncMock()
        with patch("src.plugins.llm_chat.planner.plan_chat_action", AsyncMock(return_value=PlannerDecision(action="reply"))), \
             patch.object(context_builder, "build_context_messages", context), \
             patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])), \
             patch.object(chat.memory_manager, "get_session_model", return_value=None), \
             patch.object(chat.memory_manager, "add_message", saved), \
             patch.object(chat.model_manager, "run_turn", AsyncMock(return_value="别戳啦。让我再说两句。")):
            await events._handle_poke_chat(bot, 42)
        self.assertEqual(context.await_args.kwargs["reply_policy"], POKE_REPLY)
        bot.send_private_msg.assert_awaited_once_with(user_id=42, message="别戳啦。让我再说两句")
        self.assertEqual(saved.await_args.args[2], "别戳啦。让我再说两句")


if __name__ == "__main__":
    unittest.main()
