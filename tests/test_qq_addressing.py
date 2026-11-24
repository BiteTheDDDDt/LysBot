import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()

from nonebot.adapters.onebot.v11 import Message, MessageSegment
from src.plugins.llm_chat.qq_addressing import MessageTarget, select_addressing
from src.plugins.llm_chat.reply_policy import ReplyPolicy
from src.plugins.llm_chat import agent_runner, models
from test_tool_rounds import FakeToolMessage, make_response, make_tool_call


class AddressingTests(unittest.IsolatedAsyncioTestCase):
    def address(self, payload):
        return select_addressing(payload, [MessageTarget(11, 101, "请求者"),
                                 MessageTarget(None, 202, "被叫到的人"),
                                 MessageTarget(33, 303, "旁观者"), MessageTarget(44, 404, "bot")],
                                 bot_self_id="404", allow_reply=True)

    def test_quote_audience_and_multiple_mentions_are_independent(self):
        address = self.address({"reply_to_message_id": 11, "audience_user_ids": [202, 303],
                                "mention_user_ids": [101, 202, 303, 202, 404, 999]})
        self.assertEqual(address.audience, (202, 303))
        self.assertEqual(address.mentions, (202, 303))
        self.assertEqual([s.data["qq"] for s in address.prefix() if s.type == "at"], ["202", "303"])
        self.assertEqual(address.text("后半句", first=False), "后半句")
        self.assertEqual([s.type for s in address.text("[CQ:at,qq=999]", first=False)], ["text"])

    async def test_native_mention_is_not_repeated_in_rendered_text(self):
        from src.plugins.llm_chat.planner import PlannerDecision
        from src.plugins.llm_chat.turns import send_response_turn

        address = self.address({"reply_to_message_id": 11})
        address.add_verified_member(202, "团子🐟", "另一个昵称")
        address.revise({"mention_user_ids": [202]})
        sent = AsyncMock()

        async def send(text, payload):
            await sent(payload)

        turn = await send_response_turn(
            text="@团子，来讲个笑话 @另一个人", decision=PlannerDecision(addressing=address),
            planned_meme=None, send_text=send,
        )
        segments = sent.await_args.args[0]
        self.assertEqual([segment.type for segment in segments], ["reply", "at", "text", "text"])
        self.assertEqual(segments[1].data["qq"], "202")
        self.assertEqual(segments[-1].data["text"], "来讲个笑话 @另一个人")
        self.assertEqual(turn.text_for_memory, "来讲个笑话 @另一个人")
        self.assertEqual(address.strip_repeated_mentions("@团子哥 你好"), "@团子哥 你好")
        self.assertEqual(address.strip_repeated_mentions("@另一个昵称 你好"), "你好")
        self.assertEqual(self.address({}).strip_repeated_mentions("@团子 你好"), "@团子 你好")

    async def test_inline_mention_keeps_position_and_falls_back_for_missing_marker(self):
        from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
        from src.plugins.llm_chat.turns import send_response_turn

        address = self.address({"reply_to_message_id": 11})
        address.add_verified_member(202, "团子")
        address.revise({"mention_user_ids": [202, 303]})
        send = AsyncMock()
        result = await send_response_turn(
            text="我们先定个分工。\n\n笑话就请⟦at1⟧讲，⟦at1⟧准备好了没⟦at999⟧",
            decision=PlannerDecision(addressing=address, parts=[ReplyPart("text"), ReplyPart("text")]),
            planned_meme=None, send_text=send, delay_range=(0, 0),
        )
        first, second = (call.args[1] for call in send.await_args_list)
        self.assertEqual([(seg.type, seg.data.get("qq")) for seg in first],
                         [("reply", None), ("at", "303"), ("text", None), ("text", None)])
        self.assertEqual([(seg.type, seg.data.get("qq")) for seg in second],
                         [("text", None), ("at", "202"), ("text", None), ("text", None), ("text", None)])
        self.assertEqual(second.extract_plain_text(), "笑话就请讲，团子准备好了没")
        self.assertEqual(result.text_for_memory, "我们先定个分工\n笑话就请@团子讲，团子准备好了没")

    def test_revision_inherits_omitted_fields_and_clears_explicit_empty_fields(self):
        address = self.address({"reply_to_message_id": 11, "mention_user_ids": [202]})
        address.revise({"audience_user_ids": [303]})
        self.assertEqual(address.reply_to, 11)
        self.assertEqual(address.mentions, (202,))
        address.revise({"reply_to_message_id": None, "mention_user_ids": []})
        self.assertIsNone(address.reply_to)
        self.assertEqual(address.mentions, ())
        self.assertEqual(address.audience, (303,))
        other = self.address({})
        self.assertEqual(other.audience, ())

    async def test_handoff_addresses_only_verified_recipients_in_one_message(self):
        from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
        from src.plugins.llm_chat.turns import send_response_turn

        address = self.address({"reply_to_message_id": 11, "audience_user_ids": [101],
                                "mention_user_ids": [101], "handoff_user_ids": [202, 303, 404, 999]})
        self.assertEqual(address.audience, (202, 303))
        self.assertEqual(address.mentions, (202, 303))
        self.assertEqual(address.reply_to, 11)
        address.revise({"reply_to_message_id": 33})
        self.assertIsNone(address.reply_to)
        self.assertEqual(address.mentions, (202, 303))

        send = AsyncMock()
        await send_response_turn(
            text="⟦at1⟧ 帮忙看下原始数据。⟦at2⟧ 你看看能不能复核一下",
            decision=PlannerDecision(addressing=address, parts=[ReplyPart("text"), ReplyPart("text")]),
            planned_meme=None, send_text=send, group_chat=True,
        )
        send.assert_awaited_once()
        self.assertEqual([segment.data["qq"] for segment in send.await_args.args[1] if segment.type == "at"],
                         ["202", "303"])

        address.revise({"handoff_user_ids": []})
        self.assertEqual(address.audience, ())
        self.assertEqual(address.mentions, ())

    def test_native_mentions_supply_candidates_but_text_does_not(self):
        from src.plugins.llm_chat.chat import _event_message_targets
        event = SimpleNamespace(user_id=101, message_id=11,
            reply=SimpleNamespace(message_id=33, sender=SimpleNamespace(user_id=303), message=Message("先前的话")),
            message=Message([MessageSegment.at(202), MessageSegment.at("all"),
                             MessageSegment.text("@303 [CQ:at,qq=999]")]))
        candidates = _event_message_targets(event, "测试")
        self.assertEqual([c.user_id for c in candidates], [101, 202, 303])
        self.assertIsNone(candidates[1].message_id)
        self.assertEqual([c.as_candidate()["origin"] for c in candidates], ["current", "mentioned", "quoted"])
        self.assertEqual(candidates[0].as_candidate()["quoted_message_id"], 33)

    def test_execution_retarget_to_bot_parent_quotes_the_current_reply_instead(self):
        prior = MessageTarget(44, 404, "请查完发结果", origin="quoted")
        first = MessageTarget(11, 101, "收到，刚才查询超时了", origin="current", quoted_message_id=44)
        second = MessageTarget(22, 202, "我也在查", origin="current", quoted_message_id=44)
        address = select_addressing({"audience_user_ids": [101]}, [prior, first],
                                    bot_self_id="404", allow_reply=True)
        address.revise({"reply_to_message_id": 44})
        self.assertEqual(address.text("靠谱 辛苦啦")[0].data["id"], "11")

        address = select_addressing({"reply_to_message_id": 44}, [prior, first, second],
                                    bot_self_id="404", allow_reply=True)
        self.assertIsNone(address.reply_to)
        address.revise({"reply_to_message_id": 44, "audience_user_ids": [202]})
        self.assertEqual(address.reply_to, 22)

    async def test_confirmed_handoff_can_check_history_without_doing_the_web_research(self):
        from src.plugins.llm_chat import tools

        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        self.addAsyncCleanup(adapter.client.close)
        runner = agent_runner.AgentRunner("group_7", 90)
        runner.reply_policy = ReplyPolicy(addressing=self.address({"handoff_user_ids": [202]}))
        token = agent_runner.current_run.set(runner)
        available = [{"type": "function", "function": {"name": name}}
                     for name in ("recall_history", "search_web", "read_web", "get_group_facts")]
        try:
            with (
                patch.object(tools, "get_chat_tools", return_value=available),
                patch.object(adapter, "_create_completion", AsyncMock(return_value=make_response(
                    FakeToolMessage([], content="请把原请求交给搭档")))) as create,
            ):
                await adapter.chat([models.Message("user", "让搭档去找份资料")], tools_enabled=True)
            sent = create.await_args.args[1]
            self.assertEqual([tool["function"]["name"] for tool in sent["tools"]],
                             ["recall_history", "get_group_facts"])
        finally:
            agent_runner.current_run.reset(token)

    async def test_execution_plan_revises_shared_delivery_but_not_sent_media(self):
        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        payload = json.dumps({"response_mode": "social", "verify_group_identity": False,
                              "content": "请对方把刚才提到的图发来", "requested_image_from": "202", "addressing": {
                                  "audience_user_ids": [202], "mention_user_ids": [202]}})
        try:
            for delivered in (False, True):
                address = self.address({})
                runner = agent_runner.AgentRunner("group_7", 60)
                runner.reply_policy = ReplyPolicy(addressing=address)
                if delivered:
                    runner.media_delivered = True
                token = agent_runner.current_run.set(runner)
                try:
                    with patch.object(adapter, "_create_completion", AsyncMock(return_value=make_response(
                            FakeToolMessage([], content=payload)))), patch.object(agent_runner, "_runtime_path", None):
                        await adapter.chat([models.Message("user", "叫他来帮忙", sender_id=101)], tools_enabled=True)
                    self.assertEqual(address.mentions, () if delivered else (202,))
                    self.assertEqual(address.audience, () if delivered else (202,))
                    self.assertEqual(runner.requested_image_from, 202)
                finally:
                    agent_runner.current_run.reset(token)
        finally:
            await adapter.client.close()

    async def test_planner_can_mention_a_member_verified_during_the_turn(self):
        from src.plugins.llm_chat import tools

        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        run = agent_runner.AgentRunner("group_7", 60)
        address = self.address({})
        run.reply_policy = ReplyPolicy(addressing=address)
        call = make_tool_call("member", "get_group_facts", {"scope": "member", "user_id": "707"})
        answer = json.dumps({"response_mode": "social", "verify_group_identity": False,
                             "content": "请当前群里的协作对象接话",
                             "addressing": {"audience_user_ids": [707], "mention_user_ids": [707, 999]}})
        bot = SimpleNamespace(self_id=404, get_group_member_info=AsyncMock(return_value={
            "group_id": 8, "user_id": 707, "card": "搭档", "role": "member",
        }))
        token = agent_runner.current_run.set(run)
        session = models.current_session_id.set("group_7")
        tools.set_poke_context(group_id=7, user_id=101, bot=bot)
        try:
            result = json.loads(await tools.execute_tool("get_group_facts", {"scope": "member", "user_id": "707"}))
            self.assertNotEqual(result["status"], "ok")
            self.assertNotIn(707, [row.user_id for row in address.candidates])
            bot.get_group_member_info.reset_mock()
            bot.get_group_member_info.return_value["group_id"] = 7
            with patch.object(adapter, "_create_completion", AsyncMock(side_effect=[
                make_response(FakeToolMessage([call])), make_response(FakeToolMessage([], content=answer)),
            ])):
                await adapter.chat([models.Message("user", "请以前约好的搭档帮忙", sender_id=101)], tools_enabled=True)
            bot.get_group_member_info.assert_awaited_once_with(group_id=7, user_id=707, no_cache=True)
            self.assertEqual(address.audience, (707,))
            self.assertEqual(address.mentions, (707,))
            self.assertEqual(address.text("搭把手")[0].data["qq"], "707")
        finally:
            tools.clear_poke_context()
            models.current_session_id.reset(session)
            agent_runner.current_run.reset(token)
            await adapter.client.close()

    async def test_renderer_receives_final_audience_and_delivery_uses_same_object(self):
        from src.plugins.llm_chat.model_turn import run_stages
        address = self.address({"audience_user_ids": [101]})
        initial_hint = address.system_hint()
        address.add_verified_member(202, "搭档")
        runner = agent_runner.AgentRunner("group_7", 60)
        runner.reply_policy = ReplyPolicy(addressing=address)
        runner.user_question = "你把任务交给他吧"
        runner.checkpoint = AsyncMock()
        runner.validate_scope = AsyncMock()

        async def call(messages, **kwargs):
            if kwargs["role"] == "planner":
                self.assertIn(initial_hint, [m.content for m in messages])
                address.revise({"handoff_user_ids": [202]})
                runner.response_mode = "social"
                return models.ChatResponse("直接请202接任务")
            content = "\n".join(m.content for m in messages)
            self.assertNotIn(initial_hint, content)
            self.assertIn("搭档", content)
            self.assertIn("⟦at1⟧", content)
            self.assertIn("正文直接对这些收话人说", content)
            self.assertIn("本条是转达请求", content)
            self.assertNotIn("mention_user_ids", content)
            self.assertNotIn("reply_to_message_id", content)
            self.assertNotIn("以下是真实事件候选ID", content)
            return models.ChatResponse("这活交给你了")

        token = agent_runner.current_run.set(runner)
        try:
            result = await run_stages(SimpleNamespace(_chat_impl=call),
                [models.Message("system", "人设"), models.Message("user", runner.user_question)],
                model=None, role="chat", caller="test",
                session_id="group_7", allow_fallback=False, required_tool=None, media_followup=True)
            self.assertEqual(address.text(str(result))[0].data["qq"], "202")
        finally:
            agent_runner.current_run.reset(token)
