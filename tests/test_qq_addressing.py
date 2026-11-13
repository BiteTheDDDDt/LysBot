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
from test_tool_rounds import FakeToolMessage, make_response


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

    def test_native_mentions_supply_candidates_but_text_does_not(self):
        from src.plugins.llm_chat.chat import _event_message_targets
        event = SimpleNamespace(user_id=101, message_id=11, reply=None,
            message=Message([MessageSegment.at(202), MessageSegment.at("all"),
                             MessageSegment.text("@303 [CQ:at,qq=999]")]))
        candidates = _event_message_targets(event, "测试")
        self.assertEqual([c.user_id for c in candidates], [101, 202])
        self.assertIsNone(candidates[1].message_id)

    async def test_execution_plan_revises_shared_delivery_but_not_sent_media(self):
        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        payload = json.dumps({"response_mode": "social", "verify_group_identity": False,
                              "content": "直接把任务交给对方", "addressing": {
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
                finally:
                    agent_runner.current_run.reset(token)
        finally:
            await adapter.client.close()

    async def test_renderer_receives_final_audience_and_delivery_uses_same_object(self):
        from src.plugins.llm_chat.model_turn import run_stages
        from src.plugins.llm_chat import followups
        address = self.address({"audience_user_ids": [101]})
        initial_hint = address.system_hint()
        runner = agent_runner.AgentRunner("group_7", 60)
        runner.reply_policy = ReplyPolicy(addressing=address)
        runner.user_question = "你把任务交给他吧"
        runner.checkpoint = AsyncMock()
        runner.validate_scope = AsyncMock()

        async def call(messages, **kwargs):
            if kwargs["role"] == "planner":
                address.revise({"audience_user_ids": [202], "mention_user_ids": [202]})
                runner.response_mode = "social"
                return models.ChatResponse("直接请202接任务")
            content = "\n".join(m.content for m in messages)
            self.assertNotIn(initial_hint, content)
            self.assertIn(address.system_hint(), content)
            self.assertNotIn("只使用以下真实事件候选ID", content)
            return models.ChatResponse("这活交给你了")

        token = agent_runner.current_run.set(runner)
        try:
            result = await run_stages(SimpleNamespace(_chat_impl=call),
                [models.Message("system", initial_hint)], model=None, role="chat", caller="test",
                session_id="group_7", allow_fallback=False, required_tool=None, media_followup=True)
            self.assertEqual(address.text(str(result))[0].data["qq"], "202")
            followups.answered(7001, [101, *address.audience, *address.mentions], str(result), direct=True)
            self.assertTrue(followups.eligible(7001, 202))
            self.assertFalse(followups.eligible(7001, 303))
        finally:
            followups.windows.pop(7001, None)
            agent_runner.current_run.reset(token)
