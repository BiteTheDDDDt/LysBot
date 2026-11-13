import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import planner, proactive
from src.plugins.llm_chat.proactive_style import PROACTIVE_VALUE_POLICY


class ProactiveValueTests(unittest.IsolatedAsyncioTestCase):
    async def plan(self, payload, *, mode="proactive", message="门票180真贵", **kwargs):
        with (
            patch.object(planner, "_resolve_planner_context", AsyncMock(return_value="")),
            patch.object(planner.model_manager, "chat", AsyncMock(return_value=json.dumps(payload))),
        ):
            return await planner.plan_chat_action(session_id="group_7", group_id=7,
                current_message=message, mode=mode, is_direct=mode != "proactive", **kwargs)

    async def test_empty_or_echo_contribution_blocks_text_and_media(self):
        for fields in [{}, {"proactive_value": "none", "proactive_contribution": "这票价真狠"},
                       {"proactive_value": "new_angle", "proactive_contribution": ""}]:
            decision = await self.plan({"action": "reply", "parts": [{"type": "text"}], **fields})
            self.assertEqual(decision.action, "silent")
            self.assertEqual(decision.parts, [])
        decision = await self.plan({"action": "reply", "parts": [{"type": "image"}]}, message="生成一张图片")
        self.assertEqual(decision.action, "silent")

    async def test_concrete_contribution_is_carried_and_direct_requests_unaffected(self):
        for kind in ["social_reaction", "new_angle", "useful_info", "support"]:
            decision = await self.plan({"action": "reply", "proactive_value": kind,
                                        "proactive_contribution": "具体补充", "parts": [{"type": "text"}]})
            self.assertEqual(decision.action, "reply")
            self.assertIn("具体补充", decision.reply_hint)
        self.assertEqual((await self.plan({"action": "reply"}, mode="passive")).action, "reply")
        self.assertEqual((await self.plan({"action": "wait"})).action, "wait")

    async def test_failed_proactive_planner_aborts_without_default_decision(self):
        with (
            patch.object(planner, "_resolve_planner_context", AsyncMock(return_value="")),
            patch.object(planner.model_manager, "chat", AsyncMock(side_effect=RuntimeError("offline"))),
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                await planner.plan_chat_action(session_id="group_7", group_id=7,
                    current_message="好贵", mode="proactive", default_action="reply")

    async def test_repeat_is_only_available_with_a_detected_candidate(self):
        payload = {"action": "repeat", "parts": [{"type": "text"}]}
        with self.assertRaisesRegex(ValueError, "无效动作"):
            await self.plan(payload)
        decision = await self.plan(payload, repeat_candidate="我也一样")
        self.assertEqual((decision.action, decision.parts), ("repeat", []))

        manager = proactive.ProactiveManager()
        bot = SimpleNamespace(self_id=999, send_group_msg=AsyncMock())
        with (
            patch.object(proactive, "plan_chat_action", AsyncMock(return_value=decision)),
            patch.object(proactive.repeat_manager, "mark_attempted"),
            patch.object(proactive.memory_manager, "add_message", AsyncMock()) as save,
            patch.object(manager, "generate_proactive_response", AsyncMock()) as generate,
        ):
            self.assertTrue(await manager.try_proactive(bot, 7, "我也一样", 42, repeat_candidate="我也一样"))
        bot.send_group_msg.assert_awaited_once()
        self.assertEqual(str(bot.send_group_msg.await_args.kwargs["message"]), "我也一样")
        generate.assert_not_awaited()
        save.assert_awaited_once_with(0, "assistant", "我也一样", 7)

    async def test_idle_plan_is_one_standalone_joke_or_silence(self):
        decision = await self.plan({"action": "reply", "parts": [{"type": "meme"}]}, trigger_kind="idle")
        self.assertEqual((decision.action, decision.meme_part.type, decision.meme_topic), ("reply", "meme", ""))
        decision = await self.plan({"action": "reply", "parts": [{"type": "text"}]}, trigger_kind="idle")
        self.assertEqual(decision.action, "silent")

    async def test_generation_receives_same_policy(self):
        with (
            patch.object(proactive.memory_manager, "get_history", AsyncMock(return_value=[])),
            patch.object(proactive.memory_manager, "get_session_model", return_value=""),
            patch.object(proactive, "build_context_messages", AsyncMock(return_value=[])) as build,
            patch.object(proactive.model_manager, "run_turn", AsyncMock(return_value="一句新内容")),
        ):
            await proactive.ProactiveManager().generate_proactive_response(7, "群友讨论")
        self.assertIn(PROACTIVE_VALUE_POLICY, build.await_args.kwargs["extra_system"])
        self.assertEqual("群友讨论", build.await_args.kwargs["user_content"])
