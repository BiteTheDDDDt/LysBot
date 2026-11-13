import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from nonebot.adapters.onebot.v11 import Message as QQMessage

nonebot.init()

from src.plugins.llm_chat import chat, followups, models, planner, proactive
from src.plugins.llm_chat.image_observations import unsolicited_view


class ImageObservationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        followups.windows.clear()
        self.addCleanup(followups.windows.clear)

    def test_legacy_synthetic_request_is_removed_but_real_caption_survives(self):
        legacy = "[甲] 图片1: 一只猫\n图片2: 一只狗\n\n请根据图片内容回复"
        clean = unsolicited_view(legacy)
        self.assertNotIn("请根据", clean)
        self.assertNotIn("一只猫", clean)
        self.assertIn("分享了图片", clean)
        caption = "[甲] 图片1（自动识别，仅作背景）: 一只猫\n\n用户消息: 今天收养的\n还没起名字"
        clean = unsolicited_view(caption)
        self.assertIn("今天收养的\n还没起名字", clean)
        self.assertNotIn("一只猫", clean)
        self.assertEqual(unsolicited_view("请根据图片内容回复"), "请根据图片内容回复")

    def test_proactive_planner_hides_observation_but_direct_planner_retains_it(self):
        source = models.Message("user", "[甲] 图片1: 红色跑车\n\n请根据图片内容回复")
        hidden = planner._format_planner_history([source], current_message="其他消息", max_chars=2000,
                                                 hide_image_observations=True)
        direct = planner._format_planner_history([source], current_message="这车是什么", max_chars=2000)
        self.assertNotIn("红色跑车", hidden)
        self.assertIn("红色跑车", direct)
        self.assertIn("红色跑车", source.content)  # No mutation of stored history.

    def test_hidden_observation_keeps_image_reference_for_readback(self):
        source = "[甲] 图片1（自动识别，仅作背景）: [图片引用:img_example] 芯片题\n\n用户消息: 帮忙看看"
        hidden = unsolicited_view(source)
        self.assertIn("[图片引用:img_example]", hidden)
        self.assertIn("帮忙看看", hidden)
        self.assertNotIn("芯片题", hidden)

    async def test_image_only_does_not_use_followup_window_but_explicit_at_still_works(self):
        followups.answered(7, [42], '刚才的回复', direct=True)
        event = SimpleNamespace(group_id=7, user_id=42, self_id=999, to_me=False, message=QQMessage(), get_plaintext=lambda: '')
        with patch.object(chat, 'GroupMessageEvent', SimpleNamespace):
            self.assertFalse(await chat.should_respond(event))
            event.to_me = True
            self.assertTrue(await chat.should_respond(event))

    async def test_execution_retains_visual_evidence_but_not_old_thought(self):
        manager = proactive.ProactiveManager()
        source = models.Message("user", "[甲] 图片1: 私密截图描述\n\n用户消息: 今天真倒霉")
        old_plan = models.Message("assistant", "用户明确要求根据图片内容回复", is_thinking=True)
        with (
            patch.object(proactive.memory_manager, "get_history", AsyncMock(return_value=[old_plan, source])),
            patch.object(proactive.memory_manager, "get_session_model", return_value=""),
            patch.object(proactive, "build_context_messages", AsyncMock(return_value=[])) as build,
            patch.object(proactive.model_manager, "run_turn", AsyncMock(return_value="太惨了")),
        ):
            await manager.generate_proactive_response(7, source.content, trigger_message=source.content)
        arguments = build.await_args.kwargs
        self.assertNotIn("think_context", arguments)
        self.assertEqual(arguments["history"], [source])
        self.assertIn("私密截图描述", arguments["history"][0].content)
        self.assertNotIn("私密截图描述", arguments["recent_context"])
        self.assertIn("今天真倒霉", arguments["history"][0].content)
        self.assertIn("私密截图描述", source.content)
