import unittest
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import planner
from src.plugins.llm_chat import llm_logger
from src.plugins.llm_chat.config import get_tool_usage_prompt
from src.plugins.llm_chat.tools import get_chat_tools


class PlannerMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_reference_edit_is_not_downgraded_to_text(self):
        with (
            patch.object(planner.memory_manager, "get_recent_session_messages", AsyncMock(return_value=[])),
            patch.object(planner.model_manager, "chat", AsyncMock(return_value='{"action":"reply","parts":[{"type":"text"}]}')),
        ):
            for message in [
                "参考这个图片，生成一个修改背景的版本", "把这张图的衣服换成红色",
                "[引用回复]\n原消息：[图片]\n\n[当前消息]\n参考这个图片，生成一个修改背景的版本",
                "【群员-甲】 参考这张图，生成一个修改背景的版本",
            ]:
                decision = await planner.plan_chat_action(
                    session_id="user_42", group_id=0, current_message=message, is_direct=True, has_images=True,
                )
                self.assertEqual(decision.required_tool, "generate_image")

    def test_reference_discussion_and_negated_edits_do_not_force_generation(self):
        for message in ["怎么参考这张图修改背景？", "不要把这张图改成红色", "参考这张图，但不要修改背景", "评价一下这张图片"]:
            self.assertEqual(planner._explicit_required_tool(message), "")

    def test_quoted_media_requests_are_data_not_current_instructions(self):
        for message in [
            "[引用回复]\n原消息：生成一张图片\n\n[当前消息]\n这句话什么意思？",
            "[合并转发]\n甲: 生成一张图片",
        ]:
            self.assertEqual(planner._explicit_required_tool(message), "")

    def test_planner_log_preview_keeps_recent_context_visible(self):
        prompt = (
            "规划规则" * 1000
            + "\n最近对话（不含下面单列的当前消息）:\n"
            + "【群员-甲】 上一张图是麦当劳薯条海报\n"
            + "当前消息:\n把这个薯条数量翻十倍"
        )

        preview = llm_logger._planner_prompt_preview(prompt)

        self.assertIn("保留最近对话", preview)
        self.assertIn("最近对话（不含下面单列的当前消息）", preview)
        self.assertIn("上一张图是麦当劳薯条海报", preview)
        self.assertIn("把这个薯条数量翻十倍", preview)

    def test_planner_history_includes_bot_media_and_excludes_current_message(self):
        history = [
            planner.LLMMessage(role="user", content="【群员-甲】 我喜欢蓝色", name="甲"),
            planner.LLMMessage(
                role="assistant",
                content="【生成图片 model=test】实际画面：蓝色汉服自拍",
            ),
            planner.LLMMessage(role="assistant", content="内部判断", is_thinking=True),
            planner.LLMMessage(role="user", content="【群员-甲】 按刚才那个来", name="甲"),
        ]

        context = planner._format_planner_history(
            history,
            current_message="按刚才那个来",
            max_chars=6000,
        )

        self.assertIn("我喜欢蓝色", context)
        self.assertIn("蓝色汉服自拍", context)
        self.assertNotIn("内部判断", context)
        self.assertNotIn("按刚才那个来", context)

    async def test_planner_prompt_loads_recent_session_history(self):
        recent = [
            planner.LLMMessage(role="user", content="我想要安静一点的语气", name="用户"),
            planner.LLMMessage(role="assistant", content="Bot: 好，我记下了。"),
        ]
        planner_call = AsyncMock(
            return_value=(
                '{"action":"reply","reason":"延续上文语气",'
                '"parts":[{"type":"voice"}]}'
            )
        )
        with (
            patch.object(
                planner.memory_manager,
                "get_recent_session_messages",
                AsyncMock(return_value=recent),
            ) as get_recent,
            patch.object(planner.model_manager, "chat", planner_call),
        ):
            await planner.plan_chat_action(
                session_id="private_42",
                group_id=0,
                current_message="那就念给我听",
                is_direct=True,
            )

        get_recent.assert_awaited_once_with(
            "private_42",
            limit=planner.config.planner_context_messages,
        )
        prompt = planner_call.await_args.args[0][0].content
        self.assertIn("我想要安静一点的语气", prompt)
        self.assertIn("好，我记下了", prompt)
        self.assertIn("当前消息:\n那就念给我听", prompt)
        self.assertIn("action=silent", prompt)
        self.assertIn("不按三轮机械收尾", prompt)
        self.assertIn('"parts"', prompt)
        self.assertIn("表情包是 reply 的一种可见输出", prompt)
        self.assertNotIn('"response_shape"', prompt)

    def test_close_action_produces_one_terminal_text_reply(self):
        decision = planner._normalize_decision(
            {
                "action": "close",
                "topic_state": "closing",
                "reply_hint": "自然结束斗嘴",
                "parts": [{"type": "text"}, {"type": "text"}],
            },
            "reply",
        )

        self.assertTrue(decision.should_continue)
        self.assertEqual(decision.action, "close")
        self.assertEqual(decision.topic_state, "closing")
        self.assertEqual([part.type for part in decision.parts], ["text"])
        self.assertIn("不提出新问题", decision.system_hint())

    def test_exhausted_topic_suppresses_a_contradictory_reply(self):
        decision = planner._normalize_decision(
            {
                "action": "reply",
                "topic_state": "exhausted",
                "reason": "已经连续斗嘴很多轮，没有新信息",
            },
            "reply",
        )

        self.assertEqual(decision.action, "silent")
        self.assertEqual(decision.topic_state, "exhausted")
        self.assertFalse(decision.should_continue)

    def test_new_media_request_reopens_an_exhausted_topic(self):
        decision = planner._normalize_decision(
            {
                "action": "silent",
                "topic_state": "exhausted",
                "parts": [{"type": "image"}],
            },
            "reply",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.topic_state, "active")
        self.assertEqual(decision.required_tool, "generate_image")

    def test_media_tools_are_only_exposed_when_planner_requires_them(self):
        def names(required_tool=None):
            return {
                item["function"]["name"]
                for item in get_chat_tools(required_tool)
            }

        normal = names()
        image = names("generate_image")
        voice = names("send_voice")

        self.assertNotIn("generate_image", normal)
        self.assertNotIn("send_voice", normal)
        self.assertIn("generate_image", image)
        self.assertNotIn("send_voice", image)
        self.assertIn("send_voice", voice)
        self.assertNotIn("generate_image", voice)
        self.assertTrue({"search_web", "send_poke", "recall_history"} <= normal)

    def test_tool_prompt_matches_selected_media_tool(self):
        normal = get_tool_usage_prompt()
        image = get_tool_usage_prompt("generate_image")
        voice = get_tool_usage_prompt("send_voice")
        image_only = get_tool_usage_prompt(
            "generate_image",
            media_followup=False,
        )

        self.assertNotIn("`generate_image`", normal)
        self.assertNotIn("`send_voice`", normal)
        self.assertIn("`generate_image`", image)
        self.assertNotIn("`send_voice`", image)
        self.assertIn("`send_voice`", voice)
        self.assertNotIn("`generate_image`", voice)
        self.assertIn("媒体后面没有 text", image_only)
        self.assertNotIn("media_only", image_only)

    def test_recall_is_not_a_planner_action_and_media_still_runs(self):
        decision = planner._normalize_decision(
            {
                "action": "recall",
                "parts": [{"type": "image"}],
            },
            "reply",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.required_tool, "generate_image")
        self.assertEqual([part.type for part in decision.parts], ["image"])
        self.assertFalse(decision.media_followup)
        self.assertNotIn("recall", planner.PLANNER_ACTIONS)

    def test_media_can_be_followed_by_an_explicit_text_part(self):
        decision = planner._normalize_decision(
            {
                "action": "reply",
                "parts": [{"type": "image"}, {"type": "text"}],
            },
            "reply",
        )

        self.assertEqual([part.type for part in decision.parts], ["image", "text"])
        self.assertTrue(decision.media_followup)

    def test_meme_is_an_ordered_reply_part_not_an_action(self):
        decision = planner._normalize_decision(
            {
                "action": "reply",
                "reason": "适合先接一句再用表情包收尾",
                "parts": [
                    {"type": "text"},
                    {"type": "meme", "topic": "funny"},
                ],
            },
            "reply",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(
            [(part.type, part.topic) for part in decision.parts],
            [("text", ""), ("meme", "funny")],
        )
        self.assertEqual(decision.meme_topic, "funny")
        self.assertNotIn("meme", planner.PLANNER_ACTIONS)

    def test_legacy_meme_action_is_converted_to_reply_parts(self):
        decision = planner._normalize_decision(
            {
                "action": "meme",
                "meme_topic": "shock",
                "response_shape": "meme_then_text",
            },
            "reply",
        )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(
            [(part.type, part.topic) for part in decision.parts],
            [("meme", "shock"), ("text", "")],
        )

    def test_explicit_media_fallback_avoids_capability_questions(self):
        self.assertEqual(
            planner._explicit_required_tool("你生成个图片"),
            "generate_image",
        )
        self.assertEqual(
            planner._explicit_required_tool("给我来段语音"),
            "send_voice",
        )
        self.assertEqual(planner._explicit_required_tool("你会生成图片吗"), "")
        self.assertEqual(planner._explicit_required_tool("不要生成图片"), "")
        self.assertEqual(planner._explicit_required_tool("发个自拍"), "generate_image")
        self.assertEqual(planner._explicit_required_tool("他发了个自拍"), "")

    async def test_planner_output_exposes_required_tool(self):
        with (
            patch.object(
                planner.memory_manager,
                "get_recent_session_messages",
                AsyncMock(return_value=[]),
            ),
            patch.object(
                planner.model_manager,
                "chat",
                AsyncMock(
                    return_value=(
                        '{"action":"reply","reason":"用户想听声音",'
                        '"parts":[{"type":"voice"}]}'
                    )
                ),
            ),
        ):
            decision = await planner.plan_chat_action(
                session_id="private_42",
                group_id=0,
                current_message="念给我听",
                is_direct=True,
            )

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.required_tool, "send_voice")
        self.assertEqual([part.type for part in decision.parts], ["voice"])
        self.assertFalse(decision.media_followup)


if __name__ == "__main__":
    unittest.main()
