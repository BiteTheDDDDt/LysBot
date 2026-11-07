import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import image_generation, tools, tts
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.reply_policy import enforce_reply_budget
from src.plugins.llm_chat.response_safety import INVALID_REPLY, is_tool_call_text
from src.plugins.llm_chat.utils import clean_response


class MediaPreflightTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()
        tools.clear_voice_context()

    def test_schema_limits_follow_config_without_mutating_shared_definitions(self):
        original = deepcopy(tools.MEDIA_CHAT_TOOLS)
        for limit in (1000, 123):
            with patch.object(config, "image_gen_max_prompt_chars", limit), patch.object(config, "google_tts_max_chars", limit):
                for name, field in (("generate_image", "prompt"), ("send_voice", "text")):
                    schema = tools.get_chat_tools(name)[-1]["function"]["parameters"]["properties"][field]
                    self.assertEqual(schema["maxLength"], limit)
                    self.assertEqual(schema["minLength"], 1)
                    self.assertIn(f"最多 {limit} 字", schema["description"])
        self.assertEqual(tools.MEDIA_CHAT_TOOLS, original)

    async def test_invalid_image_arguments_do_not_generate_send_or_reserve_quota(self):
        bot = AsyncMock()
        tools.set_image_context(group_id=None, user_id=42, bot=bot)
        usage = deepcopy(image_generation._daily_usage)
        generate = AsyncMock()
        with patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}), patch.object(config, "image_gen_max_prompt_chars", 1000), patch.object(image_generation, "generate_image", generate):
            for arguments in ({}, [], {"prompt": " "}, {"prompt": "画" * 1001}, {"prompt": 42}, {"prompt": "漫画", "orientation": "invalid"}, {"prompt": "漫画", "extra": True}):
                with self.subTest(arguments=str(arguments)[:80]):
                    result = await tools.execute_tool("generate_image", arguments)
                    self.assertIsInstance(result, tools.MediaValidationFailure)
        generate.assert_not_awaited()
        bot.send_private_msg.assert_not_awaited()
        self.assertEqual(image_generation._daily_usage, usage)
        self.assertFalse(tools.image_was_sent_this_round())

    async def test_invalid_voice_arguments_do_not_generate_or_send(self):
        bot = AsyncMock()
        tools.set_voice_context(group_id=None, user_id=42, bot=bot)
        generate = AsyncMock()
        with patch.dict("os.environ", {"MODEL_TTS": "google:test-tts"}), patch.object(config, "google_tts_max_chars", 200), patch.object(tts, "generate_speech", generate):
            for arguments in ({}, {"text": " "}, {"text": "说" * 201}, {"text": None}, {"text": "你好", "emotion": []}):
                with self.subTest(arguments=str(arguments)[:80]):
                    self.assertIsInstance(await tools.execute_tool("send_voice", arguments), tools.MediaValidationFailure)
        generate.assert_not_awaited()
        bot.send_private_msg.assert_not_awaited()


class ToolTextGuardTests(unittest.TestCase):
    def test_tool_envelopes_are_blocked_before_reply_truncation(self):
        call = {"name": "generate_image", "parameters": {"prompt": "群聊漫画"}}
        for payload in (call, [call], {"function": call}, {"tool_calls": [{"function": call}]}, {"name": "functions.send_voice", "arguments": '{"text":"你好"}'}):
            for fenced in (False, True):
                text = json.dumps(payload, ensure_ascii=False)
                if fenced:
                    text = f"```json\n{text}\n```"
                text += "\nWait, I need to use the actual function call format."
                with self.subTest(text=text):
                    self.assertTrue(is_tool_call_text(text))
                    self.assertEqual(clean_response(text), INVALID_REPLY)
                    self.assertEqual(enforce_reply_budget(text), INVALID_REPLY.rstrip("。"))
        self.assertEqual(clean_response('<think>internal</think>```json\n' + json.dumps(call) + '\n```'), INVALID_REPLY)

    def test_incomplete_tool_json_is_also_blocked(self):
        self.assertTrue(is_tool_call_text('```json\n{"name":"generate_image","parameters":{"prompt":"unfinished'))

    def test_ordinary_json_and_explained_examples_are_not_tool_leaks(self):
        for text in ('{"name":"Alice","age":3}', '{"name":"generate_image","description":"工具名称"}', '```json\n{"result":"ok"}\n```', '下面是调用格式示例：\n```json\n{"name":"generate_image","parameters":{}}\n```', '画面是四格群聊漫画。'):
            with self.subTest(text=text):
                self.assertFalse(is_tool_call_text(text))


if __name__ == "__main__":
    unittest.main()
