import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.bootstrap import prepare_environment
from src.plugins.llm_chat import models, roles, tools
from src.plugins.llm_chat.config import Config, config


class MinimumConfigTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"MODEL_GENERAL": "openai:test-general"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_only_general_is_required(self):
        with patch.object(config, "enable_daily_news_broadcast", False):
            self.assertEqual(roles.validate_all(), [])
        for role in ("chat", "planner", "summary", "important_info", "briefing"):
            self.assertEqual(roles.resolve_role(role), ("openai:test-general", []))
        for role in ("vision", "search", "tts", "image_gen"):
            self.assertFalse(roles.role_available(role))

    async def test_missing_optional_models_hide_and_reject_tools(self):
        for required in (None, "send_voice", "generate_image"):
            names = {tool["function"]["name"] for tool in tools.get_chat_tools(required)}
            self.assertFalse(names & {"search_web", "send_voice", "generate_image"})
        for name in ("search_web", "send_voice", "generate_image"):
            result = json.loads(await tools.execute_tool(name, {}))
            self.assertEqual(result["error"]["code"], "capability_unconfigured")

    async def test_missing_vision_preserves_image_without_model_call(self):
        result = await models.model_manager.analyze_image_for_meme("unused-image")
        self.assertFalse(result["is_meme"])
        self.assertIn("未配置", result["description"])
        result = await models.model_manager.chat_with_images([], ["unused-image"])
        self.assertIn("未配置", result)

    def test_embedding_identity_is_fixed(self):
        from src.plugins.llm_chat.memory_index import embedding_settings

        self.assertFalse({"memory_embedding_model", "memory_embedding_dimensions",
                          "memory_embedding_revision"} & Config.model_fields.keys())
        value = embedding_settings()
        self.assertEqual(value["model"], "local:bge-small-zh-v1.5")
        self.assertEqual(value["dimension"], 512)
        self.assertEqual(value["revision"], "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59")

    def test_news_dependency_is_explicit(self):
        with patch.object(config, "enable_daily_news_broadcast", True), \
                patch.object(config, "enable_daily_briefing", False):
            errors = roles.validate_all()
        self.assertTrue(any("ENABLE_DAILY_BRIEFING" in error for error in errors))
        self.assertTrue(any("MODEL_SEARCH" in error for error in errors))

    def test_master_replaces_legacy_administrators(self):
        os.environ.update(MASTER_QQ="42", SUPERUSERS='["99", "100"]')
        prepare_environment()
        self.assertEqual(json.loads(os.environ["SUPERUSERS"]), ["42"])

    async def test_group_card_switch_defaults_to_no_api_calls(self):
        from src.plugins.llm_chat.model_identity import update_group_card_for_response
        bot = SimpleNamespace(get_group_member_info=AsyncMock(), set_group_card=AsyncMock())
        with patch.object(config, "enable_model_card", False):
            await update_group_card_for_response(bot, 7, SimpleNamespace(model_spec="openai:test"))
        bot.get_group_member_info.assert_not_awaited()
        bot.set_group_card.assert_not_awaited()
