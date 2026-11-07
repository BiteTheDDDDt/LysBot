import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import nonebot


nonebot.init()

from src.plugins.llm_chat import models
from src.plugins.llm_chat.config import config


class TencentMaaSProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_kimi_k3_uses_its_fixed_parameter_template(self):
        message = SimpleNamespace(
            content="ok",
            reasoning_details=None,
            tool_calls=[],
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )
        create = AsyncMock(return_value=response)
        adapter = models.OpenAICompatibleAdapter(
            model_name="tencentmaas:kimi-k3",
            api_key="test-key",
            base_url="https://tokenhub.tencentmaas.com/v1",
            model_id="kimi-k3",
        )
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

        result = await adapter.chat(
            [
                models.Message(role="system", content="system"),
                models.Message(role="user", content="hello"),
            ],
        )

        self.assertEqual(str(result), "ok")
        request = create.await_args.kwargs
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)
        self.assertNotIn("frequency_penalty", request)
        self.assertNotIn("presence_penalty", request)
        self.assertNotIn("max_tokens", request)
        self.assertEqual(
            request["max_completion_tokens"],
            config.model_max_tokens,
        )

    def test_global_top_p_matches_fixed_model_default(self):
        self.assertEqual(config.model_top_p, 0.95)
