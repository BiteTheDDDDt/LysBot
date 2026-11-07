import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import models, tools
from src.plugins.llm_chat.providers import model_supports_tools


class FakeToolMessage:
    def __init__(self, tool_calls, content=None):
        self.tool_calls = tool_calls
        self.content = content
        self.reasoning_details = None

    def model_dump(self, exclude_none=True):
        data = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }
        if exclude_none and data["content"] is None:
            data.pop("content")
        return data


def make_response(message):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


class ProviderToolCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_current_provider_models_are_not_blacklisted(self):
        for spec in (
            "tencentmaas:kimi-k3",
            "volc:glm-5-2-260617",
            "volc:deepseek-v4-pro-260425",
            "nvidia:google/gemma-4-31b-it",
        ):
            with self.subTest(spec=spec):
                self.assertTrue(model_supports_tools(spec))

    async def test_nvidia_keeps_auto_tool_choice_across_tool_rounds(self):
        tool_call = SimpleNamespace(
            id="call-search",
            function=SimpleNamespace(
                name="search_web",
                arguments='{"query":"latest model news"}',
            ),
        )
        create = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([tool_call])),
                make_response(FakeToolMessage([], content="搜索后的回答")),
            ]
        )
        adapter = models.OpenAICompatibleAdapter(
            model_name="nvidia:google/gemma-4-31b-it",
            api_key="test-key",
            base_url="https://integrate.api.nvidia.com/v1",
            model_id="google/gemma-4-31b-it",
        )
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

        with patch.object(
            tools,
            "execute_tool",
            AsyncMock(return_value="联网搜索结果"),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="查一下最新模型新闻"),
                ],
                tools_enabled=True,
            )

        self.assertEqual(str(response), "搜索后的回答")
        self.assertEqual(create.await_count, 2)
        for request in create.await_args_list:
            self.assertIn("tools", request.kwargs)
            self.assertEqual(request.kwargs["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()
