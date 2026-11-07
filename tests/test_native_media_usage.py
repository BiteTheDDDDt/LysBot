import asyncio
import base64
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import nonebot

nonebot.init()

from src.plugins.llm_chat import image_generation as images, llm_logger, tts
from src.plugins.llm_chat.card_templates import format_usage_lines
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.conversation import session_scope


PNG = b"\x89PNG\r\n\x1a\nimage"


class NativeMediaTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        images._daily_usage.update(date="", count=0)

    async def test_all_image_callers_share_model_availability_and_atomic_daily_budget(self):
        manager = images.ImageGenerationManager()
        generated = images.GeneratedImage(PNG, "image/png", "openai:image", "1024x1024")
        async def generate(*args, **kwargs):
            await asyncio.sleep(0)
            return generated
        with patch.dict(os.environ, {"MODEL_IMAGE_GEN": ""}), \
             patch.object(images, "_generate_openai_compatible_image", AsyncMock()) as provider:
            with self.assertRaisesRegex(images.ImageGenerationError, "未启用"):
                await manager.generate("群像")
            provider.assert_not_awaited()
        with patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}), \
             patch.object(config, "image_gen_daily_limit", 1), \
             patch.object(manager, "_resolve_candidates", return_value=("openai:image", ["openai:image"])), \
             patch.object(images, "_generate_openai_compatible_image", side_effect=generate) as provider:
            results = await asyncio.gather(manager.generate("聊天生图"), manager.generate("群像"),
                                           return_exceptions=True)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(sum(isinstance(result, images.ImageGenerationError) for result in results), 1)
        self.assertEqual(images._daily_usage["count"], 1)

    async def test_direct_image_usage_has_no_driver_call(self):
        payload={"data":[{"b64_json":base64.b64encode(PNG).decode()}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}
        transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload))
        with patch.object(config,"cliproxyapi_api_key","test"),patch.object(config,"cliproxyapi_base_url","https://test.invalid/v1"),patch.object(llm_logger,"finish_call") as finish,session_scope("group_99"):
            async with httpx.AsyncClient(transport=transport) as client:
                await images._generate_openai_compatible_image("test","1024x1024","cliproxyapi","gpt-image-2.6-sunburst",client=client,fallback_attempt=2)
        self.assertEqual(finish.call_count,1)
        self.assertEqual(finish.call_args.args[0].role,"image_gen")
        self.assertEqual(finish.call_args.args[0].session_id,"group_99")
        self.assertEqual(finish.call_args.kwargs["usage"]["total_tokens"],15)
        self.assertEqual(finish.call_args.kwargs["fallback_attempt"],2)

    async def test_empty_image_records_failed_call(self):
        transport=httpx.MockTransport(lambda r:httpx.Response(200,json={"data":[]}))
        with patch.object(config,"cliproxyapi_api_key","test"),patch.object(config,"cliproxyapi_base_url","https://test.invalid/v1"),patch.object(llm_logger,"finish_call") as finish:
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(images.ImageGenerationError):
                    await images._generate_openai_compatible_image("test","1024x1024","cliproxyapi","gpt-image-test",client=client)
        self.assertEqual(finish.call_count,1)
        self.assertIsNotNone(finish.call_args.kwargs["error"])

    async def test_tts_retry_and_partial_usage_are_recorded(self):
        attempts = 0
        def respond(request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"inlineData": {
                    "data": base64.b64encode(b"\0\0" * 20).decode(),
                    "mimeType": "audio/L16;rate=24000",
                }}]}}],
                "usageMetadata": {"candidatesTokenCount": 25},
            })
        with patch.object(config, "google_api_key", "test"), \
             patch.object(config, "network_max_retries", 1), \
             patch.object(config, "network_retry_delay", 0), \
             patch.object(llm_logger, "finish_call") as finish:
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                await tts._generate_google_speech("你好", "neutral", "test-tts", client=client)
        self.assertEqual(finish.call_count, 2)
        self.assertIsNotNone(finish.call_args_list[0].kwargs["error"])
        self.assertIsNone(finish.call_args_list[1].kwargs["error"])
        self.assertEqual(finish.call_args_list[1].kwargs["usage"], {"completion_tokens": 25})

    async def test_cancelled_native_call_records_failure(self):
        with patch.object(llm_logger, "finish_call") as finish:
            with self.assertRaises(asyncio.CancelledError):
                with llm_logger.NativeCall(role="tts", model="google:test", caller="test") as call:
                    call.start()
                    raise asyncio.CancelledError
        self.assertIsInstance(finish.call_args.kwargs["error"], asyncio.CancelledError)

    async def test_cancel_during_retry_backoff_does_not_invent_a_request(self):
        provider = AsyncMock(return_value=httpx.Response(429, text="retry"))
        for module in (tts, images):
            with self.subTest(module=module.__name__), patch.object(llm_logger, "finish_call") as finish, patch.object(config, "network_max_retries", 1):
                with self.assertRaises(asyncio.CancelledError):
                    with llm_logger.NativeCall(role="tts", model="google:test", caller="test") as call:
                        with patch.object(module.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)):
                            await module._post_with_retries(
                                type("Client", (), {"post": provider})(), "https://test.invalid",
                                headers={}, body={}, call=call)
                self.assertEqual(finish.call_count, 1)
                self.assertIsNotNone(finish.call_args.kwargs["error"])

    def test_unknown_usage_is_visible_in_model_rows(self):
        row = {"spec": "openai:image", "calls": 2, "failures": 1, "unknown_usage_calls": 2}
        lines = format_usage_lines({"totals": row, "by_spec": [row]})
        self.assertIn("成功 1 / 失败 1", lines[-1])
        self.assertIn("2 次用量未完整返回", lines[-1])
        self.assertIn("Token 未知", lines[-1])
        self.assertIsNone(llm_logger.native_usage({}))
