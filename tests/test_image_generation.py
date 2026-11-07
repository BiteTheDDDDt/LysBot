import base64
import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import nonebot


nonebot.init()

from src.plugins.llm_chat import chat, image_generation, models, tools
from src.plugins.llm_chat.config import TOOL_USAGE_PROMPT, config
from src.plugins.llm_chat.providers import supports_image_generation
from src.plugins.llm_chat.roles import resolve_role


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test-image"


class CLIProxyAPIImageGenerationTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()
        tools.clear_voice_context()
        tools._image_last_sent_at.clear()
        image_generation._daily_usage.update({"date": "", "count": 0})

    async def test_remote_image_download_connects_to_checked_address(self):
        from src.plugins.llm_chat import public_net
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, content=PNG_BYTES)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(public_net, "resolve_public_address", AsyncMock(return_value="93.184.216.34")):
                result = await image_generation._download_remote_image("https://images.example/a.png", client)
        self.assertEqual(result, PNG_BYTES)
        self.assertEqual(requests[0].url.host, "93.184.216.34")
        self.assertEqual(requests[0].headers["host"], "images.example")
        self.assertEqual(requests[0].extensions["sni_hostname"], "images.example")

    async def test_cancelled_media_followup_records_delivery_and_propagates(self):
        for kind in ("image", "voice"):
            with self.subTest(kind=kind):
                receipt = {"prompt": "宝石", "text": "你好", "model": "test", "size": "1024x1024"}
                with patch.object(tools, f"get_last_sent_{kind}", return_value=receipt), \
                     patch.object(chat.memory_manager, "add_message", AsyncMock()) as save, \
                     patch.object(chat.proactive_manager, "mark_just_spoke"):
                    with self.assertRaises(asyncio.CancelledError):
                        await getattr(chat, f"_finalize_{kind}_tool_turn")(
                            42, 99, followup_text="未发出的附言",
                            send_followup=AsyncMock(side_effect=asyncio.CancelledError),
                        )
                save.assert_awaited_once()
                self.assertNotIn("未发出的附言", save.await_args.args[2])
                self.assertIn("宝石" if kind == "image" else "你好", save.await_args.args[2])

    async def test_generate_image_uses_cliproxyapi_images_endpoint(self):
        captured={}
        def handler(request):
            captured.update(url=str(request.url),body=json.loads(request.content))
            return httpx.Response(200,json={"data":[{"b64_json":base64.b64encode(PNG_BYTES).decode()}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(config,"cliproxyapi_api_key","test"), patch.object(config,"cliproxyapi_base_url","https://proxy.test/v1"), patch.dict(os.environ,{"MODEL_IMAGE_GEN":"cliproxyapi:gpt-image-2.6-sunburst","MODEL_GENERAL":"openai:text-only"}):
                result=await image_generation.generate_image("draw a circle",orientation="landscape",client=client)
        self.assertEqual(captured["url"],"https://proxy.test/v1/images/generations")
        self.assertEqual(captured["body"],{"model":"gpt-image-2.6-sunburst","prompt":"draw a circle","size":"1536x1024","n":1,"output_format":"png"})
        self.assertEqual(result.image_bytes,PNG_BYTES)
        self.assertEqual(result.driver_model_spec,"")

    async def test_role_image_gen_falls_back_only_to_explicit_image_chain(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            image_model = body["model"]
            calls.append(image_model)
            if image_model == "gpt-image-primary":
                return httpx.Response(500, json={"error": {"message": "temporary"}})
            encoded = base64.b64encode(PNG_BYTES).decode("ascii")
            return httpx.Response(200,json={"data":[{"b64_json":encoded}]})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with (
                patch.object(config, "cliproxyapi_api_key", "local-test-key"),
                patch.object(config, "cliproxyapi_base_url", "https://proxy.test/v1"),
                patch.dict(
                    os.environ,
                    {
                        "MODEL_IMAGE_GEN": (
                            "cliproxyapi:gpt-image-primary | cliproxyapi:gpt-image-2"
                        ),
                        "MODEL_FALLBACK": "nvidia:not-an-image-model",
                    },
                ),
            ):
                result = await image_generation.generate_image("fallback test", client=client)
                primary, fallback = resolve_role("image_gen")

        self.assertEqual(primary, "cliproxyapi:gpt-image-primary")
        self.assertEqual(fallback, ["cliproxyapi:gpt-image-2"])
        self.assertEqual(calls, ["gpt-image-primary", "gpt-image-2"])
        self.assertEqual(result.model_spec, "cliproxyapi:gpt-image-2")

    async def test_images_ignore_general_and_global_text_candidates(self):
        calls=[]
        def handler(request):
            calls.append(json.loads(request.content)["model"])
            return httpx.Response(200,json={"data":[{"b64_json":base64.b64encode(PNG_BYTES).decode()}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ,{"MODEL_IMAGE_GEN":"cliproxyapi:gpt-image-test","MODEL_GENERAL":"cliproxyapi:driver","MODEL_FALLBACK":"cliproxyapi:text-backup"}):
                result=await image_generation.generate_image("test",client=client)
        self.assertEqual(calls,["gpt-image-test"])
        self.assertEqual(result.driver_model_spec,"")

    async def test_images_work_without_text_candidates(self):
        transport=httpx.MockTransport(lambda r:httpx.Response(200,json={"data":[{"b64_json":base64.b64encode(PNG_BYTES).decode()}]}))
        async with httpx.AsyncClient(transport=transport) as client:
            with patch.dict(os.environ,{"MODEL_IMAGE_GEN":"cliproxyapi:gpt-image-test","MODEL_GENERAL":"","MODEL_FALLBACK":""}):
                result=await image_generation.generate_image("test",client=client)
        self.assertEqual(result.model_spec,"cliproxyapi:gpt-image-test")

    async def test_failed_image_never_tries_text_chain(self):
        calls=[]
        def handler(request):
            calls.append(json.loads(request.content)["model"])
            return httpx.Response(400,json={"error":{"message":"unsupported"}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.dict(os.environ,{"MODEL_IMAGE_GEN":"cliproxyapi:gpt-image-test","MODEL_GENERAL":"cliproxyapi:driver","MODEL_FALLBACK":"cliproxyapi:text-backup"}):
                with self.assertRaises(image_generation.ImageGenerationError):
                    await image_generation.generate_image("test",client=client)
        self.assertEqual(calls,["gpt-image-test"])

    async def test_direct_openai_images_do_not_require_responses_driver(self):
        captured = {}

        def handler(request):
            captured['url'] = str(request.url)
            captured['body'] = json.loads(request.content)
            return httpx.Response(200, json={'data': [{'b64_json': base64.b64encode(PNG_BYTES).decode('ascii')}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with (
                patch.dict(os.environ, {'MODEL_IMAGE_GEN': 'openai:gpt-image-test', 'MODEL_GENERAL': 'openai:text-only'}),
                patch.object(config, 'openai_api_key', 'test-only'),
                patch.object(config, 'openai_base_url', 'https://image.test/v1'),
            ):
                result = await image_generation.generate_image('direct image', client=client)
        self.assertEqual(captured['url'], 'https://image.test/v1/images/generations')
        self.assertEqual(captured['body']['model'], 'gpt-image-test')
        self.assertEqual(result.driver_model_spec, '')

    async def test_direct_image_backup_works_without_compatible_general_driver(self):
        captured = []

        def handler(request):
            captured.append((str(request.url), json.loads(request.content)['model']))
            if json.loads(request.content)['model']=='gpt-image-first':
                return httpx.Response(400,json={'error':{'message':'unsupported'}})
            return httpx.Response(200, json={'data': [{'b64_json': base64.b64encode(PNG_BYTES).decode('ascii')}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with (
                patch.dict(os.environ, {
                    'MODEL_IMAGE_GEN': 'cliproxyapi:gpt-image-first | openai:gpt-image-backup',
                    'MODEL_GENERAL': 'openai:text-only', 'MODEL_FALLBACK': 'google:text-only',
                }),
                patch.object(config, 'openai_api_key', 'test-only'),
                patch.object(config, 'openai_base_url', 'https://image.test/v1'),
            ):
                result = await image_generation.generate_image('direct image fallback', client=client)
        self.assertEqual([m for _,m in captured], ['gpt-image-first','gpt-image-backup'])
        self.assertEqual(result.model_spec, 'openai:gpt-image-backup')

    async def test_empty_direct_image_response_fails(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={"data":[]}))) as client:
            with patch.dict(os.environ,{"MODEL_IMAGE_GEN":"cliproxyapi:gpt-image-test"}):
                with self.assertRaisesRegex(image_generation.ImageGenerationError,"没有 b64_json"):
                    await image_generation.generate_image("empty",client=client)

    async def test_image_requests_wait_for_the_concurrency_semaphore(self):
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        first_request_entered = asyncio.Event()
        release_first_request = asyncio.Event()
        request_count = 0
        active_requests = 0
        max_active_requests = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count, active_requests, max_active_requests
            request_count += 1
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            if request_count == 1:
                first_request_entered.set()
                await release_first_request.wait()
            active_requests -= 1
            return httpx.Response(200,json={"data":[{"b64_json":encoded}]})

        transport = httpx.MockTransport(handler)
        semaphore = asyncio.Semaphore(1)
        async with httpx.AsyncClient(transport=transport) as client:
            with (
                patch.object(config, "cliproxyapi_api_key", "local-test-key"),
                patch.object(config, "cliproxyapi_base_url", "https://proxy.test/v1"),
                patch.object(image_generation, "_generation_semaphore", semaphore),
                patch.dict(os.environ, {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-2"}),
            ):
                first = asyncio.create_task(
                    image_generation.generate_image("第一张图", client=client)
                )
                await first_request_entered.wait()
                second = asyncio.create_task(
                    image_generation.generate_image("第二张图", client=client)
                )
                await asyncio.sleep(0)

                self.assertEqual(request_count, 1)
                self.assertFalse(second.done())
                release_first_request.set()
                first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(request_count, 2)
        self.assertEqual(max_active_requests, 1)
        self.assertEqual(first_result.image_bytes, PNG_BYTES)
        self.assertEqual(second_result.image_bytes, PNG_BYTES)

    def test_image_generation_capability_is_separate_from_vision(self):
        self.assertTrue(supports_image_generation("cliproxyapi:gpt-image-2"))
        self.assertTrue(supports_image_generation("openai:gpt-image-2"))
        self.assertFalse(supports_image_generation("cliproxyapi:gpt-5.6-sol"))
        self.assertFalse(supports_image_generation("google:gemini-3.1-flash-image"))

    async def test_generate_image_tool_sends_private_qq_image(self):
        bot = SimpleNamespace(
            send_group_msg=AsyncMock(),
            send_private_msg=AsyncMock(return_value={"message_id": 789}),
        )
        tools.set_image_context(group_id=None, user_id=42, bot=bot)
        generated = image_generation.GeneratedImage(
            image_bytes=PNG_BYTES,
            mime_type="image/png",
            model_spec="cliproxyapi:gpt-image-2",
            size="1024x1024",
        )

        with (
            patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}),
            patch.object(config, "image_gen_max_prompt_chars", 1000),
            patch.object(config, "image_gen_cooldown", 0),
            patch.object(config, "image_gen_daily_limit", 20),
            patch.object(config, "image_gen_describe_output", True),
            patch.object(
                image_generation,
                "generate_image",
                AsyncMock(return_value=generated),
            ) as generate_mock,
            patch.object(
                models.model_manager,
                "describe_image",
                AsyncMock(return_value="薄荷绿色晶体短发的磷叶石正在对镜自拍。"),
            ) as describe_mock,
        ):
            result = await tools.generate_image("一只绿色宝石小鸟", orientation="square")

        tool_result = json.loads(result)
        self.assertEqual(tool_result["status"], "sent")
        self.assertEqual(tool_result["media_type"], "image")
        self.assertEqual(tool_result["actual_description"], "薄荷绿色晶体短发的磷叶石正在对镜自拍。")
        generate_mock.assert_awaited_once_with("一只绿色宝石小鸟", orientation="square")
        bot.send_private_msg.assert_awaited_once()
        sent_message = bot.send_private_msg.await_args.kwargs["message"]
        self.assertEqual(sent_message[0].type, "image")
        self.assertTrue(tools.image_was_sent_this_round())
        describe_mock.assert_awaited_once()
        memory_text = tools.format_sent_image_for_memory(tools.get_last_sent_image())
        self.assertIn("一只绿色宝石小鸟", memory_text)
        self.assertIn("cliproxyapi:gpt-image-2", memory_text)
        self.assertIn("实际画面：薄荷绿色晶体短发的磷叶石正在对镜自拍。", memory_text)
        self.assertIn("生成依据：一只绿色宝石小鸟", memory_text)

    async def test_image_tool_turn_sends_and_records_tool_aware_followup(self):
        tools.set_image_context(group_id=99, user_id=42, bot=object())
        context = tools._image_context.get()
        context["sent_in_round"] = True
        context["last_sent"] = {
            "prompt": "写进聊天历史的图片",
            "description": "画面里是一块发光的绿色宝石。",
            "model": "cliproxyapi:gpt-image-2",
            "size": "1024x1024",
        }
        tools._image_context.set(context)

        add_message = AsyncMock()
        send_followup = AsyncMock()
        with (
            patch.object(chat.memory_manager, "add_message", add_message),
            patch.object(chat.proactive_manager, "mark_just_spoke") as mark_spoke,
        ):
            finalized = await chat._finalize_image_tool_turn(
                user_id=42,
                group_id=99,
                followup_text="喏，这块宝石还挺亮吧。",
                send_followup=send_followup,
            )

        self.assertTrue(finalized)
        add_message.assert_awaited_once()
        self.assertIn("写进聊天历史的图片", add_message.await_args.args[2])
        self.assertIn("画面里是一块发光的绿色宝石", add_message.await_args.args[2])
        self.assertIn("喏，这块宝石还挺亮吧", add_message.await_args.args[2])
        send_followup.assert_awaited_once_with("喏，这块宝石还挺亮吧")
        mark_spoke.assert_called_once_with(99)

    def test_generate_image_tool_is_exposed_to_chat_models(self):
        names = {item["function"]["name"] for item in tools.AVAILABLE_TOOLS}
        self.assertIn("generate_image", names)

    def test_image_tool_instructions_cover_implicit_visual_requests(self):
        description = tools.IMAGE_GENERATION_TOOL["function"]["description"]
        prompt_description = tools.IMAGE_GENERATION_TOOL["function"]["parameters"][
            "properties"
        ]["prompt"]["description"]

        self.assertIn("宝宝我要看你的自拍", description)
        self.assertIn("不要回答自己没有自拍或照片", description)
        self.assertIn("仅讨论图片", description)
        self.assertIn("当前人设", prompt_description)
        from src.plugins.llm_chat.config import PERSONA
        self.assertIn("薄荷绿色晶体头发", PERSONA.appearance)
        self.assertIn("宝宝我要看你的自拍", TOOL_USAGE_PROMPT)
        self.assertIn("不要求用户必须说出“生成图片”", TOOL_USAGE_PROMPT)


if __name__ == "__main__":
    unittest.main()
