import asyncio
import base64
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import nonebot
from nonebot.adapters.onebot.v11 import Message, MessageSegment

nonebot.init()

from src.plugins.llm_chat import chat, image_generation, tools
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.image_references import ImageReferenceStore, image_reference_store
from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart

PNG = b"\x89PNG\r\n\x1a\nreference"
JPEG = b"\xff\xd8\xffreference"
ENCODED = base64.b64encode(PNG).decode("ascii")


class ImageReferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        image_reference_store.clear()
        tools.clear_image_context()
        tools.clear_voice_context()
        tools._image_last_sent_at.clear()
        image_generation._daily_usage.update(date="", count=0)
        for name, value in {"image_gen_cooldown": 0,
                            "image_gen_describe_output": False}.items():
            patcher = patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(image_reference_store.clear)
        self.addCleanup(tools.clear_image_context)
        self.addCleanup(tools._image_last_sent_at.clear)
        self.bot = SimpleNamespace(send_private_msg=AsyncMock(return_value={"message_id": 123}),
                                   send_group_msg=AsyncMock(return_value={"message_id": 123}))

    def test_cache_enforces_byte_budget_ttl_and_session_isolation(self):
        store = ImageReferenceStore(max_items=2, max_bytes=8, ttl=10)
        with patch("src.plugins.llm_chat.image_references.time.monotonic", return_value=0):
            first = store.remember("group_1", "AAAA", "first")
            other = store.remember("group_2", "AAAA", "same pixels, different group")
            self.assertEqual(list(store.snapshot("group_1")), [first.id])
            store.remember("group_1", "BBBB", "newest")
            self.assertNotIn(first.id, store.snapshot("group_1"))
            self.assertIn(other.id, store.snapshot("group_2"))
            self.assertLessEqual(store._bytes, 8)
        with patch("src.plugins.llm_chat.image_references.time.monotonic", return_value=11):
            self.assertEqual(store.snapshot("group_2"), {})
            self.assertEqual(store._bytes, 0)

    async def test_cliproxyapi_edit_preserves_original_bytes_and_mime(self):
        captured=[]
        def handler(request):
            captured.append(request)
            return httpx.Response(200,json={"data":[{"b64_json":ENCODED}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(config,"cliproxyapi_api_key","test"),patch.object(config,"cliproxyapi_base_url","https://proxy.test/v1"):
                result=await image_generation._generate_openai_compatible_image("edit","1024x1024","cliproxyapi","gpt-image-test",reference_images=(PNG,JPEG),client=client)
        r=captured[0]
        self.assertEqual(str(r.url),"https://proxy.test/v1/images/edits")
        self.assertIn("multipart/form-data; boundary=",r.headers["content-type"])
        self.assertIn(PNG,r.content)
        self.assertIn(JPEG,r.content)
        self.assertIn(b'name="image[]"',r.content)
        self.assertIn(b"Content-Type: image/png",r.content)
        self.assertIn(b"Content-Type: image/jpeg",r.content)
        self.assertEqual(result.image_bytes,PNG)

    async def test_direct_openai_edit_uploads_multipart_originals(self):
        captured = []
        def handler(request):
            captured.append(request)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await image_generation._generate_openai_compatible_image(
                "只改背景", "1024x1024", "openai", "gpt-image-test",
                reference_images=(PNG, JPEG), client=client,
            )
        request = captured[0]
        self.assertTrue(request.url.path.endswith("/images/edits"))
        self.assertIn("multipart/form-data; boundary=", request.headers["content-type"])
        self.assertEqual(request.content.count(b'name="image[]"'), 2)
        self.assertIn(PNG, request.content)
        self.assertIn(JPEG, request.content)
        self.assertIn("只改背景".encode(), request.content)

    async def test_fallback_keeps_references_and_edit_action(self):
        captured = []
        def handler(request):
            captured.append(request)
            return httpx.Response(400, json={"error": "unsupported edit"}) if len(captured) == 1 else httpx.Response(200,json={"data":[{"b64_json":ENCODED}]})
        def candidates(role="image_gen"):
            self.assertEqual(role,"image_gen")
            return ("cliproxyapi:gpt-image-first", ["cliproxyapi:gpt-image-first", "cliproxyapi:gpt-image-second"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(image_generation.image_generation_manager, "_resolve_candidates", side_effect=candidates):
                await image_generation.generate_image("修改背景", reference_images=[ENCODED], client=client)
        self.assertEqual(len(captured), 2)
        for request in captured:
            self.assertTrue(request.url.path.endswith("/images/edits"))
            self.assertIn(PNG,request.content)
            self.assertIn(b'name="image[]"',request.content)

    async def test_knowledge_attachment_flows_through_image_edit_request(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "phos.md").write_text("# 磷叶石\n![角色参考](phos.png)\n", encoding="utf-8")
            (root / "phos.png").write_bytes(PNG)
            tools.set_image_context(None, 42, self.bot)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                async def generate(prompt, **kwargs):
                    return await image_generation.image_generation_manager.generate(prompt, client=client, **kwargs)
                with (
                    patch.object(config, "knowledge_dir", directory),
                    patch.object(image_generation, "generate_image", side_effect=generate),
                    patch.object(image_generation.image_generation_manager, "_resolve_candidates",
                                 return_value=("cliproxyapi:gpt-image-test", ["cliproxyapi:gpt-image-test"])),
                ):
                    found = json.loads(await tools.search_knowledge("磷叶石"))
                    read = json.loads(await tools.read_knowledge(found["items"][0]["document_id"]))
                    reference = read["items"][0]["image_refs"][0]
                    result = json.loads(await tools.generate_image("磷叶石穿雨衣", reference_image_ids=[reference["id"]]))
        self.assertEqual(result["status"], "sent")
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0].url.path.endswith("/images/edits"))
        self.assertIn(PNG, requests[0].content)
        self.assertIn("磷叶石穿雨衣".encode(), requests[0].content)

    async def test_tool_selects_history_original_and_retains_generated_image(self):
        historical = image_reference_store.remember("user_42", ENCODED, "上一轮原图")
        current = image_reference_store.remember("user_42", base64.b64encode(JPEG).decode(), "当前附图")
        tools.set_image_context(None, 42, self.bot, reference_images=[current])
        generated = image_generation.GeneratedImage(PNG + b"edited", "image/png", "test", "1024x1024")
        with patch.object(image_generation, "generate_image", AsyncMock(return_value=generated)) as generate:
            result = json.loads(await tools.generate_image("修改上一张", reference_image_ids=[historical.id]))
        generate.assert_awaited_once_with("修改上一张", orientation="auto", reference_images=[ENCODED])
        tools.clear_image_context()
        tools.set_image_context(None, 42, self.bot)
        retained = tools._image_context.get()["references"][result["reference_id"]]
        self.assertEqual(base64.b64decode(retained.base64), generated.image_bytes)
        self.assertEqual(tools._image_context.get()["current_reference_ids"], [])

    async def test_missing_cross_session_or_invalid_selection_does_not_spend_quota(self):
        other = image_reference_store.remember("group_2", ENCODED, "other group")
        with patch.object(image_generation, "generate_image", AsyncMock()) as generate:
            tools.set_image_context(1, 42, self.bot, missing_reference_images=True)
            self.assertIsInstance(await tools.generate_image("改成红色"), tools.MediaValidationFailure)
            for selection in ([other.id], "not an array", [5]):
                result = await tools.generate_image("改成红色", reference_image_ids=selection)
                self.assertIsInstance(result, tools.MediaValidationFailure)
        generate.assert_not_awaited()
        self.assertEqual(image_generation._daily_usage["count"], 0)
        self.bot.send_group_msg.assert_not_awaited()

    async def test_current_original_is_default_but_explicit_empty_means_new_image(self):
        current = image_reference_store.remember("user_42", ENCODED, "附图")
        generated = image_generation.GeneratedImage(PNG, "image/png", "test", "1024x1024")
        with patch.object(image_generation, "generate_image", AsyncMock(return_value=generated)) as generate:
            tools.set_image_context(None, 42, self.bot, reference_images=[current])
            await tools.generate_image("改成红色")
            generate.assert_awaited_with("改成红色", orientation="auto", reference_images=[ENCODED])
            tools.set_image_context(None, 42, self.bot, reference_images=[current])
            await tools.generate_image("重新画一个苹果", reference_image_ids=[])
            generate.assert_awaited_with("重新画一个苹果", orientation="auto")

    async def test_ambiguous_multiuser_default_requires_explicit_selection(self):
        current = image_reference_store.remember("group_1", ENCODED, "甲附图")
        tools.set_image_context(1, 42, self.bot, reference_images=[current], allow_auto_reference=False)
        result = await tools.generate_image("把背景换掉")
        self.assertIsInstance(result, tools.MediaValidationFailure)
        self.assertIn("多位用户", result)

    async def test_chat_passes_quoted_original_to_tool_context(self):
        event = SimpleNamespace(user_id=42, message_id=123, sender=SimpleNamespace(nickname="甲"))
        event._llm_resolved_incoming_message = chat.ResolvedIncomingMessage(
            plain_text="参考这个图修改", context_image_urls=["https://test/quoted.png"],
        )
        chat._capture_incoming_images(event, [("https://test/quoted.png", ENCODED)])
        async def inspect_context(*args):
            context = tools._image_context.get()
            self.assertEqual(len(context["current_reference_ids"]), 1)
            ref = context["references"][context["current_reference_ids"][0]]
            self.assertEqual(ref.base64, ENCODED)
            self.assertIn("引用/转发", ref.label)
        with patch.object(chat, "_do_handle_single_chat", AsyncMock(side_effect=inspect_context)):
            await chat.handle_single_chat(self.bot, event, "参考这个图修改", [ENCODED], [], PlannerDecision(action="reply"))
        self.assertIsNone(tools._image_context.get())

    async def test_private_reply_edit_keeps_original_through_vision_text_conversion(self):
        class PrivateEvent(SimpleNamespace):
            def get_plaintext(self):
                return self.message.extract_plain_text()
        event = PrivateEvent(
            self_id=999, user_id=42, message_id=77, to_me=True,
            message=Message("参考这个图片，把背景改成夜晚"),
            reply={"sender": {"user_id": 42, "nickname": "甲"}, "message": [
                {"type": "image", "data": {"url": "https://test/original.png"}},
            ]},
        )
        built = AsyncMock(return_value=[])
        saved = AsyncMock()
        generated = image_generation.GeneratedImage(PNG + b"night", "image/png", "test", "1024x1024")
        async def model_chat(*args, **kwargs):
            self.assertIsNone(built.await_args.kwargs["user_images"])
            self.assertIn("白天的海滩", built.await_args.kwargs["user_content"])
            await tools.generate_image("只把背景改成夜晚，保留主体")
            return ""
        with (
            patch.object(chat, "PrivateMessageEvent", PrivateEvent),
            patch.object(chat, "download_images_as_base64", AsyncMock(return_value=[("https://test/original.png", ENCODED)])),
            patch.object(chat.model_manager, "describe_image", AsyncMock(return_value="白天的海滩")),
            patch.object(chat.model_manager, "run_turn", AsyncMock(side_effect=model_chat)),
            patch.object(chat.memory_manager, "get_history", AsyncMock(return_value=[])),
            patch.object(chat.memory_manager, "get_session_model", return_value="test"),
            patch.object(chat.memory_manager, "add_message", saved),
            patch.object(chat, "build_context_messages", built),
            patch.object(chat, "plan_chat_action", AsyncMock(return_value=PlannerDecision(action="reply", parts=[ReplyPart("image")]))),
            patch.object(image_generation, "generate_image", AsyncMock(return_value=generated)) as generate,
        ):
            await chat._process_chat(self.bot, event)
        generate.assert_awaited_once_with("只把背景改成夜晚，保留主体", orientation="auto", reference_images=[ENCODED])
        self.bot.send_private_msg.assert_awaited_once()
        user_memory = next(call.args[2] for call in saved.await_args_list if call.args[1] == "user")
        self.assertIn("白天的海滩", user_memory)
        self.assertIn("[图片引用:img_", user_memory)
        self.assertNotIn(ENCODED, user_memory)
        self.assertIsNone(tools._image_context.get())

    async def test_aggregation_retains_both_owners_originals_and_requires_selection(self):
        pending = []
        for user_id, encoded in [(42, ENCODED), (84, base64.b64encode(JPEG).decode())]:
            event = SimpleNamespace(user_id=user_id, group_id=99, message_id=user_id)
            pending.append(chat.PendingMessage(
                user_id=user_id, sender_name=str(user_id), message="改这张图",
                event=event, images_base64=[encoded],
            ))
        async def inspect_context(*args, **kwargs):
            context = tools._image_context.get()
            self.assertFalse(context["allow_auto_reference"])
            selected = [context["references"][ref_id] for ref_id in context["current_reference_ids"]]
            self.assertEqual([ref.base64 for ref in selected], [pm.images_base64[0] for pm in pending])
            self.assertIn("QQ:42", selected[0].label)
            self.assertIn("QQ:84", selected[1].label)
        with (
            patch.object(chat, "plan_chat_action", AsyncMock(return_value=PlannerDecision(action="reply"))),
            patch.object(chat, "_do_handle_merged_chat", AsyncMock(side_effect=inspect_context)),
        ):
            await chat.handle_merged_chat(self.bot, 99, pending)
        self.assertIsNone(tools._image_context.get())

    async def test_tool_catalog_is_per_context_and_does_not_expose_original_bytes(self):
        ready, proceed = asyncio.Event(), asyncio.Event()
        async def first():
            ref = image_reference_store.remember("group_1", ENCODED, "first")
            tools.set_image_context(1, 42, self.bot, reference_images=[ref])
            ready.set()
            await proceed.wait()
            return tools.get_chat_tools("generate_image")[-1]
        async def second():
            await ready.wait()
            tools.set_image_context(2, 42, self.bot)
            result = tools.get_chat_tools("generate_image")[-1]
            proceed.set()
            return result
        first_tool, second_tool = await asyncio.gather(first(), second())
        self.assertIn("first", first_tool["function"]["description"])
        self.assertNotIn("first", second_tool["function"]["description"])
        self.assertNotIn(ENCODED, json.dumps(first_tool))
