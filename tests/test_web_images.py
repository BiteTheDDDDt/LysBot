import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()

from src.plugins.llm_chat import planner, tools, web_images


class WebImageTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()

    async def test_planned_lookup_delivers_chosen_picture_with_source_and_memory(self):
        decision = planner._normalize_decision({"action": "reply", "parts": [{"type": "web_image"}]}, "reply")
        self.assertEqual(decision.required_tool, "send_web_image")
        selected = {t["function"]["name"] for t in tools.get_chat_tools(decision.required_tool)}
        self.assertTrue({"search_web_images", "send_web_image"} <= selected)
        self.assertNotIn("generate_image", selected)
        image = web_images.WebImage("commons:123", "Ada Lovelace portrait", "https://upload.wikimedia.org/a",
                                    "https://commons.wikimedia.org/?curid=123", "Artist · CC BY 4.0")
        for group_id in (123, None):
            bot = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": 45}),
                                  send_private_msg=AsyncMock(return_value={"message_id": 46}))
            tools.set_image_context(group_id, 99, bot=bot)
            with patch.object(web_images, "search_images", AsyncMock(return_value=[image])), patch.object(
                web_images, "download_preview", AsyncMock(return_value=b"image data")
            ) as download:
                found = json.loads(await tools.execute_tool("search_web_images", {"query": "Ada Lovelace portrait"}))
                candidate = found["items"][0]["id"]
                await tools.execute_tool("send_web_image", {"image_id": candidate})
                await tools.execute_tool("send_web_image", {"image_id": candidate})
                download.assert_awaited_once_with(image)
            send = bot.send_group_msg if group_id else bot.send_private_msg
            send.assert_awaited_once()
            msg = send.await_args.kwargs["message"]
            self.assertEqual([s.type for s in msg], ["image", "text"])
            self.assertIn(image.credit, msg[1].data["text"])
            self.assertIn(image.source, tools.format_sent_image_for_memory(tools.get_last_sent_image()))

    async def test_native_search_is_preferred_and_never_sends_page_as_image(self):
        from src.plugins.llm_chat import web_search
        page = "https://example.org/photo-page"
        asset = "https://cdn.example.org/media?id=123"
        answer = {"items": [{"title": "网页", "url": page}], "search_summary": {
            "content": json.dumps({"images": [{"image_url": asset, "title": "肖像"}]})}}
        with patch("src.plugins.llm_chat.roles.role_available", return_value=True), patch.object(
            web_search, "search", AsyncMock(return_value=answer)
        ) as native, patch.object(web_images, "_search_commons", AsyncMock(return_value=[])) as commons:
            found = await web_images.search_images("某人肖像")
            native.assert_awaited_once_with("某人肖像", find_images=True, limit=15)
            self.assertEqual([image.url for image in found], [asset])
            commons.assert_not_awaited()
            native.return_value = {"items": [{"url": page}]}
            self.assertEqual(await web_images.search_images("另一张"), [])
            commons.assert_awaited_once()
        with self.assertRaises(OSError):
            web_images._verify_image(b"<html>This is not an image</html>")

    async def test_cannot_send_arbitrary_url_or_another_turns_candidate(self):
        tools.set_image_context(None, 99, bot=object())
        with patch.object(web_images, "download_preview", AsyncMock()) as download:
            for invalid in ("", "commons:123", "https://example.com/test.png"):
                result = await tools.execute_tool("send_web_image", {"image_id": invalid})
                self.assertIsInstance(result, tools.MediaInputUnavailable)
            download.assert_not_awaited()
        self.assertFalse(tools.image_was_sent_this_round())
