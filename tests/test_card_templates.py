import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from PIL import Image, ImageDraw

nonebot.init()

from src.plugins.llm_chat.card_templates import build_card, render_card, usage_models
from src.plugins.llm_chat.card_renderer import save_card_image
from src.plugins.llm_chat import commands


CONTEXT = dict(group_name="测试群", group_id=7, date_label="2026-10-17")
STATS = {
    "totals": {"calls": 600, "prompt_tokens": 123456, "completion_tokens": 23456},
    "by_spec": [
        {
            "spec": f"provider:model-{i}",
            "calls": i + 1,
            "prompt_tokens": i * 1000,
            "completion_tokens": i * 100,
            "total_tokens": i * 1100,
            "failures": 2 if i == 6 else 0,
        }
        for i in range(7)
    ],
}


class CardTemplateTests(unittest.TestCase):
    def test_briefing_and_news_expose_actual_platform_labels(self):
        label = "CLIProxyAPI/Antigravity:claude-sonnet-4-6"
        briefing = build_card("briefing", {"briefing_model": label}, **CONTEXT)
        self.assertIn(label, briefing.footer)
        self.assertFalse(any(block.heading == "简报模型" for block in briefing.blocks))
        news = build_card("news", {"search_model": label}, **CONTEXT)
        self.assertTrue(any(label in block.entries for block in news.blocks))

    def test_model_breakdown_is_in_rendered_image_and_rest_is_accounted_for(self):
        drawn = []
        original = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            drawn.append(str(text))
            return original(draw, xy, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", capture):
            image = render_card("briefing", {"llm_stats": STATS}, **CONTEXT)
        text = "\n".join(drawn)
        self.assertIn("provider:model-6", text)
        self.assertIn("6,600 Token", text)
        self.assertIn("成功 5 / 失败 2", text)
        self.assertNotIn("其他 2 个模型", text)
        for i in range(7):
            self.assertIn(f"provider:model-{i}", text)
        self.assertIn("1,100 Token", text)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(len(usage_models(STATS)), 7)
        document = build_card("briefing", {"llm_stats": STATS}, **CONTEXT)
        self.assertEqual(len(document.blocks), 4)
        self.assertEqual(document.blocks[-1].heading, "AI 调用统计")
        self.assertIn("provider:model-6", "\n".join(document.blocks[-1].entries))

    def test_memory_review_shows_withdrawal_and_correction(self):
        document = build_card(
            "memory_review",
            {
                "memories": [
                    {"ref": "h:0:important_info:1", "content": "旧偏好", "provenance": {"memory_state": "withdrawn"}},
                    {
                        "ref": "h:0:important_info:2",
                        "content": "转述偏好",
                        "provenance": {"assertion": "reported", "correction": {"reason": "记错人"}},
                    },
                ]
            },
            **CONTEXT,
        )
        self.assertEqual(document.blocks[0].heading, "已撤回")
        self.assertEqual(document.blocks[1].heading, "转述")
        self.assertIn("人工纠正：记错人", document.blocks[1].metadata)

    def test_long_model_name_and_mixed_news_headings_stay_inside_canvas(self):
        original = ImageDraw.ImageDraw.text
        overflows = []

        def capture(draw, xy, text, *args, **kwargs):
            box = draw.textbbox(xy, text, font=kwargs.get("font"))
            if box[2] > 1040:
                overflows.append(text)
            return original(draw, xy, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", capture):
            render_card(
                "news",
                {
                    "items": [
                        {
                            "headline": "复杂中文标题 " + "LongModelIdentifier" * 20,
                            "summary": "这里是较长的新闻正文。" * 40,
                            "source": "来源",
                            "published_at": "2026-10-17",
                        }
                    ]
                },
                **CONTEXT,
            )
        self.assertEqual(overflows, [])

    def test_same_group_concurrent_saves_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Image.new("RGB", (20, 20), "blue")
            with ThreadPoolExecutor(2) as pool:
                paths = list(pool.map(lambda _: save_card_image(image.copy(), Path(tmp), "news", 7), range(2)))
            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                with Image.open(path) as saved:
                    self.assertEqual(saved.size, (20, 20))


class MemoryCardCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_command_reads_only_requester_in_current_group_without_llm(self):
        class GroupEvent(SimpleNamespace):
            pass

        event = GroupEvent(user_id=101, group_id=7, sender=SimpleNamespace(card="甲"))
        with (
            patch.object(commands, "GroupMessageEvent", GroupEvent),
            patch.object(commands.memory_manager, "_get_db", AsyncMock()),
            patch("src.plugins.llm_chat.memory_changes.inspect", return_value={"memories": []}) as inspect,
            patch.object(commands.memory_card_cmd, "finish", AsyncMock()) as finish,
            patch.object(commands.model_manager, "chat", AsyncMock()) as model,
        ):
            await commands.handle_memory_card(event)
        self.assertEqual(inspect.call_args.args[1], "group_7")
        self.assertEqual(inspect.call_args.kwargs["owner_id"], 101)
        finish.assert_awaited_once()
        model.assert_not_awaited()
