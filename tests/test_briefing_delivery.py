import tempfile
import asyncio
import base64
import json
import sqlite3
from contextlib import closing
from io import BytesIO
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from PIL import Image, ImageDraw


nonebot.init()

from src.plugins.llm_chat import briefing


class BriefingAttemptTests(unittest.TestCase):
    def test_attempt_budget_survives_reopening_and_expires_old_records(self):
        now = datetime(2026, 10, 21)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.db"
            claim = lambda when: briefing._claim_briefing_attempt(path, 123, when.date(), when)
            self.assertEqual(claim(now), 1)
            self.assertIsNone(claim(now + timedelta(minutes=10)))
            self.assertEqual(claim(now + timedelta(minutes=30)), 2)
            self.assertIsNone(claim(now + timedelta(minutes=60)))
            self.assertEqual(claim(now + timedelta(minutes=90)), 3)
            self.assertIsNone(claim(now + timedelta(hours=11)))
            self.assertEqual(claim(now + timedelta(days=1)), 1)
            self.assertEqual(claim(now + timedelta(days=32)), 1)
            with closing(sqlite3.connect(path)) as db:
                keys = [row[0] for row in db.execute("SELECT key FROM kv_meta")]
            self.assertEqual(keys, ["briefing_attempts:123:2026-11-22"])


class BriefingRenderTests(unittest.TestCase):
    def test_portrait_budget_counts_serialization_and_preserves_scene(self):
        import json
        scene = '群友围着“2.0”音箱争论，另一人举起写有"-0.41%"的手机。'
        card = {
            "portrait_prompt": scene,
            "topics": ['音箱与电视的连接方式，讨论"5.1"布局。' * 30] * 5,
            "active_members": ["群友举起遥控器，另一人移动音箱。" * 30] * 5,
        }
        with patch.object(briefing.config, "image_gen_max_prompt_chars", 1000):
            prompt = briefing.build_portrait_prompt(card)
        self.assertLessEqual(len(prompt), 1000)
        material = json.loads(prompt.split("\n", 1)[1])
        self.assertEqual(material["场景草案"], scene)
        self.assertIn("话题1", material)
        self.assertEqual(card["portrait_prompt"], scene)

    def test_verbose_scene_keeps_complete_clauses_and_topic_anchors(self):
        import json
        card = {"portrait_prompt": "群友举起遥控器，音箱发出蓝色声波。" * 100,
                "topics": ["音箱布局与音频直通"], "active_members": ["一人拉窗帘"]}
        with patch.object(briefing.config, "image_gen_max_prompt_chars", 700):
            prompt = briefing.build_portrait_prompt(card)
        material = json.loads(prompt.split("\n", 1)[1])
        self.assertLessEqual(len(prompt), 700)
        self.assertTrue(material["场景草案"].endswith(("，", "。")))
        self.assertEqual(material["话题1"], card["topics"][0])
        self.assertEqual(material["人物行动1"], card["active_members"][0])

    def test_member_grid_keeps_order_and_removes_closing_section(self):
        atlas = Image.new("RGB", (200, 200))
        draw = ImageDraw.Draw(atlas)
        colors = ["red", "green", "blue", "yellow"]
        for i, color in enumerate(colors):
            x, y = (i % 2) * 100, (i // 2) * 100
            draw.rectangle((x, y, x + 99, y + 99), fill=color)
        members = [{"name": f"群友{i}", "personality_tag": "原文质检员", "comment": "亲自核对了原文", "source_ids": [i]} for i in range(4)]
        with briefing.compose_member_portraits(atlas, members) as grid:
            for i, color in enumerate(colors):
                from PIL import ImageColor
                self.assertEqual(grid.getpixel(((i % 2) * 452 + 10, (i // 2) * 662 + 10)), ImageColor.getrgb(color))
            from src.plugins.llm_chat.card_templates import build_card
            doc = build_card("briefing", {"active_members": members, "summary": "不应显示"},
                             group_name="测试", group_id=1, date_label="今天", member_portraits=grid)
            self.assertNotIn("群聊收尾", [b.heading for b in doc.blocks])
            block = next(b for b in doc.blocks if b.heading == "今日群友画像")
            self.assertIs(block.image, grid)
            self.assertEqual(block.entries, [])
        self.assertNotIn("群聊收尾", briefing.build_briefing_text({"summary": "不应显示"}))
        atlas.close()

    def test_stray_markup_at_json_end_preserves_real_content(self):
        raw = '{"topics":["额度重置与煎蛋"],"summary":"A > B 是原文","portrait_prompt":"群友围坐厨房">\n}'
        card = briefing.parse_briefing_payload(raw)
        self.assertEqual(card['topics'], ['额度重置与煎蛋'])
        self.assertEqual(card['summary'], 'A > B 是原文')
        self.assertEqual(card['portrait_prompt'], '群友围坐厨房')

    def test_invalid_payload_never_becomes_generic_published_briefing(self):
        for raw in ['', '输出失败', '{}', '{"topics":[]}', '{"topics":"不是列表"}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                briefing.parse_briefing_payload(raw)

    def test_portrait_is_embedded_without_cropping_and_failure_stays_compact(self):
        card = {"topics": ["显卡讨论"], "summary": "群友各显神通"}
        args = ("测试群", 123, "2026-10-13", "用户消息 200 条")
        plain = briefing.render_briefing_image(card, *args)
        portrait = Image.new("RGB", (1536, 1024), (255, 0, 0))
        composite = briefing.render_briefing_image(card, *args, portrait=portrait)
        self.assertGreater(composite.height, plain.height + 600)
        # The full 3:2 image occupies 896x597; all its edges survive embedding.
        red_pixels = sum(count for count, color in composite.getcolors(composite.width * composite.height) if color == (255, 0, 0))
        self.assertEqual(red_pixels, 896 * 597)
        failed = briefing.render_briefing_image({**card, "portrait_status": "failed"}, *args)
        self.assertEqual(failed.height, plain.height)

    def test_render_removes_explanatory_copy_and_keeps_portrait_first(self):
        drawn = []
        original = ImageDraw.ImageDraw.text
        def capture(draw, xy, text, *args, **kwargs):
            drawn.append(str(text))
            return original(draw, xy, text, *args, **kwargs)
        with patch.object(ImageDraw.ImageDraw, "text", capture):
            briefing.render_briefing_image(
                {"briefing_model": "internal-model", "topics": ["话题"], "summary": "总结"},
                "测试群", 123, "2026-10-16", "200 条群聊消息",
                portrait=Image.new("RGB", (1536, 1024)),
            )
        text = "\n".join(drawn)
        self.assertIn("internal-model", text)
        for removed in ("Group ID", "并非成员真实照片", "群聊的一天"):
            self.assertNotIn(removed, text)
        self.assertLess(drawn.index("每日群像"), drawn.index("热门话题"))

    def test_editorial_briefing_image_renders_dynamic_sections(self):
        image = briefing.render_briefing_image(
            {
                "title": "今日群聊简报",
                "subtitle": "过去 24 小时群聊速览",
                "topics": ["群友在讨论大模型和新显卡。"],
                "active_members": ["群友甲：贡献了实测记录。"],
                "summary": "今天的信息密度很高。",
            },
            "测试群",
            123,
            "2026-08-29",
            "过去 24 小时共记录 42 条消息",
        )

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.width, briefing.SVG_WIDTH)
        self.assertGreater(image.height, 600)

    def test_pixel_wrap_does_not_leave_chinese_punctuation_on_its_own_line(self):
        font = briefing._get_font(24)
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        text_width = draw.textbbox((0, 0), "两行内容", font=font)[2]

        lines = briefing.wrap_text_pixel("两行内容。", font, text_width, draw)

        self.assertEqual(lines, ["两行内容。"])


class BriefingDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_daily_run_sends_one_composite_and_cleans_temp_file_even_without_portrait(self):
        raw = BytesIO()
        Image.new("RGB", (1536, 1024), (255, 0, 0)).save(raw, format="PNG")
        for success in (True, False):
            bot = SimpleNamespace(
                get_group_list=AsyncMock(return_value=[{"group_id": 123, "group_name": "测试群"}]),
                send_group_msg=AsyncMock(return_value={"message_id": 1}),
            )
            msg = SimpleNamespace(timestamp="2026-10-12", role="user", name="群友甲", sender_id=42,
                                  source_id=101, content="显卡评测还要亲自跑一遍")
            result = SimpleNamespace(image_bytes=raw.getvalue(), model_spec="openai:image")
            generator = AsyncMock(return_value=result) if success else AsyncMock(side_effect=RuntimeError("quota"))
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(briefing, "BRIEFING_IMAGE_DIR", Path(directory)),
                patch.object(briefing.memory_manager, "_db_path", Path(self.outbox_temp.name) / f"{success}.db"),
                patch.object(briefing, "get_bots", return_value={"bot": bot}),
                patch.object(briefing, "ensure_send_allowed"),
                patch.object(briefing, "hour", 0),
                patch.object(briefing, "minute", 0),
                patch.object(briefing.config, "enable_daily_briefing", True),
                patch.object(briefing.config, "briefing_enable_portrait", True),
                patch.object(briefing.memory_manager, "has_published_briefing_for_date", AsyncMock(return_value=False)),
                patch.object(briefing.memory_manager, "get_digest_messages", AsyncMock(return_value=([msg], 1000))),
                patch.object(briefing.model_manager, "chat", AsyncMock(return_value=json.dumps({
                    "topics": ["显卡评测"], "portrait_prompt": "群友围着显卡讨论",
                    "active_members": [{"user_id": 42, "personality_tag": "原文质检员", "comment": "结论不算数，跑过再签字。", "source_ids": [101]}],
                }, ensure_ascii=False))) as model,
                patch.object(briefing.usage_db, "query_session", AsyncMock(return_value={})) as usage,
                patch.object(briefing, "generate_image", generator),
                patch.object(briefing, "record_briefing_memory", AsyncMock()) as record,
                patch.object(briefing.asyncio, "sleep", AsyncMock()),
            ):
                await briefing._send_daily_briefing_impl(now=datetime(2026, 10, 13, 2, 0))
                model.assert_awaited_once()
                prompt = model.await_args.args[0][-1].content
                self.assertIn("#101", prompt)
                self.assertIn("user_id=42", prompt)
                self.assertIn("今日群友画像", prompt)
                self.assertIn("本人正文", prompt)
                self.assertIn("动态创造一个简短鲜明的人格标签", prompt)
                self.assertNotIn("DeepSeek·大肥鱼", prompt)
                self.assertIn("不诊断永久性格", prompt)
                self.assertIn("不限类别", prompt)
                saved_card = record.await_args.args[1]
                self.assertEqual(saved_card["active_members"][0]["name"], "群友甲")
                self.assertEqual(saved_card["active_members"][0]["source_ids"], [101])
                bot.send_group_msg.assert_awaited_once()
                usage.assert_awaited_once_with(
                    "group_123", start=datetime(2026, 10, 12), end=datetime(2026, 10, 13),
                )
                payload = bot.send_group_msg.await_args.kwargs["message"]
                encoded = payload.split("base64://", 1)[1].rstrip("]")
                with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                    colors = image.convert("RGB").getcolors(image.width * image.height)
                    has_portrait = any(color == (255, 0, 0) for _, color in colors)
                self.assertEqual(has_portrait, success)
                self.assertEqual(record.await_args.args[1]["portrait_status"], "generated" if success else "failed")
                self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_recovery_defers_generation_and_never_regenerates_a_delivered_briefing(self):
        now = datetime(2026, 10, 21)
        bot = SimpleNamespace(
            get_group_list=AsyncMock(return_value=[{"group_id": 123}]),
            send_group_msg=AsyncMock(return_value={"message_id": 1}),
        )
        msg = SimpleNamespace(timestamp="2026-10-20", role="user", name="群友", sender_id=42, content="测试")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(briefing, "BRIEFING_IMAGE_DIR", Path(directory)),
            patch.object(briefing, "get_bots", return_value={"bot": bot}),
            patch.object(briefing, "ensure_send_allowed"),
            patch.object(briefing.config, "enable_daily_briefing", True),
            patch.object(briefing.memory_manager, "has_published_briefing_for_date", AsyncMock(return_value=False)),
            patch.object(briefing.memory_manager, "get_digest_messages", AsyncMock(return_value=([msg], 0))) as messages,
            patch.object(briefing.model_manager, "chat", AsyncMock(side_effect=["{}", '{"topics":["测试"]}'])) as model,
            patch.object(briefing.usage_db, "query_session", AsyncMock(return_value={})),
            patch.object(briefing, "generate_briefing_portrait", AsyncMock(return_value=None)) as portrait,
            patch.object(briefing, "record_briefing_memory", AsyncMock()),
            patch.object(briefing.asyncio, "sleep", AsyncMock()),
        ):
            await briefing._send_daily_briefing_impl(now=now)
            model.assert_not_awaited()
            messages.return_value = ([msg], 1000)
            # Insufficient messages did not consume a generation attempt.
            await briefing._send_daily_briefing_impl(now=now)
            await briefing._send_daily_briefing_impl(now=now + timedelta(minutes=10))
            self.assertEqual(model.await_count, 1)
            portrait.assert_not_awaited()
            await briefing._send_daily_briefing_impl(now=now + timedelta(minutes=30))
            # Even with a missing structured snapshot, the durable outbox blocks duplicates.
            await briefing._send_daily_briefing_impl(now=now + timedelta(hours=3))
        self.assertEqual(model.await_count, 2)
        portrait.assert_awaited_once()
        bot.send_group_msg.assert_awaited_once()
        with closing(sqlite3.connect(briefing.memory_manager._db_path)) as db:
            state = json.loads(db.execute(
                "SELECT value FROM kv_meta WHERE key='briefing_attempts:123:2026-10-21'",
            ).fetchone()[0])
        self.assertEqual(state["attempts"], 2)

    async def test_portrait_uses_existing_generator_and_keeps_bytes_out_of_snapshot(self):
        raw = BytesIO()
        Image.new("RGB", (30, 20), "blue").save(raw, format="PNG")
        result = SimpleNamespace(image_bytes=raw.getvalue(), model_spec="openai:image")
        card = briefing.parse_briefing_payload('{"topics":["显卡"],"portrait_prompt":"三位虚构群友围着显卡讨论"}')
        with patch.object(briefing.config, "briefing_enable_portrait", True), patch.object(briefing, "generate_image", AsyncMock(return_value=result)) as generate:
            portrait = await briefing.generate_briefing_portrait(card, 123)
        self.assertEqual(portrait.size, (30, 20))
        self.assertEqual(generate.await_args.kwargs["orientation"], "landscape")
        self.assertIn(card["portrait_prompt"], generate.await_args.args[0])
        self.assertEqual(card["portrait_status"], "generated")
        self.assertFalse(any(isinstance(value, (bytes, Image.Image)) for value in card.values()))

    async def test_portrait_failure_timeout_and_disabled_do_not_block(self):
        async def stalled(*args, **kwargs):
            await asyncio.sleep(10)
        for enabled, result, outcome in (
            (True, RuntimeError("quota"), "failed"),
            (True, SimpleNamespace(image_bytes=b"invalid"), "failed"),
            (True, stalled, "failed"),
            (False, RuntimeError("must not call"), None),
        ):
            card = {"topics": ["显卡"], "active_members": ["群友甲：实测"]}
            mock = AsyncMock(side_effect=result) if isinstance(result, Exception) or callable(result) else AsyncMock(return_value=result)
            with patch.object(briefing.config, "briefing_enable_portrait", enabled), patch.object(briefing.config, "briefing_portrait_timeout", 1), patch.object(briefing, "generate_image", mock):
                self.assertIsNone(await briefing.generate_briefing_portrait(card, 123))
            self.assertEqual(card.get("portrait_status"), outcome)
            self.assertEqual(mock.await_count, int(enabled))
            if enabled:
                self.assertIn("显卡", mock.await_args.args[0])

    def setUp(self):
        self.outbox_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.outbox_temp.cleanup)
        patcher = patch.object(briefing.memory_manager, "_db_path", Path(self.outbox_temp.name) / "history.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.card_data = {
            "title": "今日群聊简报",
            "subtitle": "测试副标题",
            "topics": ["话题一"],
            "active_members": ["群友甲：测试"],
            "summary": "测试总结",
        }

    async def test_napcat_ack_timeout_does_not_send_text_fallback(self):
        timeout_error = RuntimeError("NapCat acknowledgement timeout")
        timeout_error.info = {
            "retcode": 1200,
            "message": (
                "Timeout: NTEvent serviceAndMethod:NodeIKernelMsgService/sendMsg "
                "ListenerName:NodeIKernelMsgListener/onMsgInfoListUpdate EventRet: {}"
            ),
        }
        bot = SimpleNamespace(send_group_msg=AsyncMock(side_effect=timeout_error))

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "briefing.png"
            image_path.write_bytes(b"fake-png")
            with patch.object(briefing, "record_briefing_memory", AsyncMock()) as record:
                outcome = await briefing.deliver_briefing(
                    bot,
                    123,
                    self.card_data,
                    image_path,
                )

        self.assertEqual(outcome, "image_unconfirmed")
        bot.send_group_msg.assert_awaited_once()
        sent_message = bot.send_group_msg.await_args.kwargs["message"]
        self.assertTrue(sent_message.startswith("[CQ:image,file=base64://"))
        record.assert_awaited_once_with(
            123,
            self.card_data,
            sent_as="image_unconfirmed",
            briefing_date=None,
        )

    async def test_explicit_image_failure_still_uses_text_fallback(self):
        rejected = RuntimeError("image rejected")
        rejected.info = {"retcode": 100, "message": "image rejected"}
        bot = SimpleNamespace(
            send_group_msg=AsyncMock(
                side_effect=[rejected, {"message_id": 456}]
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "briefing.png"
            image_path.write_bytes(b"fake-png")
            with patch.object(briefing, "record_briefing_memory", AsyncMock()) as record:
                outcome = await briefing.deliver_briefing(
                    bot,
                    123,
                    self.card_data,
                    image_path,
                )

        self.assertEqual(outcome, "text")
        self.assertEqual(bot.send_group_msg.await_count, 2)
        fallback_message = bot.send_group_msg.await_args_list[1].kwargs["message"]
        self.assertEqual(fallback_message, briefing.build_briefing_text(self.card_data))
        record.assert_awaited_once_with(
            123,
            self.card_data,
            sent_as="text",
            briefing_date=None,
        )

    async def test_successful_image_send_records_image_delivery(self):
        bot = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": 789}))

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "briefing.png"
            image_path.write_bytes(b"fake-png")
            with patch.object(briefing, "record_briefing_memory", AsyncMock()) as record:
                outcome = await briefing.deliver_briefing(
                    bot,
                    123,
                    self.card_data,
                    image_path,
                )

        self.assertEqual(outcome, "image")
        bot.send_group_msg.assert_awaited_once()
        record.assert_awaited_once_with(
            123,
            self.card_data,
            sent_as="image",
            briefing_date=None,
        )

    async def test_recovery_uses_previous_date_after_midnight(self):
        now = datetime(2026, 8, 28, 0, 45)
        with patch.object(briefing, "hour", 23), patch.object(briefing, "minute", 30), patch.object(briefing, "send_daily_briefing", AsyncMock()) as send:
            slot = briefing.latest_briefing_schedule_slot(now)
            self.assertEqual(slot, datetime(2026, 8, 27, 23, 30))
            await briefing.recover_missed_daily_briefing(now=now)

        send.assert_awaited_once_with(
            run_for_date=slot.date(),
            now=now,
            reason="recovery",
        )


if __name__ == "__main__":
    unittest.main()
