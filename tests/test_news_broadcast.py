import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from PIL import ImageDraw


nonebot.init()

from src.plugins.llm_chat import news_broadcast
from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.models import ChatResponse


class NewsPayloadTests(unittest.TestCase):
    def test_parser_keeps_only_recent_linked_unique_news(self):
        raw = json.dumps(
            {
                "intro": "这是今天值得看的更新。",
                "items": [
                    {
                        "topic": "大模型",
                        "headline": "模型发布新版本",
                        "summary": "新版本改进了推理能力。",
                        "source": "示例科技媒体",
                        "published_at": "2026-08-26T09:00:00+08:00",
                        "url": "https://example.com/model-v2",
                    },
                    {
                        "topic": "大模型",
                        "headline": "模型发布新版本",
                        "summary": "同一篇新闻的重复结果。",
                        "source": "重复来源",
                        "published_at": "2026-08-26",
                        "url": "https://mirror.example.com/model-v2",
                    },
                    {
                        "topic": "游戏",
                        "headline": "过期新闻",
                        "summary": "这条已经超出时间窗。",
                        "source": "游戏媒体",
                        "published_at": "2026-08-22",
                        "url": "https://example.com/old-news",
                    },
                    {
                        "topic": "硬件",
                        "headline": "没有有效链接",
                        "summary": "链接协议不受支持。",
                        "source": "硬件媒体",
                        "published_at": "2026-08-25",
                        "url": "ftp://example.com/hardware",
                    },
                ],
            },
            ensure_ascii=False,
        )

        payload = news_broadcast.parse_news_payload(
            raw,
            earliest_date=date(2026, 8, 24),
            latest_date=date(2026, 8, 26),
            max_items=5,
        )

        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["headline"], "模型发布新版本")
        self.assertEqual(payload["focus_topics"], ["大模型"])

    def test_parser_distinguishes_valid_empty_result_from_malformed_output(self):
        payload = news_broadcast.parse_news_payload(
            '{"intro":"没有合格新闻","items":[]}',
            earliest_date=date(2026, 8, 24),
            latest_date=date(2026, 8, 26),
            max_items=5,
        )
        self.assertEqual(payload["items"], [])
        self.assertIsNone(
            news_broadcast.parse_news_payload(
                "不是 JSON",
                earliest_date=date(2026, 8, 24),
                latest_date=date(2026, 8, 26),
                max_items=5,
            )
        )

    def test_search_prompt_only_exposes_topics(self):
        card_data = {
            "topics": ["大家在讨论某款新显卡"],
            "active_members": ["真实群友昵称：说了很多话"],
            "summary": "这里是完整群聊总结，不应发送给搜索模型。",
        }
        prompt = news_broadcast.build_news_search_prompt(
            card_data,
            now=datetime(2026, 8, 26, 8, 0),
        )
        self.assertIn("某款新显卡", prompt)
        self.assertNotIn("真实群友昵称", prompt)
        self.assertNotIn("完整群聊总结", prompt)

    def test_native_selection_uses_real_source_metadata(self):
        raw = json.dumps(
            {
                "items": [
                    {
                        "source_index": 2,
                        "published_at": "2026-08-26",
                        "editor_title": "新模型正式开放使用",
                        "summary": "模型发布了新版本。[2]",
                        "why_it_matters": "开发者现在可以直接验证新能力。",
                        "watch_for": "官方基准测试与价格说明。",
                        "signal": "明确进展",
                        "newsworthiness": 5,
                        "url": "https://fake.example/hallucinated",
                    },
                    {
                        "source_index": 1,
                        "published_at": "2026-08-26",
                        "summary": "自动热点汇总。",
                    },
                    {
                        "source_index": 3,
                        "published_at": "2026-08-26",
                        "summary": "同一媒体的第二条。",
                    },
                    {
                        "source_index": 4,
                        "published_at": "2026-08-26",
                        "summary": "游戏公司背景介绍。",
                        "newsworthiness": 2,
                    },
                ]
            },
            ensure_ascii=False,
        )
        sources = [
            {
                "index": 1,
                "title": "今日热点小时报",
                "site_name": "媒体乙",
                "url": "https://example.com/roundup",
            },
            {
                "index": 2,
                "title": "真实搜索标题",
                "published_at": "2026-08-26",
                "site_name": "媒体乙",
                "url": "https://example.com/real",
            },
            {
                "index": 3,
                "title": "同一媒体的另一篇",
                "site_name": "媒体乙的另一个展示名",
                "url": "https://www.example.com/same-site",
            },
            {
                "index": 4,
                "title": "这款游戏开发商是谁？公司背景与制作团队介绍",
                "site_name": "攻略站",
                "url": "https://guide.example.com/company",
            },
        ]

        items = news_broadcast.parse_native_news_selection(
            raw,
            sources,
            interest_label="大模型",
            earliest_date=date(2026, 8, 24),
            latest_date=date(2026, 8, 26),
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["headline"], "真实搜索标题")
        self.assertEqual(items[0]["source"], "媒体乙")
        self.assertEqual(items[0]["url"], "https://example.com/real")
        self.assertEqual(items[0]["editor_title"], "新模型正式开放使用")
        self.assertEqual(items[0]["signal"], "明确进展")
        self.assertEqual(items[0]["newsworthiness"], 5)
        self.assertIn("直接验证", items[0]["why_it_matters"])
        self.assertEqual(items[0]["watch_for"], "")
        self.assertNotIn("[2]", items[0]["summary"])

    def test_confirmed_followup_uses_structured_flag(self):
        source = {"index": 1, "title": "开发商回应玩家提问", "site_name": "媒体",
                  "url": "https://example.com/next", "published_at": "2026-08-26"}
        item = {"source_index": 1, "summary": "开发商确认了下一次发布安排。",
                "watch_for": "9月17日发布新版本", "watch_for_confirmed": True, "newsworthiness": 4}
        result = news_broadcast.parse_native_news_selection(
            json.dumps({"items": [item]}), [source], interest_label="游戏",
            earliest_date=date(2026, 8, 25), latest_date=date(2026, 8, 26),
        )
        self.assertEqual(result[0]["watch_for"], item["watch_for"])

    def test_ranking_picks_strongest_hero_then_prefers_topic_diversity(self):
        items = [
            {
                "topic": "AI",
                "headline": "AI 工具发布新版本",
                "published_at": "2026-08-30",
                "newsworthiness": 5,
                "why_it_matters": "有明确的新能力。",
            },
            {
                "topic": "AI",
                "headline": "AI 公司公布更多细节",
                "published_at": "2026-08-30",
                "newsworthiness": 4,
            },
            {
                "topic": "游戏",
                "headline": "新作确认发售日期",
                "published_at": "2026-08-29",
                "newsworthiness": 3,
            },
        ]

        ranked = news_broadcast.rank_news_items(
            items,
            now=datetime(2026, 8, 30, 8, 0),
            max_items=2,
        )

        self.assertEqual(ranked[0]["headline"], "AI 工具发布新版本")
        self.assertEqual(ranked[1]["topic"], "游戏")

    def test_news_image_uses_briefing_dimensions_and_contains_all_items(self):
        payload = {
            "title": "群友关注 · 今日新闻",
            "intro": "大家最近比较关心新模型和游戏动态。",
            "items": [
                {
                    "topic": "大模型",
                    "headline": "模型发布新版本",
                    "editor_title": "新模型把复杂推理再往前推了一步",
                    "summary": "新版本改进了推理能力，也补充了新的开发工具。",
                    "why_it_matters": "开发者可以更快验证复杂任务是否真正可用。",
                    "watch_for": "价格与正式基准测试。",
                    "signal": "明确进展",
                    "source": "示例科技媒体",
                    "published_at": "2026-08-29",
                    "url": "https://example.com/model-v2",
                },
                {
                    "topic": "游戏",
                    "headline": "新作公布发售日期",
                    "editor_title": "备受关注的新作终于定档",
                    "summary": "官方公开了发售时间和首支实机演示。",
                    "why_it_matters": "这次终于从传闻进入可验证的发布节奏。",
                    "signal": "风向变化",
                    "source": "示例游戏媒体",
                    "published_at": "2026-08-28",
                    "url": "https://example.com/game",
                },
            ],
        }

        drawn = []
        original = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            drawn.append(str(text))
            return original(draw, xy, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", capture):
            image = news_broadcast.render_news_image(payload, "测试群", 456, "2026-08-29")

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.width, news_broadcast.SVG_WIDTH)
        for item in payload["items"]:
            self.assertIn(item["editor_title"], "".join(drawn))
            self.assertIn(item["summary"], "".join(drawn))


class NewsBroadcastAsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.outbox_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.outbox_temp.cleanup)
        patcher = patch.object(news_broadcast.memory_manager, "_db_path", Path(self.outbox_temp.name) / "history.db")
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_sent_briefing_is_consumed_only_once_across_persistent_store(self):
        manager = object.__new__(MemoryManager)
        manager._init()
        with tempfile.TemporaryDirectory() as temp_dir:
            manager._db_path = Path(temp_dir) / "history.db"
            published_at = datetime(2026, 8, 25, 23, 30)
            await manager.save_published_briefing(
                456,
                {"topics": ["大模型"]},
                sent_as="image",
                published_at=published_at,
                briefing_date="2026-08-25",
            )
            self.assertTrue(
                await manager.has_published_briefing_for_date(456, "2026-08-25")
            )
            self.assertTrue(
                await manager.has_pending_briefing_for_news(
                    published_after=datetime(2026, 8, 25, 0, 0)
                )
            )
            pending = await manager.get_pending_briefing_for_news(
                456,
                published_after=datetime(2026, 8, 25, 0, 0),
            )
            self.assertIsNotNone(pending)
            await manager.save_news_broadcast(
                pending["id"],
                456,
                {"items": [{"headline": "已发送"}]},
                sent_as="image",
                broadcast_at=datetime(2026, 8, 26, 8, 0),
            )
            consumed = await manager.get_pending_briefing_for_news(
                456,
                published_after=datetime(2026, 8, 25, 0, 0),
            )
            self.assertIsNone(consumed)
            self.assertFalse(
                await manager.has_pending_briefing_for_news(
                    published_after=datetime(2026, 8, 25, 0, 0)
                )
            )
            latest = await manager.get_latest_published_briefing(
                456,
                published_after=datetime(2026, 8, 25, 0, 0),
            )
            self.assertEqual(latest["id"], pending["id"])
            await manager.save_news_broadcast(
                pending["id"],
                456,
                {"items": [{"headline": "补跑成功"}]},
                sent_as="text",
                broadcast_at=datetime(2026, 8, 26, 9, 0),
            )
            db = await manager._get_db()
            async with db.execute(
                "SELECT sent_as, payload_json FROM news_broadcasts WHERE briefing_id = ?",
                (pending["id"],),
            ) as cursor:
                saved = await cursor.fetchone()
            self.assertEqual(saved[0], "image")
            self.assertIn("已发送", saved[1])
            await manager.close()

    async def test_interest_extraction_only_exposes_topics(self):
        response = ChatResponse(
            '{"intro":"大家最近比较关心新显卡。","interests":'
            '[{"label":"显卡","query":"新显卡 发布"}]}',
            model_spec="google:example",
        )
        card_data = {
            "topics": ["某款新显卡"],
            "active_members": ["真实群友昵称"],
            "summary": "完整群聊总结",
        }
        with patch.object(news_broadcast.model_manager, "chat", AsyncMock(return_value=response)) as chat:
            interests = await news_broadcast.extract_news_interests(card_data)
        self.assertEqual(interests, [{"label": "显卡", "query": "新显卡 发布"}])
        self.assertEqual(chat.await_args.kwargs['role'], 'news_interest')
        exposed = "\n".join(message.content for message in chat.await_args.args[0])
        self.assertIn("某款新显卡", exposed)
        self.assertNotIn("真实群友昵称", exposed)
        self.assertNotIn("完整群聊总结", exposed)

    async def test_news_text_omits_old_intro_and_fixed_editorial_copy(self):
        text = news_broadcast.build_news_text(
            {
                "title": "群友关注 · 今日新闻",
                "intro": "大家最近比较关心新模型和游戏动态。",
                "focus_topics": ["新模型", "游戏"],
                "items": [],
            }
        )
        self.assertNotIn("大家最近比较关心新模型和游戏动态。", text)
        self.assertNotIn("从最近的群聊简报里挑了", text)

    async def test_news_recovery_runs_inside_same_day_window(self):
        now = datetime(2026, 8, 28, 8, 15)
        self.assertEqual(
            news_broadcast.latest_news_schedule_slot(now),
            datetime(2026, 8, 28, 8, 0),
        )
        with patch.object(
            news_broadcast,
            "send_daily_news_broadcast",
            AsyncMock(),
        ) as send:
            await news_broadcast.recover_missed_daily_news_broadcast(now=now)
        send.assert_awaited_once_with(now=now, reason="recovery")

    async def test_generate_returns_none_when_all_native_searches_fail(self):
        with patch.object(
            news_broadcast,
            "extract_news_plan",
            AsyncMock(
                return_value={
                    "intro": "大家最近比较关心大模型。",
                    "interests": [{"label": "大模型", "query": "大模型 发布"}],
                }
            ),
        ), patch.object(
            news_broadcast,
            "search_news_interest",
            AsyncMock(side_effect=RuntimeError("search unavailable")),
        ):
            payload = await news_broadcast.generate_news_payload(
                {"topics": ["大模型"]},
                now=datetime(2026, 8, 26, 8, 0),
            )
        self.assertIsNone(payload)

    async def test_generate_combines_and_deduplicates_native_results(self):
        real_item = {
            "topic": "大模型",
            "headline": "模型发布新版本",
            "summary": "新版本改进了推理能力。",
            "source": "示例科技媒体",
            "published_at": "2026-08-26",
            "url": "https://example.com/model-v2",
        }
        with patch.object(
            news_broadcast,
            "extract_news_plan",
            AsyncMock(
                return_value={
                    "intro": "大家最近比较关心大模型和 AI。",
                    "interests": [
                        {"label": "大模型", "query": "大模型 发布"},
                        {"label": "AI", "query": "AI 发布"},
                    ],
                }
            ),
        ), patch.object(
            news_broadcast,
            "search_news_interest",
            AsyncMock(return_value={"items": [real_item], "model_spec": "dashscope:qwen"}),
        ):
            payload = await news_broadcast.generate_news_payload(
                {"topics": ["大模型"]},
                now=datetime(2026, 8, 26, 8, 0),
            )
        self.assertEqual(payload["items"], [real_item])
        self.assertEqual(payload["focus_topics"], ["大模型"])
        self.assertEqual(payload["intro"], "")

    async def test_delivery_sends_image_without_trailing_text_and_records_memory(self):
        bot = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": 123}))
        briefing = {"id": 7, "group_id": 456}
        payload = {
            "title": "群友关注 · 今日新闻",
            "intro": "",
            "focus_topics": ["大模型"],
            "items": [
                {
                    "topic": "大模型",
                    "headline": "模型发布新版本",
                    "summary": "新版本改进了推理能力。",
                    "source": "示例科技媒体",
                    "published_at": "2026-08-26",
                    "url": "https://example.com/model-v2",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            news_broadcast,
            "NEWS_IMAGE_DIR",
            Path(temp_dir),
        ), patch.object(
            news_broadcast,
            "_record_news_result",
            AsyncMock(),
        ) as record:
            outcome = await news_broadcast.deliver_news_broadcast(
                bot,
                briefing,
                payload,
                group_name="测试群",
                now=datetime(2026, 8, 29, 8, 0),
            )

        self.assertEqual(outcome, "image")
        bot.send_group_msg.assert_awaited_once()
        record.assert_awaited_once()
        sent_message = bot.send_group_msg.await_args.kwargs["message"]
        self.assertTrue(sent_message.startswith("[CQ:image,file=base64://"))
        self.assertTrue(sent_message.endswith("]"))
        self.assertNotIn("\n", sent_message)
        self.assertNotIn("https://example.com/model-v2", sent_message)
        self.assertEqual(record.await_args.kwargs["sent_as"], "image")

    async def test_delivery_falls_back_to_text_when_image_send_explicitly_fails(self):
        rejected = RuntimeError("image rejected")
        rejected.info = {"retcode": 100, "message": "image rejected"}
        bot = SimpleNamespace(
            send_group_msg=AsyncMock(
                side_effect=[rejected, {"message_id": 124}]
            )
        )
        briefing = {"id": 7, "group_id": 456}
        payload = {
            "title": "群友关注 · 今日新闻",
            "items": [
                {
                    "topic": "大模型",
                    "headline": "模型发布新版本",
                    "summary": "新版本改进了推理能力。",
                    "source": "示例科技媒体",
                    "published_at": "2026-08-29",
                    "url": "https://example.com/model-v2",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            news_broadcast,
            "NEWS_IMAGE_DIR",
            Path(temp_dir),
        ), patch.object(
            news_broadcast,
            "_record_news_result",
            AsyncMock(),
        ) as record:
            outcome = await news_broadcast.deliver_news_broadcast(bot, briefing, payload)

        self.assertEqual(outcome, "text")
        self.assertEqual(bot.send_group_msg.await_count, 2)
        self.assertFalse(
            bot.send_group_msg.await_args_list[1].kwargs["message"].startswith("[CQ:image")
        )
        self.assertEqual(record.await_args.kwargs["sent_as"], "text")

    async def test_delivery_does_not_duplicate_after_ambiguous_image_timeout(self):
        timeout_error = RuntimeError("timeout")
        timeout_error.info = {
            "retcode": 1200,
            "message": "Error: timeout in SendMsg",
            "wording": "OnMsgInfoListUpdate timeout",
        }
        bot = SimpleNamespace(send_group_msg=AsyncMock(side_effect=timeout_error))
        briefing = {"id": 7, "group_id": 456}
        payload = {
            "title": "群友关注 · 今日新闻",
            "items": [
                {
                    "topic": "大模型",
                    "headline": "模型发布新版本",
                    "summary": "新版本改进了推理能力。",
                    "source": "示例科技媒体",
                    "published_at": "2026-08-29",
                    "url": "https://example.com/model-v2",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            news_broadcast,
            "NEWS_IMAGE_DIR",
            Path(temp_dir),
        ), patch.object(
            news_broadcast,
            "_record_news_result",
            AsyncMock(),
        ) as record:
            outcome = await news_broadcast.deliver_news_broadcast(bot, briefing, payload)

        self.assertEqual(outcome, "image_unconfirmed")
        self.assertEqual(bot.send_group_msg.await_count, 1)
        self.assertEqual(record.await_args.kwargs["sent_as"], "image_unconfirmed")


if __name__ == "__main__":
    unittest.main()
