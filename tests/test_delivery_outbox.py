import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from PIL import Image

nonebot.init()

from src.plugins.llm_chat import briefing, conversation, delivery, news_broadcast as news
from src.plugins.llm_chat.delivery_outbox import DeliveryOutbox


def rejected():
    error = RuntimeError("image rejected")
    error.info = {"retcode": 100, "message": "image rejected"}
    return error


class DeliveryOutboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.db"
        self.outbox = DeliveryOutbox(self.path)
        self.bot = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": 42}))
        self.key = "briefing:71:2026-10-17"
        conversation.start()

    async def send(self, *, image="image", outbox=None):
        return await delivery.deliver_digest(
            self.bot, 71, outbox=outbox or self.outbox, key=self.key,
            payload={"summary": "test"}, text="text", image=image,
        )

    def status(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT status FROM delivery_outbox WHERE delivery_key=?", (self.key,)).fetchone()[0]

    async def test_concurrent_workers_send_once(self):
        outcomes = await asyncio.gather(self.send(), self.send(outbox=DeliveryOutbox(self.path)))
        self.assertCountEqual(outcomes, ["image", "already_attempted"])
        self.bot.send_group_msg.assert_awaited_once()
        self.assertEqual(self.status(), "sent")

    async def test_old_attempt_cannot_finalize_a_new_claim(self):
        old = await self.outbox.claim(self.key, group_id=71, medium="image", payload={"text": "private"})
        await self.outbox.finish(self.key, "failed", attempt_token=old)
        current = await self.outbox.claim(self.key, group_id=71, medium="text", payload={"text": "private"})
        await self.outbox.finish(self.key, "sent", attempt_token=old, message_id=1)
        self.assertEqual(self.status(), "sending")
        await self.outbox.finish(self.key, "sent", attempt_token=current, message_id=2)
        self.assertEqual(self.status(), "sent")
        with closing(sqlite3.connect(self.path)) as db:
            record = db.execute("SELECT * FROM delivery_outbox").fetchone()
        self.assertNotIn("private", str(record))

    async def test_fallback_text_timeout_blocks_next_run_after_restart(self):
        self.bot.send_group_msg.side_effect = [rejected(), TimeoutError("missing text receipt")]
        self.assertEqual(await self.send(), "text_unconfirmed")
        self.assertEqual(self.status(), "unknown")
        self.assertEqual(await self.send(outbox=DeliveryOutbox(self.path)), "already_attempted")
        self.assertEqual(self.bot.send_group_msg.await_count, 2)

    async def test_generic_transport_error_does_not_trigger_text_fallback(self):
        self.bot.send_group_msg.side_effect = ConnectionError("connection dropped")
        self.assertEqual(await self.send(), "image_unconfirmed")
        self.bot.send_group_msg.assert_awaited_once()
        self.assertEqual(self.status(), "unknown")

    async def test_generic_onebot_timeout_is_not_a_definite_rejection(self):
        error = RuntimeError("timeout")
        error.info = {"retcode": 1200, "message": "request timeout"}
        self.bot.send_group_msg.side_effect = error
        self.assertEqual(await self.send(), "image_unconfirmed")
        self.bot.send_group_msg.assert_awaited_once()
        self.assertEqual(self.status(), "unknown")

    async def test_missing_message_receipt_is_unknown(self):
        self.bot.send_group_msg.return_value = {}
        self.assertEqual(await self.send(image=None), "text_unconfirmed")
        self.assertEqual(self.status(), "unknown")

    async def test_explicit_text_rejection_can_retry(self):
        self.bot.send_group_msg.side_effect = [rejected(), {"message_id": 43}]
        with self.assertRaises(RuntimeError):
            await self.send(image=None)
        self.assertEqual(self.status(), "failed")
        self.assertEqual(await self.send(image=None), "text")

    async def test_checkpoint_failure_keeps_durable_claim(self):
        with patch.object(self.outbox, "finish", AsyncMock(side_effect=OSError("disk unavailable"))):
            self.assertEqual(await self.send(), "image")
        self.assertEqual(self.status(), "sending")
        self.assertEqual(await self.send(outbox=DeliveryOutbox(self.path)), "already_attempted")
        self.bot.send_group_msg.assert_awaited_once()

    async def test_cancelled_inflight_send_is_not_retried(self):
        started = asyncio.Event()
        async def blocked_send(**kwargs):
            started.set()
            await asyncio.Event().wait()
        self.bot.send_group_msg.side_effect = blocked_send
        task = asyncio.create_task(self.send())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.status(), "unknown")
        self.assertEqual(await self.send(outbox=DeliveryOutbox(self.path)), "already_attempted")
        self.bot.send_group_msg.assert_awaited_once()

    async def test_invalidation_after_claim_prevents_send_without_poisoning_retry(self):
        original_claim = self.outbox.claim
        async def claim_then_reset(*args, **kwargs):
            result = await original_claim(*args, **kwargs)
            conversation.invalidate("group_71", forget=True)
            return result
        with conversation.session_scope("group_71"), patch.object(self.outbox, "claim", claim_then_reset):
            with self.assertRaises(conversation.TurnSuperseded):
                await self.send()
        self.bot.send_group_msg.assert_not_awaited()
        self.assertEqual(self.status(), "failed")
        self.assertEqual(await self.send(), "image")

    async def test_briefing_snapshot_failure_cannot_repeat_network_send(self):
        image = Path(self.temp.name) / "image.png"
        image.write_bytes(b"test")
        with (
            patch.object(briefing.memory_manager, "_db_path", self.path),
            patch.object(briefing.memory_manager, "add_message", AsyncMock()),
            patch.object(briefing.memory_manager, "save_published_briefing", AsyncMock(side_effect=OSError("disk unavailable"))),
        ):
            for expected in ("image", "already_attempted"):
                result = await briefing.deliver_briefing(
                    self.bot, 71, {"summary": "test"}, image, briefing_date="2026-10-17",
                )
                self.assertEqual(result, expected)
        self.bot.send_group_msg.assert_awaited_once()

    async def test_briefing_reset_during_generation_skips_only_affected_group(self):
        async def generate(*args, **kwargs):
            if kwargs["session_id"] == "group_71":
                conversation.invalidate("group_71", forget=True)
            return '{"topics":["test"],"summary":"test"}'
        message = SimpleNamespace(timestamp="2026-10-16", role="user", name="user", sender_id=1, content="test")
        self.bot.get_group_list = AsyncMock(return_value=[{"group_id": 71}, {"group_id": 72}])
        with (
            patch.object(briefing.memory_manager, "_db_path", self.path),
            patch.object(briefing, "BRIEFING_IMAGE_DIR", Path(self.temp.name)),
            patch.object(briefing, "get_bots", return_value={"bot": self.bot}),
            patch.object(briefing.config, "enable_daily_briefing", True),
            patch.object(briefing.memory_manager, "has_published_briefing_for_date", AsyncMock(return_value=False)),
            patch.object(briefing.memory_manager, "get_digest_messages", AsyncMock(return_value=([message], 1000))),
            patch.object(briefing.model_manager, "chat", AsyncMock(side_effect=generate)),
            patch.object(briefing.usage_db, "query_session", AsyncMock(return_value={})),
            patch.object(briefing, "generate_briefing_portrait", AsyncMock(return_value=None)),
            patch.object(briefing, "render_briefing_image", return_value=Image.new("RGB", (2, 2))),
            patch.object(briefing, "record_briefing_memory", AsyncMock()) as record,
            patch.object(briefing.asyncio, "sleep", AsyncMock()),
        ):
            await briefing._send_daily_briefing_impl(now=datetime(2026, 10, 17, 0, 0))
        self.bot.send_group_msg.assert_awaited_once()
        self.assertEqual(self.bot.send_group_msg.await_args.kwargs["group_id"], 72)
        self.assertEqual(record.await_args.args[0], 72)

    async def test_news_reset_during_generation_skips_only_affected_group(self):
        payload = {"items": [{"url": "https://example.com/a"}]}
        async def pending(group_id, **kwargs):
            return {"id": group_id, "group_id": group_id, "card_data": {}}
        async def generate(*args, **kwargs):
            if kwargs["session_id"] == "group_71":
                conversation.invalidate("group_71", forget=True)
            return payload
        async def guarded_delivery(bot, row, *args, **kwargs):
            conversation.ensure_send_allowed(row["group_id"])
            await bot.send_group_msg(group_id=row["group_id"], message="test")
            return "text"
        with (
            patch.object(news.memory_manager, "_db_path", self.path),
            patch.object(news.memory_manager, "get_pending_briefing_for_news", AsyncMock(side_effect=pending)),
            patch.object(news.memory_manager, "recent_news_topics", AsyncMock(return_value=[])),
            patch.object(news.memory_manager, "recent_news_urls", AsyncMock(return_value=set())),
            patch.object(news, "generate_news_payload", AsyncMock(side_effect=generate)),
            patch.object(news, "deliver_news_broadcast", AsyncMock(side_effect=guarded_delivery)),
            patch.object(news.asyncio, "sleep", AsyncMock()),
        ):
            await news.run_daily_news_broadcast(self.bot, [{"group_id": 71}, {"group_id": 72}])
        self.bot.send_group_msg.assert_awaited_once()
        self.assertEqual(self.bot.send_group_msg.await_args.kwargs["group_id"], 72)

    async def test_history_dedup_precedes_ranking_and_limit(self):
        items = [dict(url=f"https://example.com/{i}", headline=f"news{i}", topic=f"topic{i//2}",
                      source="source", published_at="2026-10-17", newsworthiness=5 if i < 5 else 1)
                 for i in range(6)]
        with (
            patch.object(news.config, "news_max_items", 5),
            patch.object(news, "extract_news_plan", AsyncMock(return_value={"interests": [{}, {}, {}]})),
            patch.object(news, "search_news_interest", AsyncMock(side_effect=[
                {"items": items[i:i+2], "model_spec": "openai:test"} for i in range(0, 6, 2)
            ])),
            patch.object(news, "format_model_spec_label", return_value="test"),
        ):
            payload = await news.generate_news_payload(
                {"topics": ["test"]}, now=datetime(2026, 10, 17, 8),
                excluded_urls={item["url"] for item in items[:5]},
            )
        self.assertEqual(payload["items"], [items[5]])
        self.assertEqual(payload["focus_topics"], [items[5]["topic"]])


if __name__ == "__main__":
    unittest.main()
