import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot
from nonebot.adapters.onebot.v11 import Message, MessageSegment

nonebot.init()

from src.plugins.llm_chat import incoming, chat
from src.plugins.llm_chat.chat import record_group_message, resolve_incoming_message
from src.plugins.llm_chat.models import Message as LLMMessage


class FakeEvent(SimpleNamespace):
    def get_plaintext(self):
        return self.message.extract_plain_text()


def stored_message(_user_id, role, content, *args, **kwargs):
    return LLMMessage(role=role, content=content, source_id=1, timestamp=kwargs.get("timestamp"))


def make_event(
    message,
    *,
    original_message=None,
    reply=None,
    message_id=100,
):
    return FakeEvent(
        self_id=999,
        user_id=123,
        group_id=456,
        message_id=message_id,
        message=message,
        original_message=original_message or Message(message),
        reply=reply,
        to_me=True,
    )


class IncomingMessageExpansionTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        incoming._mention_names.clear()
        chat.image_reference_store.clear()

    async def test_direct_mentions_preserve_target_names_ids_and_spacing(self):
        event = make_event(Message([
            MessageSegment.at(999), MessageSegment.text("请问 "),
            MessageSegment.at(42), MessageSegment.text(" 你怎么看 "),
            MessageSegment.at("all"),
        ]))
        bot = SimpleNamespace(call_api=AsyncMock(return_value={"card": "xxx_user"}))
        resolved = await resolve_incoming_message(bot, event)
        self.assertIn("@你(QQ:999)", resolved.plain_text)
        self.assertIn("@xxx_user(QQ:42)", resolved.plain_text)
        self.assertIn("@全体成员", resolved.plain_text)
        self.assertNotIn("@你", resolved.response_text)
        self.assertIn("请问 @xxx_user(QQ:42)", resolved.response_text)
        bot.call_api.assert_awaited_once_with("get_group_member_info", group_id=456, user_id=42)

    async def test_mention_lookup_failure_keeps_id_and_does_not_drop_literal_at_text(self):
        event = make_event(Message([MessageSegment.at(42), MessageSegment.text(" @xxx_user 看这里")]))
        bot = SimpleNamespace(call_api=AsyncMock(side_effect=RuntimeError("unavailable")))
        resolved = await resolve_incoming_message(bot, event)
        self.assertIn("@QQ:42", resolved.plain_text)
        self.assertIn("@xxx_user 看这里", resolved.plain_text)

    async def test_forward_mentions_resolve_from_other_nodes_without_member_api(self):
        event = make_event(Message([MessageSegment("forward", {"content": [
            {"sender": {"user_id": 1, "nickname": "甲"}, "content": "找 [CQ:at,qq=2] 看看"},
            {"sender": {"user_id": 2, "nickname": "乙"}, "content": "好的"},
        ]})]))
        bot = SimpleNamespace(call_api=AsyncMock())
        resolved = await resolve_incoming_message(bot, event)
        self.assertIn("找 @乙(QQ:2) 看看", resolved.compose())
        bot.call_api.assert_not_awaited()

    async def test_quoted_mentions_and_image_file_url_are_preserved(self):
        event = make_event(Message("参考这个图片，改成红色"), reply={
            "sender": {"user_id": 42, "nickname": "小甲"},
            "message": "给 [CQ:at,qq=42] 的图[CQ:image,file=https://example.com/image.png]",
        })
        resolved = await resolve_incoming_message(SimpleNamespace(call_api=AsyncMock()), event)
        self.assertIn("给 @小甲(QQ:42) 的图", resolved.compose())
        self.assertEqual(resolved.image_urls, ["https://example.com/image.png"])
        self.assertFalse(resolved.unavailable_images)

    async def test_unreadable_image_marks_reference_unavailable(self):
        event = make_event(Message("照这张改") + MessageSegment.image("missing-file"))
        resolved = await resolve_incoming_message(SimpleNamespace(call_api=AsyncMock()), event)
        self.assertTrue(resolved.unavailable_images)
        self.assertTrue(chat._missing_image_references(event, []))

    async def test_recorder_keeps_mention_only_messages(self):
        event = make_event(Message([MessageSegment("at", {"qq": "42", "name": "甲"})]))
        event.sender = SimpleNamespace(card="乙", nickname="乙", user_id=123)
        event.to_me = False
        saved = AsyncMock(side_effect=stored_message)
        with (
            patch.object(chat.config, "enable_group_repeat", False),
            patch.object(chat.config, "enable_sentiment_analysis", False),
            patch.object(chat.memory_manager, "add_message", saved),
            patch.object(chat.memory_manager, "enrich_message", AsyncMock(return_value=True)) as enriched,
            patch.object(chat.proactive_manager, "try_proactive", AsyncMock()),
        ):
            await record_group_message(SimpleNamespace(call_api=AsyncMock()), event)
        saved.assert_awaited_once()
        enriched.assert_awaited_once()
        self.assertIsInstance(enriched.await_args.args[1], LLMMessage)
        self.assertIn("@甲(QQ:42)", enriched.await_args.args[2])

    async def test_uses_adapter_resolved_reply_without_duplicate_get_msg(self):
        image_url = "https://example.com/quoted.png"
        event = make_event(
            Message([MessageSegment.text("这是什么意思")]),
            original_message=Message(
                [
                    MessageSegment.reply(42),
                    MessageSegment.at(100),
                    MessageSegment.text("这是什么意思"),
                ]
            ),
            reply=SimpleNamespace(
                sender=SimpleNamespace(
                    user_id=100,
                    card="甲",
                    nickname="Alice",
                ),
                message=Message(
                    [
                        MessageSegment.text("被引用的原话"),
                        MessageSegment("image", {"url": image_url, "file": "quoted.png"}),
                    ]
                ),
            ),
        )
        bot = SimpleNamespace(call_api=AsyncMock())

        resolved = await resolve_incoming_message(bot, event)
        resolved_again = await resolve_incoming_message(bot, event)

        self.assertIs(resolved_again, resolved)
        self.assertEqual(resolved.plain_text, "这是什么意思")
        self.assertIn("[引用回复]", resolved.compose())
        self.assertIn("被回复者：甲", resolved.compose())
        self.assertIn("被引用的原话[图片]", resolved.compose())
        self.assertIn("[当前消息]\n这是什么意思", resolved.compose())
        self.assertEqual(resolved.context_image_urls, [image_url])
        self.assertEqual(resolved.direct_image_urls, [])
        bot.call_api.assert_not_awaited()

    async def test_falls_back_to_original_reply_segment_when_adapter_has_no_reply(self):
        event = make_event(
            Message([MessageSegment.reply(77), MessageSegment.text("继续说")]),
            reply=None,
        )
        event.to_me = False
        bot = SimpleNamespace(
            call_api=AsyncMock(
                return_value={
                    "sender": {"user_id": 999, "nickname": "Bot"},
                    "message": [
                        {"type": "text", "data": {"text": "上一条回答"}},
                    ],
                }
            )
        )

        resolved = await resolve_incoming_message(bot, event)

        self.assertIn("被回复者：你", resolved.compose())
        self.assertIn("原消息：上一条回答", resolved.compose())
        self.assertTrue(event.to_me)
        bot.call_api.assert_awaited_once_with("get_msg", message_id=77)

    async def test_expands_standard_forward_and_collects_context_images(self):
        image_url = "https://example.com/forwarded.jpg"
        event = make_event(
            Message(
                [
                    MessageSegment.forward("forward-1"),
                    MessageSegment.text("帮我总结"),
                ]
            )
        )
        bot = SimpleNamespace(
            call_api=AsyncMock(
                return_value={
                    "messages": [
                        {
                            "sender": {"nickname": "甲", "user_id": 1},
                            "content": [
                                {"type": "text", "data": {"text": "第一条"}},
                                {"type": "image", "data": {"url": image_url}},
                            ],
                        },
                        {
                            "sender": {"nickname": "乙", "user_id": 2},
                            "content": "第二条",
                        },
                    ]
                }
            )
        )

        resolved = await resolve_incoming_message(bot, event)
        content = resolved.compose()

        self.assertIn("[合并转发]", content)
        self.assertIn("甲: 第一条[图片]", content)
        self.assertIn("乙: 第二条", content)
        self.assertIn("[当前消息]\n帮我总结", content)
        self.assertEqual(resolved.context_image_urls, [image_url])
        bot.call_api.assert_awaited_once_with(
            "get_forward_msg",
            id="forward-1",
        )

    async def test_parses_legacy_cq_content_inside_forward_nodes(self):
        image_url = "https://example.com/legacy.jpg"
        event = make_event(Message([MessageSegment.forward("legacy-forward")]))
        bot = SimpleNamespace(
            call_api=AsyncMock(
                return_value={
                    "messages": [
                        {
                            "sender": {"nickname": "甲", "user_id": 1},
                            "content": (
                                "旧格式文字"
                                f"[CQ:image,file=legacy.jpg,url={image_url}]"
                            ),
                        }
                    ]
                }
            )
        )

        resolved = await resolve_incoming_message(bot, event)

        self.assertIn("甲: 旧格式文字[图片]", resolved.compose())
        self.assertEqual(resolved.context_image_urls, [image_url])

    async def test_json_forward_uses_preview_when_api_cannot_expand_it(self):
        payload = {
            "app": "com.tencent.multimsg",
            "config": {"forward": 1},
            "meta": {
                "detail": {
                    "news": [
                        {"text": "甲：预览第一条"},
                        {"text": "乙：预览第二条"},
                    ]
                }
            },
        }
        event = make_event(
            Message([MessageSegment.json(json.dumps(payload, ensure_ascii=False))]),
            message_id=321,
        )
        bot = SimpleNamespace(
            call_api=AsyncMock(side_effect=RuntimeError("unsupported card")),
        )

        resolved = await resolve_incoming_message(bot, event)
        content = resolved.compose()

        self.assertEqual(resolved.plain_text, "")
        self.assertIn("[合并转发]", content)
        self.assertIn("甲：预览第一条", content)
        self.assertIn("乙：预览第二条", content)
        bot.call_api.assert_awaited_once_with(
            "get_forward_msg",
            message_id="321",
        )

    async def test_nested_forward_and_reply_are_expanded_within_budget(self):
        event = make_event(Message([MessageSegment.forward("outer")]))

        async def call_api(api, **kwargs):
            if api == "get_forward_msg" and kwargs == {"id": "outer"}:
                return {
                    "messages": [
                        {
                            "sender": {"nickname": "甲", "user_id": 1},
                            "content": [
                                {"type": "forward", "data": {"id": "inner"}},
                                {"type": "reply", "data": {"id": "88"}},
                            ],
                        }
                    ]
                }
            if api == "get_forward_msg" and kwargs == {"id": "inner"}:
                return {
                    "messages": [
                        {
                            "sender": {"nickname": "乙", "user_id": 2},
                            "content": "嵌套内容",
                        }
                    ]
                }
            if api == "get_msg" and kwargs == {"message_id": 88}:
                return {
                    "sender": {"nickname": "丙", "user_id": 3},
                    "message": [
                        {"type": "text", "data": {"text": "被回复内容"}},
                    ],
                }
            raise AssertionError((api, kwargs))

        bot = SimpleNamespace(call_api=AsyncMock(side_effect=call_api))

        content = (await resolve_incoming_message(bot, event)).compose()

        self.assertIn("[嵌套合并转发]", content)
        self.assertIn("乙: 嵌套内容", content)
        self.assertIn("[回复: 被回复内容]", content)
        self.assertEqual(bot.call_api.await_count, 3)

    async def test_forward_node_limit_is_enforced(self):
        event = make_event(Message([MessageSegment.forward("large-forward")]))
        nodes = [
            {
                "sender": {"nickname": f"成员{i}", "user_id": i},
                "content": f"第{i}条",
            }
            for i in range(40)
        ]
        bot = SimpleNamespace(
            call_api=AsyncMock(return_value={"messages": nodes}),
        )

        content = (await resolve_incoming_message(bot, event)).compose()

        self.assertIn("成员29", content)
        self.assertNotIn("成员30:", content)
        self.assertIn("后续转发节点已省略", content)

    async def test_pure_group_forward_is_written_to_memory(self):
        event = make_event(Message([MessageSegment.forward("memory-forward")]))
        event.sender = SimpleNamespace(card="转发者", nickname="转发者")
        event.to_me = False
        bot = SimpleNamespace(
            call_api=AsyncMock(
                return_value={
                    "messages": [
                        {
                            "sender": {"nickname": "甲", "user_id": 1},
                            "content": "需要进入记忆的内容",
                        }
                    ]
                }
            ),
            send=AsyncMock(),
        )

        with (
            patch.object(chat.config, "enable_group_repeat", False),
            patch.object(chat.config, "enable_sentiment_analysis", False),
            patch.object(chat.memory_manager, "add_message", AsyncMock(side_effect=stored_message)) as add_message,
            patch.object(chat.memory_manager, "enrich_message", AsyncMock(return_value=True)) as enriched,
            patch.object(
                chat.proactive_manager,
                "try_proactive",
                AsyncMock(),
            ) as try_proactive,
        ):
            await record_group_message(bot, event)

        add_message.assert_awaited_once()
        enriched.assert_awaited_once()
        self.assertIn("等待展开", add_message.await_args.args[2])
        saved_content = enriched.await_args.args[2]
        self.assertIn("[合并转发]", saved_content)
        self.assertIn("甲: 需要进入记忆的内容", saved_content)
        try_proactive.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
