import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import proactive, tools
from src.plugins.llm_chat.planner import PlannerDecision, _normalize_decision


class ProactiveMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_meme_only_skips_generation_and_records_only_success(self):
        from src.plugins.llm_chat.planner import ReplyPart
        for picked, sent in ((True, True), (True, False), (False, False)):
            manager = proactive.ProactiveManager()
            meme = {'id': 1, 'topic': 'funny', 'description': '笑哭'}
            decision = PlannerDecision(parts=[ReplyPart('meme', 'funny')])
            bot = SimpleNamespace(self_id='999', send_group_msg=AsyncMock())
            with patch.object(proactive.config, 'enable_proactive', True), patch.object(proactive.config, 'enable_sentiment_analysis', False), patch.object(proactive.config, 'proactive_msg_threshold', 1), patch.object(manager, 'is_on_planner_cooldown', return_value=False), patch.object(proactive, 'plan_chat_action', AsyncMock(return_value=decision)), patch.object(proactive, 'execute_planned_side_effect', AsyncMock()), patch.object(proactive, 'prepare_planned_meme', AsyncMock(return_value=meme if picked else None)), patch.object(manager, 'generate_proactive_response', AsyncMock()) as generate, patch.object(tools, 'send_picked_meme', AsyncMock(return_value='已发送' if sent else '发送失败')) as send, patch.object(tools, 'get_last_sent_meme', return_value=meme), patch.object(proactive.memory_manager, 'add_message', AsyncMock()) as save:
                result = await manager.try_proactive(bot, 99, '夫目前犯的背德感', user_id=42)
            self.assertEqual(result, sent)
            generate.assert_not_awaited()
            self.assertEqual(send.await_count, int(picked))
            self.assertEqual(save.await_count, int(sent))
            self.assertEqual(manager.is_on_cooldown(99), sent)
            bot.send_group_msg.assert_not_awaited()

    async def test_planner_skipping_participation_does_not_retry(self):
        from src.plugins.llm_chat.models import ChatResponse
        manager = proactive.ProactiveManager()
        response = ChatResponse('')
        response.agent_run = SimpleNamespace(should_reply=False)
        with patch.object(proactive.config, 'enable_proactive', True), patch.object(proactive.config, 'enable_sentiment_analysis', False), patch.object(proactive.config, 'proactive_msg_threshold', 1), patch.object(manager, 'is_on_planner_cooldown', return_value=False), patch.object(proactive, 'plan_chat_action', AsyncMock(return_value=PlannerDecision())), patch.object(proactive, 'execute_planned_side_effect', AsyncMock()), patch.object(proactive, 'prepare_planned_meme', AsyncMock(return_value=None)), patch.object(manager, 'generate_proactive_response', AsyncMock(return_value=response)) as generate:
            self.assertFalse(await manager.try_proactive(SimpleNamespace(self_id='999'), 99, '下半张没转发', user_id=42))
        generate.assert_awaited_once()
        self.assertFalse(manager.is_on_cooldown(99))

    async def test_generation_passes_actual_trigger_and_proactive_policy(self):
        from src.plugins.llm_chat.models import Message
        from src.plugins.llm_chat.reply_policy import PROACTIVE_REPLY
        trigger = "[引用回复] 克洛耶 [当前消息] 这是什么呢"
        manager = proactive.ProactiveManager()
        with (
            patch.object(proactive.memory_manager, "get_history", AsyncMock(return_value=[])),
            patch.object(proactive.memory_manager, "get_session_model", return_value=None),
            patch.object(proactive, "build_context_messages", AsyncMock(return_value=[Message("user", trigger)])) as build,
            patch.object(proactive.model_manager, "run_turn", AsyncMock(return_value="")) as run,
        ):
            await manager.generate_proactive_response(99, recent_context="", trigger_message=trigger)
        self.assertEqual(build.call_args.kwargs["user_content"], trigger)
        self.assertEqual(run.call_args.kwargs["reply_policy"], PROACTIVE_REPLY)

    async def test_poke_target_is_deduplicated_across_reasons_and_sender_alias(self):
        bot = SimpleNamespace(call_api=AsyncMock(return_value={"message_id": 1}))
        tools.set_poke_context(group_id=99, user_id=42, bot=bot)
        first = await tools.send_poke("sender", reason="接梗")
        second = await tools.send_poke("42", reason="回戳")
        self.assertEqual(first, second)
        bot.call_api.assert_awaited_once()
        tools.set_poke_context(group_id=99, user_id=42, bot=bot)
        with patch.object(tools, "_send_poke_once", AsyncMock(side_effect=RuntimeError("unknown"))) as send:
            with self.assertRaises(RuntimeError):
                await tools.send_poke("sender")
            self.assertIn("结果未知", await tools.send_poke("42", reason="retry"))
            send.assert_awaited_once()
    async def test_poke_only_delivery_and_cooldown_require_success(self):
        for success in (True, False):
            with self.subTest(success=success):
                manager = proactive.ProactiveManager()
                decision = _normalize_decision({'action': 'poke', 'parts': [], 'poke_target_id': '999'}, 'silent', 'proactive')
                self.assertEqual(decision.parts, [])
                bot = SimpleNamespace(self_id='999', send_group_msg=AsyncMock())
                result_text = '戳一戳已发送给 42' if success else '戳一戳失败：接口不可用'
                with patch.object(proactive.config, 'enable_proactive', True), patch.object(proactive.config, 'enable_sentiment_analysis', False), patch.object(proactive.config, 'proactive_msg_threshold', 1), patch.object(proactive.config, 'proactive_cooldown', 60), patch.object(manager, 'is_on_planner_cooldown', return_value=False), patch.object(proactive, 'plan_chat_action', AsyncMock(return_value=decision)), patch.object(tools, 'send_poke', AsyncMock(return_value=result_text)) as poke, patch.object(manager, 'generate_proactive_response', AsyncMock()) as generate, patch.object(proactive.memory_manager, 'add_message', AsyncMock()) as save:
                    result = await manager.try_proactive(bot, 99, '又被自己蠢笑了', user_id=42)
                    self.assertEqual(result, success)
                    poke.assert_awaited_once_with('sender', reason='')
                    generate.assert_not_awaited()
                    bot.send_group_msg.assert_not_awaited()
                    self.assertEqual(manager.is_on_cooldown(99), success)
                    if success:
                        save.assert_awaited_once_with(0, 'assistant', '[已戳一戳 QQ:42]', 99)
                        self.assertFalse(await manager.try_proactive(bot, 99, '还在笑', user_id=42))
                        self.assertEqual(poke.await_count, 1)
                    else:
                        save.assert_not_awaited()
                    self.assertIsNone(tools._poke_context.get())

    def tearDown(self):
        tools.clear_poke_context()
        tools.clear_meme_context()
        tools.clear_voice_context()
        tools.clear_image_context()

    async def test_proactive_image_gets_send_context_and_finishes_without_retry(self):
        manager = proactive.ProactiveManager()
        bot = SimpleNamespace(self_id='999', send_group_msg=AsyncMock())

        async def generate_media(*args, **kwargs):
            context = tools._image_context.get()
            self.assertIsNotNone(context)
            self.assertIs(context['bot'], bot)
            self.assertEqual(context['group_id'], 99)
            self.assertEqual(context['user_id'], 42)
            context['sent_in_round'] = True
            context['last_sent'] = {'prompt': '主动发言生成的群聊合影', 'description': '四位群友的动漫群像海报。', 'model': 'cliproxyapi:gpt-image-2', 'size': '1536x1024'}
            tools._image_context.set(context)
            return '喏，群友合影给你们整出来了。接下来再详细解释这张图的构思。'
        add_message = AsyncMock()
        generate_mock = AsyncMock(side_effect=generate_media)
        with patch.object(proactive.config, 'enable_proactive', True), patch.object(proactive.config, 'enable_sentiment_analysis', False), patch.object(proactive.config, 'proactive_msg_threshold', 1), patch.object(manager, 'is_on_cooldown', return_value=False), patch.object(manager, 'is_on_planner_cooldown', return_value=False), patch.object(proactive, 'plan_chat_action', AsyncMock(return_value=PlannerDecision(action='reply'))), patch.object(proactive, 'execute_planned_side_effect', AsyncMock()), patch.object(proactive, 'prepare_planned_meme', AsyncMock(return_value=None)), patch.object(manager, 'generate_proactive_response', generate_mock), patch.object(proactive.memory_manager, 'add_message', add_message):
            result = await manager.try_proactive(bot, group_id=99, message='生成一下你想象中的群友图片', user_id=42)
        self.assertTrue(result)
        self.assertEqual(generate_mock.await_count, 1)
        add_message.assert_awaited_once()
        memory_text = add_message.await_args.args[2]
        self.assertIn('主动发言生成的群聊合影', memory_text)
        self.assertIn('四位群友的动漫群像海报', memory_text)
        self.assertIn('喏，群友合影给你们整出来了', memory_text)
        self.assertIn('详细解释', memory_text)
        bot.send_group_msg.assert_awaited_once_with(group_id=99, message='喏，群友合影给你们整出来了。接下来再详细解释这张图的构思')
        self.assertIsNone(tools._image_context.get())
        self.assertIsNone(tools._voice_context.get())


if __name__ == "__main__":
    unittest.main()
