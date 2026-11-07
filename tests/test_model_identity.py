import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import model_identity


class ModelIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from src.plugins.llm_chat.config import config
        switch = patch.object(config, "enable_model_card", True)
        switch.start()
        self.addCleanup(switch.stop)
        model_identity._group_last_cards.clear()
        model_identity._group_base_cards.clear()
        model_identity._proxy_owners.clear()

    async def test_group_card_includes_provider_and_model(self):
        bot = SimpleNamespace(
            self_id="42",
            get_group_member_info=AsyncMock(
                return_value={"card": "磷叶石（Google:gemini-old）", "nickname": "磷叶石"}
            ),
            set_group_card=AsyncMock(),
        )
        response = SimpleNamespace(model_spec="cliproxyapi:gpt-5.6-luna")

        await model_identity.update_group_card_for_response(bot, 99, response)

        bot.set_group_card.assert_awaited_once_with(
            group_id=99,
            user_id=42,
            card="磷叶石（CLIProxyAPI:gpt-5.6-luna）",
        )

    def test_preserves_non_model_parentheses(self):
        self.assertEqual(model_identity._strip_existing_model_label('磷叶石（休息中）'), '磷叶石（休息中）')
        self.assertEqual(model_identity._strip_existing_model_label('磷叶石（休息中）（Google:gemini-old）'), '磷叶石（休息中）')

    async def test_briefing_platform_labels_survive_empty_renderer_cache(self):
        from src.plugins.llm_chat.card_templates import usage_models
        stats = {'by_spec': [{'spec': 'cliproxyapi:custom'}, {'spec': 'codexproxy:custom'}, {'spec': 'cliproxyapi:missing'}]}
        model_identity._proxy_owners['custom'] = 'antigravity'
        with patch.object(model_identity, 'refresh_proxy_model_identity', AsyncMock()):
            await model_identity.label_usage_models(stats)
        model_identity._proxy_owners.clear()
        labels = [row[0] for row in usage_models(stats)]
        self.assertEqual(labels[:2], ['CLIProxyAPI/Antigravity:custom'] * 2)
        self.assertIn('上游未确认', labels[2])

    async def test_only_new_confirmed_text_delivery_updates_card(self):
        from src.plugins.llm_chat.agent_runner import deliver_reply
        response = SimpleNamespace(model_spec='test:actual')
        bot = SimpleNamespace(self_id='42')
        with patch.object(model_identity, 'update_group_card_for_response', AsyncMock()) as update:
            for receipt in [None, {}, {'status': 'ok'}]:
                await deliver_reply(response, '回答', AsyncMock(return_value=receipt), bot=bot, group_id=7)
            with self.assertRaises(RuntimeError):
                await deliver_reply(response, '回答', AsyncMock(side_effect=RuntimeError('failed')), bot=bot, group_id=7)
            response.agent_run = SimpleNamespace(deliver=AsyncMock(return_value={'message_id': 9}))
            await deliver_reply(response, '旧回执', AsyncMock(), bot=bot, group_id=7)
            update.assert_not_awaited()
            del response.agent_run
            send = AsyncMock(return_value={'message_id': 10})
            await deliver_reply(response, '回答', send, bot=bot, group_id=7)
            update.assert_awaited_once_with(bot, 7, response)

    async def test_parallel_delivery_keeps_card_order(self):
        from src.plugins.llm_chat.agent_runner import deliver_reply
        bot = SimpleNamespace(self_id='42')
        events = []
        async def send(name):
            events.append('send:'+name)
            await asyncio.sleep(0)
            return {'message_id': name}
        async def update(bot, group_id, response):
            await asyncio.sleep(0)
            events.append('card:'+response.model_spec)
        with patch.object(model_identity, 'update_group_card_for_response', side_effect=update):
            await asyncio.gather(*[
                deliver_reply(SimpleNamespace(model_spec=name), name, lambda name=name: send(name), bot=bot, group_id=7)
                for name in ['first', 'second']
            ])
        self.assertEqual(events, ['send:first', 'card:first', 'send:second', 'card:second'])

    async def test_refresh_preserves_external_rename_and_separates_bot_accounts(self):
        response = SimpleNamespace(model_spec='google:gemini-test')
        bot = SimpleNamespace(self_id='42', get_group_member_info=AsyncMock(return_value={'card': '新名字（休息中）'}), set_group_card=AsyncMock())
        model_identity._group_last_cards[('42', 7)] = ('Google:gemini-test', -100)
        with patch.object(model_identity, 'refresh_proxy_model_identity', AsyncMock()):
            await model_identity.update_group_card_for_response(bot, 7, response)
            bot.set_group_card.assert_awaited_once_with(group_id=7, user_id=42, card='新名字（休息中）（Google:gemini-test）')
            await model_identity.update_group_card_for_response(bot, 7, response)
            self.assertEqual(bot.set_group_card.await_count, 1)
            bot.self_id = '43'
            await model_identity.update_group_card_for_response(bot, 7, response)
            self.assertEqual(bot.set_group_card.await_count, 2)

    async def test_card_failure_does_not_discard_receipt_or_poison_cache(self):
        from src.plugins.llm_chat.agent_runner import deliver_reply
        bot = SimpleNamespace(self_id='42', get_group_member_info=AsyncMock(side_effect=TimeoutError()), set_group_card=AsyncMock())
        with patch.object(model_identity, 'refresh_proxy_model_identity', AsyncMock()):
            receipt = await deliver_reply(SimpleNamespace(model_spec='google:gemini-test'), '回答', AsyncMock(return_value={'message_id': 9}), bot=bot, group_id=7)
        self.assertEqual(receipt, {'message_id': 9})
        self.assertNotIn(('42', 7), model_identity._group_last_cards)

    def test_proxy_platform_comes_from_catalog_not_model_family(self):
        model_identity._proxy_owners.update({"claude-sonnet-4-6": "antigravity", "gpt-6-luna": "openai", "grok-4.7": "xai"})
        self.assertEqual(model_identity.format_model_spec_label("cliproxyapi:claude-sonnet-4-6"), "CLIProxyAPI/Antigravity:claude-sonnet-4-6")
        self.assertEqual(model_identity.format_model_spec_label("codexproxy:gpt-6-luna"), "CLIProxyAPI/OpenAI:gpt-6-luna")
        self.assertEqual(model_identity.format_model_spec_label("cliproxyapi:grok-4.7"), "CLIProxyAPI/xAI:grok-4.7")
        self.assertEqual(model_identity.format_model_spec_label("cliproxyapi:unknown"), "CLIProxyAPI:unknown")
        self.assertEqual(model_identity.format_model_spec_label("openrouter:vendor/model"), "OpenRouter:model")

    async def test_catalog_refresh_is_cached_without_inference(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [{"id": "custom", "owned_by": "antigravity"}]})
        client = SimpleNamespace(get=AsyncMock(return_value=response))
        manager = AsyncMock()
        manager.__aenter__.return_value = client
        with patch("httpx.AsyncClient", return_value=manager), patch.object(model_identity, "_proxy_refresh_at", 0):
            await model_identity.refresh_proxy_model_identity()
            await model_identity.refresh_proxy_model_identity()
        client.get.assert_awaited_once()
        self.assertTrue(client.get.call_args.args[0].endswith("/models"))
        self.assertEqual(model_identity.format_model_spec_label("cliproxyapi:custom"), "CLIProxyAPI/Antigravity:custom")


if __name__ == "__main__":
    unittest.main()
