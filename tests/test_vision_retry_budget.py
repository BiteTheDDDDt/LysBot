import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nonebot
nonebot.init()
from src.plugins.llm_chat import models, network_retry


class VisionRetryBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_wait_is_inside_total_budget_but_outside_single_attempt(self):
        manager = models.ModelManager()
        attempts = 0
        async def request():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=6)
            return models.ChatResponse('识图成功')
        async def adapter_call(*a, **kw):
            return await network_retry.request_with_retry(request, operation='vision')
        with patch.object(models.config, 'vision_request_timeout', 6), \
             patch.object(models.config, 'background_llm_total_timeout', 20), \
             patch.object(models.config, 'network_max_retries', 1), \
             patch.object(models.config, 'network_retry_delay', 0.03), \
             patch.object(manager, '_resolve_candidates', return_value=('openai:test', ['openai:test'])), \
             patch.object(manager, 'get_or_build_adapter', return_value=SimpleNamespace(chat_with_images=adapter_call)):
            result = await manager._chat_with_images_impl([], ['test'], session_id='_test_')
        self.assertEqual(str(result), '识图成功')
        self.assertEqual(attempts, 2)

    async def test_total_timeout_cancels_inflight_attempt_with_clear_reason(self):
        manager = models.ModelManager()
        stopped = asyncio.Event()
        async def stuck(*a, **kw):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        with patch.object(models.config, 'background_llm_total_timeout', 0.05), \
             patch.object(manager, '_chat_with_images_candidates', side_effect=stuck):
            with self.assertRaisesRegex(TimeoutError, '识图总预算耗尽'):
                await manager._chat_with_images_impl([], ['test'])
        self.assertTrue(stopped.is_set())

    async def test_inner_timeout_is_not_reported_as_total_timeout(self):
        manager = models.ModelManager()
        with patch.object(manager, '_chat_with_images_candidates', side_effect=TimeoutError('上游超时')):
            with self.assertRaisesRegex(TimeoutError, '^上游超时$'):
                await manager._chat_with_images_impl([], ['test'])

    async def test_stuck_primary_leaves_time_for_successful_fallback(self):
        manager = models.ModelManager()
        calls = []
        async def primary(*a, **kw):
            calls.append('primary')
            await asyncio.Event().wait()
        async def fallback(*a, **kw):
            calls.append('fallback')
            await asyncio.sleep(0.2)
            return models.ChatResponse('备用识图成功')
        adapters = {'openai:primary': SimpleNamespace(chat_with_images=primary),
                    'openai:fallback': SimpleNamespace(chat_with_images=fallback)}
        with patch.object(models.config, 'vision_request_timeout', 0.4), \
             patch.object(models.config, 'background_llm_total_timeout', 1.2), \
             patch.object(models.config, 'network_max_retries', 1), \
             patch.object(models.config, 'network_retry_delay', 0.3), \
             patch.object(manager, '_resolve_candidates', return_value=('openai:primary', list(adapters))), \
             patch.object(manager, 'get_or_build_adapter', side_effect=adapters.get):
            result = await manager._chat_with_images_impl([], ['test'])
        self.assertEqual(str(result), '备用识图成功')
        self.assertEqual(calls, ['primary', 'fallback'])
