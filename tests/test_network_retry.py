import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import nonebot
nonebot.init()

from src.plugins.llm_chat import network_retry as retry


class NetworkRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(retry.config, 'network_max_retries', 1))
        self.enterContext(patch.object(retry.config, 'network_retry_delay', 25))
        self.sleep = self.enterContext(patch.object(retry.asyncio, 'sleep', AsyncMock()))

    async def test_temporary_status_retries_same_request_once_after_waiting(self):
        request = AsyncMock(side_effect=[retry.ResponseError('HTTP 503', status_code=503), 'ok'])
        self.assertEqual(await retry.request_with_retry(request, operation='test'), 'ok')
        self.assertEqual(request.await_count, 2)
        self.sleep.assert_awaited_once_with(25)

    async def test_server_retry_after_takes_precedence(self):
        request = AsyncMock(side_effect=[retry.ResponseError('HTTP 429', status_code=429, headers={'Retry-After': '32'}), 'ok'])
        await retry.request_with_retry(request, operation='test')
        self.sleep.assert_awaited_once_with(32)

    async def test_expired_budget_does_not_start_wait_or_second_request(self):
        request = AsyncMock(side_effect=httpx.ReadTimeout('timeout'))
        with retry.retry_scope(29), self.assertRaises(httpx.ReadTimeout):
            await retry.request_with_retry(request, operation='test')
        request.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_retries_are_bounded_and_final_failure_propagates(self):
        error = httpx.ConnectError('offline')
        request = AsyncMock(side_effect=error)
        with self.assertRaises(httpx.ConnectError):
            await retry.request_with_retry(request, operation='test')
        self.assertEqual(request.await_count, 2)

    async def test_auth_quota_bad_arguments_and_invalid_output_do_not_retry(self):
        for error in [retry.ResponseError('invalid_api_key', status_code=401),
                      retry.ResponseError('insufficient_quota', status_code=429),
                      retry.ResponseError('bad request', status_code=400), ValueError('invalid JSON')]:
            request = AsyncMock(side_effect=error)
            with self.assertRaises(type(error)):
                await retry.request_with_retry(request, operation='test')
            request.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_submitted_media_read_timeout_or_503_is_not_replayed(self):
        for error in [httpx.ReadTimeout('unknown result'), retry.ResponseError('HTTP 503', status_code=503)]:
            request = AsyncMock(side_effect=error)
            with self.assertRaises(type(error)):
                await retry.request_with_retry(request, operation='image', replay_safe=False)
            request.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_media_connection_failure_can_retry_before_submission(self):
        request = AsyncMock(side_effect=[httpx.ConnectError('not connected'), 'ok'])
        self.assertEqual(await retry.request_with_retry(request, operation='image', replay_safe=False), 'ok')
        self.assertEqual(request.await_count, 2)

    async def test_cancellation_during_request_or_wait_stops_retry(self):
        request = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await retry.request_with_retry(request, operation='test')
        request.assert_awaited_once()
        request = AsyncMock(side_effect=httpx.ConnectError('offline'))
        self.sleep.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await retry.request_with_retry(request, operation='test')
        request.assert_awaited_once()

    async def test_dynamic_remaining_budget_is_respected(self):
        request = AsyncMock(side_effect=httpx.ReadTimeout('timeout'))
        with self.assertRaises(httpx.ReadTimeout):
            await retry.request_with_retry(request, operation='test', remaining=lambda: 10)
        request.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_retry_after_over_limit_is_not_shortened_to_early_retry(self):
        request = AsyncMock(side_effect=retry.ResponseError('HTTP 429', status_code=429, headers={'retry-after': '120'}))
        with self.assertRaises(retry.ResponseError):
            await retry.request_with_retry(request, operation='test')
        request.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_httpx_server_status_uses_response_status(self):
        response = httpx.Response(503, request=httpx.Request('POST', 'https://test.invalid'))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            self.assertEqual(retry.retry_delay(error), 25)

    async def test_stream_only_retries_rejected_connection_and_closes_it(self):
        attempts = 0
        def respond(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(429 if attempts == 1 else 200, text='data')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            async with retry.stream_with_retry(client, 'https://test.invalid', headers={}, body={}, operation='image') as response:
                self.assertEqual(await response.aread(), b'data')
            self.assertTrue(response.is_closed)
        self.assertEqual(attempts, 2)
        self.sleep.assert_awaited_once_with(25)

    async def test_accepted_stream_failure_is_not_replayed(self):
        attempts = 0
        def respond(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(200, text='data')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with self.assertRaises(httpx.ReadTimeout):
                async with retry.stream_with_retry(client, 'https://test.invalid', headers={}, body={}, operation='image'):
                    raise httpx.ReadTimeout('stream interrupted')
        self.assertEqual(attempts, 1)
        self.sleep.assert_not_awaited()

    async def test_local_step_deadline_is_not_replayed_as_a_network_fault(self):
        error = retry.StepTimeout('step budget exhausted after 30s')
        self.assertIsNone(retry.retry_delay(error))
        request = AsyncMock(side_effect=error)
        with self.assertRaises(retry.StepTimeout):
            await retry.request_with_retry(request, operation='test', remaining=lambda: 120)
        request.assert_awaited_once()
        self.sleep.assert_not_awaited()

    async def test_wrapped_timeout_keeps_diagnostic_reason(self):
        from src.plugins.llm_chat.diagnostics import api_failure, error_summary
        try:
            try:
                raise TimeoutError()
            except TimeoutError as cause:
                raise RuntimeError('API 请求出现错误: ') from cause
        except RuntimeError as error:
            self.assertEqual(api_failure(error, operation='test')['code'], 'timeout')
            self.assertIn('请求达到时间限制', error_summary(error))
