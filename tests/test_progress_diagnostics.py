import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import agent_runner, models, tools
from src.plugins.llm_chat.diagnostics import api_failure
from src.plugins.llm_chat.progress import report_progress


class ProgressDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def runner(self):
        r = agent_runner.AgentRunner()
        r.memory_path = None
        r.progress_callback = AsyncMock(return_value={"message_id": 1})
        r.progress_last_at -= 30
        r.evidence = {"h:0:messages:1": {"excerpt": "有人这样称呼该账号"}}
        return r

    def test_quota_error_retains_meaning_without_raw_secrets(self):
        e = RuntimeError("Error code: 429 - {'code':'usage_limit_reached','reset_seconds':123} "
                         "Authorization: Bearer secret-value https://private.example?api_key=secret-value")
        data = api_failure(e, operation="search_web")
        self.assertEqual(data["code"], "quota_exhausted")
        self.assertEqual(data["http_status"], 429)
        self.assertEqual(data["retry_after_seconds"], 123)
        self.assertFalse(data["retryable"])
        self.assertNotIn("secret-value", json.dumps(data))
        self.assertNotIn("private.example", json.dumps(data))

    async def test_fallback_request_receives_sanitized_failure(self):
        r = self.runner()
        backend = SimpleNamespace(model_name="test:model", _create_completion=AsyncMock(side_effect=[
            RuntimeError("Error code: 429 usage_limit_reached secret-value"), "ok",
        ]))
        args = {"messages": [{"role": "user", "content": "查询"}]}
        with self.assertRaises(RuntimeError):
            await r.step(backend, None, args)
        await r.step(backend, None, args)
        sent = json.dumps(backend._create_completion.await_args.args[1], ensure_ascii=False)
        self.assertIn("quota_exhausted", sent)
        self.assertNotIn("secret-value", sent)

    async def test_tool_failure_retains_diagnostic_details(self):
        with patch.dict(tools.TOOL_FUNCTIONS, {"search_web": AsyncMock(side_effect=RuntimeError("HTTP 503 private-secret"))}):
            data = json.loads(await tools.execute_tool("search_web", {"query": "test"}))
        self.assertEqual(data["error"]["http_status"], 503)
        self.assertEqual(data["error"]["code"], "upstream_unavailable")
        self.assertNotIn("private-secret", json.dumps(data))

    async def test_progress_is_rendered_as_chat_and_rate_limited(self):
        r = self.runner()
        with patch.object(models.model_manager, "chat", AsyncMock(return_value="找到相关称呼了，还在核对。")) as render:
            first = json.loads(await report_progress(r, "发现称呼线索", ["h:0:messages:1"]))
            second = json.loads(await report_progress(r, "继续核对", ["h:0:messages:1"]))
        self.assertEqual(first["status"], "sent")
        self.assertEqual(second["status"], "skipped")
        self.assertEqual(render.await_args.kwargs["role"], "chat")
        r.progress_callback.assert_awaited_once()
        self.assertEqual(r.progress_messages, ["找到相关称呼了，还在核对。"])

    async def test_no_evidence_and_unknown_source_never_send(self):
        r = self.runner()
        for refs in ([], ["h:0:messages:999"]):
            result = json.loads(await report_progress(r, "已找到答案", refs))
            self.assertNotEqual(result["status"], "sent")
        r.progress_callback.assert_not_awaited()

    async def test_uncertain_send_is_not_repeated(self):
        r = self.runner()
        r.progress_callback.side_effect = TimeoutError()
        with patch.object(models.model_manager, "chat", AsyncMock(return_value="正在核对称呼线索。")):
            await report_progress(r, "核对中", ["h:0:messages:1"])
            r.progress_last_at -= 30
            await report_progress(r, "核对中", ["h:0:messages:1"])
        r.progress_callback.assert_awaited_once()

    async def test_cancellation_propagates_without_send(self):
        r = self.runner()
        with (
            patch.object(models.model_manager, "chat", AsyncMock(side_effect=asyncio.CancelledError())),
            self.assertRaises(asyncio.CancelledError),
        ):
            await report_progress(r, "核对中", ["h:0:messages:1"])
        r.progress_callback.assert_not_awaited()
