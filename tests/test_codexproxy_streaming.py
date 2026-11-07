import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import nonebot
from openai import AsyncOpenAI

nonebot.init()

from src.plugins.llm_chat import models, tools


def chunk(delta=None, *, finish=None, usage=None):
    return {
        "id": "test-stream", "object": "chat.completion.chunk", "created": 1,
        "model": "gpt-oss-120b-medium",
        "choices": [] if usage else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        "usage": usage,
    }


def text_chunks(text="OK"):
    return [chunk({"role": "assistant", "content": None}),
            chunk({"role": "assistant", "reasoning_content": "internal reasoning"}),
            chunk({"role": "assistant", "content": text[:1]}), chunk({"role": "assistant", "content": text[1:]}, finish="stop"),
            chunk(usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})]


def tool_chunks(name, arguments):
    return [chunk({"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call-test", "type": "function",
                 "function": {"name": name[:4], "arguments": arguments[:8]}}]}),
            chunk({"role": "assistant", "tool_calls": [{"index": 0, "function": {"name": name[4:], "arguments": arguments[8:]}}]}, finish="tool_calls"),
            chunk(usage={"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8})]


class SSEBody(httpx.AsyncByteStream):
    def __init__(self, chunks, *, error=None, done=True):
        self.chunks = chunks
        self.error = error
        self.done = done
        self.closed = False

    async def __aiter__(self):
        for payload in self.chunks:
            yield ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
        if self.error:
            raise self.error
        if self.done:
            yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


class CLIProxyAPIStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def make_adapter(self, bodies, *, spec="cliproxyapi:gpt-oss-120b-medium"):
        requests = []

        async def handle(request):
            requests.append(json.loads(request.content))
            body = bodies[len(requests) - 1]
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

        adapter = models.OpenAICompatibleAdapter(model_name=spec, api_key="test-key",
                                                  base_url="https://proxy.test/v1", model_id=spec.split(":", 1)[1])
        await adapter.client.close()
        adapter.client = AsyncOpenAI(api_key="test-key", base_url="https://proxy.test/v1",
                                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), max_retries=0)
        self.addAsyncCleanup(adapter.client.close)
        return adapter, requests

    async def test_text_and_final_usage_are_accumulated_without_reasoning_leak(self):
        body = SSEBody(text_chunks("你好"))
        adapter, requests = await self.make_adapter([body])
        result = await adapter.chat([models.Message(role="user", content="hello")])
        self.assertEqual(str(result), "你好")
        self.assertEqual(result.usage["total_tokens"], 13)
        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[0]["stream_options"], {"include_usage": True})
        self.assertTrue(body.closed)

    async def test_fragmented_tool_call_then_followup_use_streaming(self):
        bodies = [SSEBody(tool_chunks("search_web", '{"query":"test query"}')), SSEBody(text_chunks("查到了"))]
        adapter, requests = await self.make_adapter(bodies)
        execute = AsyncMock(return_value="synthetic search result")
        with patch.object(tools, "execute_tool", execute):
            result = await adapter.chat([models.Message(role="user", content="search")], tools_enabled=True)
        execute.assert_awaited_once_with("search_web", {"query": "test query"})
        self.assertEqual(str(result), "查到了")
        self.assertEqual(result.usage["total_tokens"], 21)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request["stream"] for request in requests))
        self.assertTrue(all(body.closed for body in bodies))
        self.assertTrue(any(message.get("tool_call_id") == "call-test" for message in requests[1]["messages"]))
        assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call-test")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "search_web")

    async def test_forced_media_tool_and_tool_free_final_request_are_streamed(self):
        bodies = [SSEBody(text_chunks("等一下")), SSEBody(tool_chunks("send_voice", '{"text":"你好"}')),
                  SSEBody(text_chunks("晚安"))]
        adapter, requests = await self.make_adapter(bodies)
        execute = AsyncMock(return_value='{"status":"sent","media_type":"voice"}')
        with patch.object(tools, "execute_tool", execute), \
             patch.object(tools, "voice_was_sent_this_round", return_value=True), \
             patch.object(tools, "image_was_sent_this_round", return_value=False):
            result = await adapter.chat([models.Message(role="user", content="say hello")],
                                        tools_enabled=True, required_tool="send_voice")
        self.assertEqual(str(result), "晚安")
        execute.assert_awaited_once_with("send_voice", {"text": "你好"})
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(request["stream"] for request in requests))
        self.assertEqual(requests[1]["tool_choice"]["function"]["name"], "send_voice")
        self.assertNotIn("tools", requests[2])

    async def test_validation_correction_stays_streamed_and_accumulates_usage(self):
        bodies = [SSEBody(tool_chunks("send_voice", '{"text":"too long"}')),
                  SSEBody(tool_chunks("send_voice", '{"text":"你好"}'))]
        adapter, requests = await self.make_adapter(bodies)
        sent = False

        async def execute(name, args):
            nonlocal sent
            if args["text"] == "too long":
                return tools.MediaValidationFailure("send_voice 失败：朗读内容超过上限")
            sent = True
            return '{"status":"sent"}'

        mocked = AsyncMock(side_effect=execute)
        with patch.object(tools, "execute_tool", mocked), \
             patch.object(tools, "voice_was_sent_this_round", side_effect=lambda: sent), \
             patch.object(tools, "image_was_sent_this_round", return_value=False):
            result = await adapter.chat([models.Message(role="user", content="say hello")],
                                        tools_enabled=True, required_tool="send_voice", media_followup=False)
        self.assertEqual(str(result), "")
        self.assertEqual(mocked.await_count, 2)
        self.assertEqual(result.usage["total_tokens"], 16)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request["stream"] for request in requests))
        self.assertEqual(requests[1]["tool_choice"]["function"]["name"], "send_voice")
        self.assertEqual([tool["function"]["name"] for tool in requests[1]["tools"]], ["send_voice"])
        self.assertTrue(all(body.closed for body in bodies))

    async def test_media_generation_failure_stops_without_another_stream(self):
        body = SSEBody(tool_chunks("send_voice", '{"text":"你好"}'))
        adapter, requests = await self.make_adapter([body])
        execute = AsyncMock(return_value="send_voice 生成失败：上游超时")
        with patch.object(tools, "execute_tool", execute), \
             patch.object(tools, "voice_was_sent_this_round", return_value=False), \
             patch.object(tools, "image_was_sent_this_round", return_value=False):
            result = await adapter.chat([models.Message(role="user", content="say hello")],
                                        tools_enabled=True, required_tool="send_voice")
        self.assertEqual(str(result), "语音生成或发送失败，请稍后再试。")
        self.assertEqual(len(requests), 1)
        execute.assert_awaited_once()
        self.assertTrue(body.closed)

    async def test_stream_failure_or_cancellation_closes_connection_without_partial_reply(self):
        for error in [httpx.ReadError("stream disconnected"), asyncio.CancelledError()]:
            body = SSEBody([chunk({"role": "assistant", "content": "partial"})], error=error)
            adapter, _ = await self.make_adapter([body])
            with self.assertRaises(asyncio.CancelledError if isinstance(error, asyncio.CancelledError) else RuntimeError):
                await adapter.chat([models.Message(role="user", content="hello")])
            self.assertTrue(body.closed)

    async def test_unfinished_or_reasoning_only_stream_is_not_a_reply(self):
        for payloads in [[chunk({"role": "assistant", "content": "partial"})],
                         [chunk({"role": "assistant", "reasoning_content": "private reasoning"}, finish="stop")]]:
            body = SSEBody(payloads)
            adapter, _ = await self.make_adapter([body])
            with self.assertRaises(RuntimeError):
                await adapter.chat([models.Message(role="user", content="hello")])
            self.assertTrue(body.closed)

    async def test_incomplete_tool_call_is_never_executed(self):
        body = SSEBody(tool_chunks("send_poke", '{"target_id":"sender"}')[:1], done=False)
        adapter, _ = await self.make_adapter([body])
        execute = AsyncMock()
        with patch.object(tools, "execute_tool", execute), self.assertRaises(RuntimeError):
            await adapter.chat([models.Message(role="user", content="poke")], tools_enabled=True)
        execute.assert_not_awaited()
        self.assertTrue(body.closed)

    async def test_repeated_tool_metadata_is_not_concatenated(self):
        payloads = tool_chunks("send_poke", '{"target_id":"sender"}')
        payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] = "send_poke"
        part = payloads[1]["choices"][0]["delta"]["tool_calls"][0]
        part.update(id="call-test", type="function")
        part["function"]["name"] = "send_poke"
        adapter, requests = await self.make_adapter([SSEBody(payloads), SSEBody(text_chunks())])
        execute = AsyncMock(return_value="synthetic result")
        with patch.object(tools, "execute_tool", execute):
            await adapter.chat([models.Message(role="user", content="poke")], tools_enabled=True)
        execute.assert_awaited_once_with("send_poke", {"target_id": "sender"})

    async def test_finished_stream_with_invalid_arguments_is_never_executed(self):
        for arguments in ['{"target_id":', '[]']:
            adapter, _ = await self.make_adapter([SSEBody(tool_chunks("send_poke", arguments))])
            execute = AsyncMock()
            with patch.object(tools, "execute_tool", execute), self.assertRaises(RuntimeError):
                await adapter.chat([models.Message(role="user", content="poke")], tools_enabled=True)
            execute.assert_not_awaited()

    async def test_ignored_tool_call_does_not_expose_reasoning_as_final_reply(self):
        for tools_enabled in [False, True]:
            payloads = tool_chunks("search_web", '{"query":"test"}')
            payloads[0]["choices"][0]["delta"]["reasoning_content"] = "private reasoning"
            bodies = [SSEBody(payloads) for _ in range(2 if tools_enabled else 1)]
            adapter, _ = await self.make_adapter(bodies)
            execute = AsyncMock(return_value="synthetic result")
            with patch.object(tools, "execute_tool", execute), \
                 patch.object(models.config, "llm_max_tool_rounds", 1):
                # An unfinished turn must fall back; its reasoning is neither
                # a final answer nor evidence of task completion.
                with self.assertRaisesRegex(RuntimeError, "最终答复"):
                    await adapter.chat([models.Message(role="user", content="hello")],
                                       tools_enabled=tools_enabled)
            self.assertEqual(execute.await_count, 1 if tools_enabled else 0)

    async def test_other_model_routes_keep_non_streaming_requests(self):
        for spec in ["cliproxyapi:gpt-5.6-sol", "openrouter:gpt-oss-120b-medium"]:
            adapter, _ = await self.make_adapter([], spec=spec)
            response = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="OK", tool_calls=[]), finish_reason="stop")], usage=None)
            with patch.object(adapter.client.chat.completions, "create", AsyncMock(return_value=response)) as create:
                result = await adapter.chat([models.Message(role="user", content="hello")])
            self.assertEqual(str(result), "OK")
            self.assertFalse(create.await_args.kwargs["stream"])
            self.assertNotIn("stream_options", create.await_args.kwargs)


if __name__ == "__main__":
    unittest.main()
