import asyncio
import copy
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import nonebot

nonebot.init()

from model_env import TEST_MODEL_ENV

from src.plugins.llm_chat import models, providers, roles, web_search
from src.plugins.llm_chat import native_search as native
from src.plugins.llm_chat import news_broadcast as news
from src.plugins.llm_chat.agent_runner import AgentRunner, current_run
from src.plugins.llm_chat.config import config

GPT = "cliproxyapi:gpt-5.6-luna"
URL = "https://example.com/news"


def completed():
    return {
        "status": "completed",
        "output": [
            {"type": "web_search_call", "status": "completed", "action": {"type": "search"}},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "A cited update.",
                        "annotations": [{"type": "url_citation", "url": URL, "title": "Actual title"}],
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    }


def sse(data):
    return "data: " + json.dumps(data) + "\n\n"


class SearchCapabilityTests(unittest.TestCase):
    def test_known_gpt_families_and_snapshots_are_allowed(self):
        for provider in ("openai", "cliproxyapi"):
            for model in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.5", "gpt-6-astra", "gpt-6-luna"):
                for suffix in ("", "-2026-09-01"):
                    self.assertEqual(providers.web_search_backend(f"{provider}:{model}{suffix}"), "openai_responses")

    def test_proxy_does_not_implicitly_enable_every_model(self):
        for spec in (
            "cliproxyapi:gemini-3.8-flash-high",
            "cliproxyapi:grok-4.6",
            "cliproxyapi:gpt-image-2",
            "openai:ordinary-text",
            "cliproxyapi:gpt-5.6-luna-fake",
            "cliproxyapi:gpt-6-luna-fake",
            "openrouter:gpt-5.6-luna",
        ):
            self.assertFalse(providers.supports_web_search(spec))
        self.assertTrue(providers.supports_web_search("dashscope:test-search"))

    def test_search_chain_accepts_gpt_without_dashscope_parameters(self):
        with patch.dict(os.environ, {**TEST_MODEL_ENV, "MODEL_SEARCH": GPT + " | dashscope:test-search"}, clear=True):
            self.assertEqual(roles.validate_all(), [])
        extra = models.model_manager._extra_for_role("search")
        self.assertIsNone(extra(GPT))
        self.assertTrue(extra("dashscope:test-search")["enable_search"])

    def test_gpt6_luna_search_and_vision_pass_startup_validation(self):
        with patch.dict(os.environ, {**TEST_MODEL_ENV,
                                    "MODEL_SEARCH": "cliproxyapi:gpt-6-luna",
                                    "MODEL_VISION": "cliproxyapi:gpt-6-luna"}, clear=True):
            self.assertEqual(roles.validate_all(), [])


class SourceParsingTests(unittest.TestCase):
    def test_source_metadata_preserves_actual_publication_date_and_snippet(self):
        data = completed()
        data["output"][0]["action"]["sources"] = [{
            "url": URL, "published_time": "2026-10-20T11:30:00Z", "snippet": "Official announcement",
        }]
        parsed = native.parse_search_response(data)
        source = parsed["sources"][0]
        self.assertEqual(source["published_at"], "2026-10-20")
        self.assertEqual(source["date_evidence"], "search_provider")
        self.assertEqual(source["excerpt"], "Official announcement")
        self.assertEqual(source["title"], "Actual title")

    def test_annotations_work_when_proxy_omits_action_sources(self):
        parsed = native.parse_search_response(completed())
        self.assertEqual(parsed["sources"][0]["url"], URL)
        self.assertEqual(parsed["sources"][0]["published_at"], "")
        self.assertEqual(parsed["usage"], {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14})
        self.assertIsNone(parsed["effective_freshness_days"])

    def test_sources_are_deduplicated_and_citation_titles_upgrade_bare_urls(self):
        data = completed()
        data["output"][0]["action"]["sources"] = [{"type": "url", "url": URL}]
        parsed = native.parse_search_response(data)
        self.assertEqual(len(parsed["sources"]), 1)
        self.assertEqual(parsed["sources"][0]["title"], "Actual title")

    def test_action_sources_are_usable_without_output_text(self):
        data = completed()
        data["output"] = data["output"][:1]
        data["output"][0]["action"]["sources"] = [{"url": URL}]
        self.assertEqual(native.parse_search_response(data)["sources"][0]["title"], "example.com")

    def test_text_urls_are_not_promoted_to_sources(self):
        data = completed()
        data["output"][1]["content"][0].update(text="News 2026-09-15 " + URL, annotations=[])
        with self.assertRaisesRegex(RuntimeError, "no verifiable sources"):
            native.parse_search_response(data)

    def test_actual_completed_search_is_required(self):
        for kind in ("missing", "in_progress", "incomplete"):
            data = completed()
            if kind == "missing":
                data["output"].pop(0)
            elif kind == "incomplete":
                data["status"] = kind
            else:
                data["output"][0]["status"] = kind
            with self.assertRaises(RuntimeError):
                native.parse_search_response(data)

    def test_private_and_non_http_sources_are_rejected(self):
        for url in (
            "file:///secret",
            "http://localhost/a",
            "http://127.0.0.1/a",
            "http://10.0.0.1/a",
            "https://example.internal/a",
            "https://user:pass@example.com",
            "http://example.com:8000/a",
        ):
            data = completed()
            data["output"][1]["content"][0]["annotations"][0]["url"] = url
            with self.assertRaisesRegex(RuntimeError, "no verifiable sources"):
                native.parse_search_response(data)


class NativeTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_is_sent_with_search_tool_without_entering_text_prompt(self):
        image = "data:image/jpeg;base64,aW1hZ2U="
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual(body["input"][0]["content"][1],
                             {"type": "input_image", "image_url": image})
            self.assertEqual(body["tools"], [{"type": "web_search"}])
            return httpx.Response(200, json=completed())
        await self.request(handle, image_data_urls=[image])

    async def test_image_is_not_silently_discarded_by_text_only_backend(self):
        with self.assertRaisesRegex(ValueError, "does_not_support_image_input"):
            await native.native_search("dashscope:test-search", "who", image_data_urls=["data:image/jpeg;base64,eA=="])

    def setUp(self):
        # Avoid fire-and-forget usage DB writes leaking across isolated event loops.
        self.log_patch = patch.object(native, "finish_call")
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    async def request(self, handler, *, spec=GPT, expected_timeout=None, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(native.httpx, "AsyncClient", return_value=client) as create:
            result = await native.native_search(spec, "Public query", **kwargs)
        if expected_timeout is not None:
            timeout = create.call_args.kwargs["timeout"]
            self.assertEqual(timeout.read if isinstance(timeout, httpx.Timeout) else timeout, expected_timeout)
        return result

    async def test_real_responses_request_and_terminal_sse(self):
        def handle(request):
            self.assertTrue(str(request.url).endswith("/v1/responses"))
            body = json.loads(request.content)
            self.assertEqual(body["model"], "gpt-5.6-luna")
            self.assertEqual(body["input"][0]["content"][0], {"type": "input_text", "text": "Public query"})
            self.assertEqual(body["tools"], [{"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}])
            self.assertEqual(body["tool_choice"], "required")
            self.assertTrue(body["stream"])
            self.assertFalse(body["store"])
            self.assertEqual(body["include"], ["web_search_call.action.sources"])
            self.assertNotIn("enable_search", body)
            events = sse({"type": "response.output_text.delta", "delta": "Untrusted partial"})
            events += sse({"type": "response.completed", "response": completed()})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=events)

        result = await self.request(handle, domains=["example.com"])
        self.assertEqual(result["content"], "A cited update.")
        self.assertEqual(result["spec"], GPT)

    async def test_non_streaming_json_compatibility(self):
        result = await self.request(lambda request: httpx.Response(200, json=completed()))
        self.assertEqual(result["usage"]["total_tokens"], 14)

    async def test_disconnect_and_done_without_terminal_response_fail(self):
        for ending in ("", "data: [DONE]\n\n"):
            with self.assertRaisesRegex(RuntimeError, "before response.completed"):
                await self.request(
                    lambda r, ending=ending: httpx.Response(
                        200, text=sse({"type": "response.output_text.delta", "delta": URL}) + ending
                    )
                )

    async def test_failed_and_incomplete_streams_fail(self):
        for kind in ("response.failed", "response.incomplete", "error"):
            with self.assertRaisesRegex(RuntimeError, kind):
                await self.request(
                    lambda r, kind=kind: httpx.Response(
                        200, text=sse({"type": kind, "error": {"code": "insufficient_quota"}})
                    )
                )

    async def test_terminal_frame_at_eof_and_sse_comments(self):
        text = ": heartbeat\n\nevent: response.completed\n" + sse(
            {"type": "response.completed", "response": completed()}
        ).rstrip("\n")
        result = await self.request(lambda r: httpx.Response(200, text=text))
        self.assertEqual(len(result["sources"]), 1)

    async def test_http_error_keeps_code_without_echoing_payload(self):
        with self.assertRaisesRegex(RuntimeError, "401: invalid_api_key") as error:
            await self.request(
                lambda r: httpx.Response(
                    401, json={"error": {"code": "invalid_api_key", "message": "private-query-secret"}}
                )
            )
        self.assertNotIn("private-query-secret", str(error.exception))

    async def test_cancellation_is_logged_and_propagated(self):
        async def cancel(request):
            raise asyncio.CancelledError()

        with patch.object(native, "finish_call") as finish, self.assertRaises(asyncio.CancelledError):
            await self.request(cancel)
        self.assertIsInstance(finish.call_args.kwargs["error"], asyncio.CancelledError)

    async def test_deadline_stops_slow_backend(self):
        async def slow(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json=completed())

        # A diagnostic caller label must not change the chat search deadline.
        for caller in ("tools:native_search", "news_broadcast:native_search"):
            with patch.object(config, "web_search_request_timeout", 0.1), self.assertRaises(asyncio.TimeoutError):
                await self.request(slow, caller=caller)

    async def test_explicit_budget_allows_slow_news_search_on_both_backends(self):
        async def slow(request):
            await asyncio.sleep(0.15)
            if str(request.url).endswith("/responses"):
                return httpx.Response(200, json=completed())
            return httpx.Response(200, json={"output": {"text": "A cited update."}})

        with patch.object(config, "web_search_request_timeout", 0.1), patch.object(news, "finish_call"):
            for spec in (GPT, "dashscope:test-search"):
                result = await self.request(slow, spec=spec, request_timeout=0.5, expected_timeout=0.5)
                self.assertEqual(result["content"], "A cited update.")

    async def test_explicit_budget_still_respects_global_request_cap(self):
        async def slow(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json=completed())

        with patch.object(config, "llm_request_timeout", 0.1), patch.object(news, "finish_call") as finish:
            for spec in (GPT, "dashscope:test-search"):
                with self.assertRaises(asyncio.TimeoutError):
                    await self.request(slow, spec=spec, request_timeout=0.5)
        # Cancellation of the DashScope request must also close its usage record.
        self.assertIsInstance(finish.call_args.kwargs["error"], asyncio.CancelledError)

    async def test_response_size_limit(self):
        with patch.object(native, "MAX_RESPONSE_BYTES", 10), self.assertRaisesRegex(RuntimeError, "too large"):
            await self.request(lambda r: httpx.Response(200, json=completed()))


class SearchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.log_patch = patch.object(models, "finish_call")
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.run = AgentRunner("search-test")
        self.token = current_run.set(self.run)
        self.addCleanup(current_run.reset, self.token)
        self.cooldown = patch.dict(web_search._search_pause_until, {}, clear=True)
        self.cooldown.start()
        self.addCleanup(self.cooldown.stop)
        self.env = patch.dict(
            os.environ, {**TEST_MODEL_ENV, "MODEL_SEARCH": GPT + " | dashscope:test-search"}, clear=True
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    async def test_search_registers_only_provider_sources_with_honest_filters(self):
        payload = native.parse_search_response(completed())
        with patch.object(native, "native_search", AsyncMock(return_value=payload)) as search:
            result = await web_search.search("query", domains=["example.com"], freshness_days=2)
        self.assertEqual(search.await_args.kwargs["domains"], ["example.com"])
        item = result["items"][0]
        self.assertIn(item["id"], self.run.sources)
        self.assertEqual(item["provenance"], "search_provider")
        self.assertEqual(item["read_state"], "search_snippet")
        self.assertIsNone(result["effective_filters"]["freshness_days"])
        self.assertIn("freshness_is_prompt_only", result["warnings"])
        self.assertNotIn("A cited update", item["excerpt"])

    async def test_search_uses_session_original_and_keeps_image_provenance(self):
        from src.plugins.llm_chat.image_references import ImageReferenceStore
        store = ImageReferenceStore()
        ref = store.remember(self.run.session_id, "aW1hZ2U=", "original")
        with patch('src.plugins.llm_chat.image_references.image_reference_store', store), \
             patch.object(native, "native_search", AsyncMock(return_value=native.parse_search_response(completed()))) as search:
            response = await web_search.search("谁", image_refs=[ref.id])
        self.assertEqual(search.await_args.kwargs["image_data_urls"], ["data:image/jpeg;base64,aW1hZ2U="])
        self.assertEqual(response["search_summary"]["input_image_refs"], [ref.id])

    async def test_search_does_not_replace_missing_or_other_session_image_with_text(self):
        from src.plugins.llm_chat.image_references import ImageReferenceStore
        store = ImageReferenceStore()
        ref = store.remember("other_session", "aW1hZ2U=", "private")
        with patch('src.plugins.llm_chat.image_references.image_reference_store', store), \
             patch.object(native, "native_search", AsyncMock()) as search:
            response = await web_search.search("谁", image_refs=[ref.id])
        search.assert_not_awaited()
        self.assertEqual(response["error"]["code"], "image_reference_unavailable_or_expired")

    async def test_domain_filter_is_enforced_locally_even_if_proxy_ignores_it(self):
        with patch.object(native, "native_search", AsyncMock(return_value=native.parse_search_response(completed()))):
            result = await web_search.search("query", domains=["different.example.com"])
        self.assertEqual(result["items"], [])
        self.assertFalse(self.run.sources)

    async def test_native_search_answer_retains_source_references(self):
        payload = native.parse_search_response(completed())
        payload["content"] = "2026 年的新价格与旧公告不同，请核对最新调价来源。"
        with patch.object(native, "native_search", AsyncMock(return_value=payload)):
            response = await web_search.search("现在 PS5 多少钱")
        summary = response["search_summary"]
        self.assertEqual(summary["content"], payload["content"])
        self.assertEqual(summary["read_state"], "search_summary")
        self.assertEqual(summary["related_source_ids"], [response["items"][0]["id"]])
        self.assertNotIn(payload["content"], response["items"][0]["excerpt"])
        self.assertIsNone(response["items"][0]["published_at"])
        self.assertIn("searched_at", response)

    async def test_read_web_only_exposed_and_allowed_for_user_links(self):
        from src.plugins.llm_chat import news_sources, tools
        source_id = web_search.register_url(self.run, URL)
        self.assertNotIn("read_web", [t["function"]["name"] for t in tools.get_chat_tools()])
        with patch.object(news_sources, "fetch_public_page", AsyncMock()) as fetch:
            rejected = await web_search.read(source_id)
        self.assertEqual(rejected["error"]["code"], "read_web_requires_user_provided_link")
        fetch.assert_not_awaited()
        user_id = web_search.register_url(self.run, URL + "/user", input_url=True)
        read_tool = next(t for t in tools.get_chat_tools() if t["function"]["name"] == "read_web")
        self.assertIn(user_id, read_tool["function"]["description"])
        self.assertNotIn(source_id, read_tool["function"]["description"])
        with patch.object(news_sources, "fetch_public_page", AsyncMock()) as fetch:
            rejected = await web_search.read(source_id)
        fetch.assert_not_awaited()
        self.assertEqual(rejected["status"], "invalid_arguments")

    async def test_read_keeps_publication_date_separate_from_fetch_time(self):
        from src.plugins.llm_chat import news_sources
        payload = native.parse_search_response(completed())
        payload["sources"][0]["published_at"] = "2022-08-25"
        with patch.object(native, "native_search", AsyncMock(return_value=payload)):
            response = await web_search.search("现在 PS5 多少钱")
        self.run.sources[response["items"][0]["id"]]["input_url"] = True
        page = {"status": "ok", "url": URL, "content_type": "text/html", "body": "<p>2022年8月25日的历史价格公告</p>"}
        with patch.object(news_sources, "fetch_public_page", AsyncMock(return_value=page)):
            read = await web_search.read(response["items"][0]["id"])
        item = read["items"][0]
        self.assertEqual(item["published_at"], "2022-08-25")
        self.assertEqual(item["publication_date_source"], "search_provider")
        self.assertNotEqual(item["fetched_at"][:10], item["published_at"])
        self.assertEqual(item["read_state"], "original")

    async def test_unreadable_page_keeps_failure_and_actionable_reason(self):
        from src.plugins.llm_chat import news_sources
        with patch.object(native, "native_search", AsyncMock(return_value=native.parse_search_response(completed()))):
            response = await web_search.search("query")
        self.run.sources[response["items"][0]["id"]]["input_url"] = True
        for status in ("too_large", "not_html", "ConnectError"):
            with self.subTest(status=status), patch.object(
                news_sources, "fetch_public_page", AsyncMock(return_value={"status": status}),
            ) as fetch:
                read = await web_search.read(response["items"][0]["id"])
                self.assertEqual(read["items"], [])
                self.assertEqual(read["error"]["code"], "page_" + status)
                self.assertTrue(read["warnings"])
                self.assertEqual(read["error"]["retryable"], status == "ConnectError")
                self.assertEqual(fetch.await_args.kwargs["max_bytes"], 8 * 1024 * 1024)
                self.assertNotIn("body", self.run.sources[response["items"][0]["id"]])

    async def test_account_failure_tries_other_provider_and_cooldown_is_scoped(self):
        payload = native.parse_search_response(completed())
        mock = AsyncMock(side_effect=[RuntimeError("insufficient_quota"), payload, payload])
        with patch.object(native, "native_search", mock):
            first = await web_search.search("query")
            second = await web_search.search("query again")
        self.assertTrue(first["items"] and second["items"])
        self.assertEqual(
            [call.args[0] for call in mock.await_args_list], [GPT, "dashscope:test-search", "dashscope:test-search"]
        )

    async def test_no_evidence_tries_next_search_backend(self):
        with patch.object(
            native, "native_search", AsyncMock(side_effect=[{"sources": []}, native.parse_search_response(completed())])
        ) as search:
            result = await web_search.search("query")
        self.assertEqual(search.await_count, 2)
        self.assertTrue(result["items"])

    async def test_search_cancellation_never_calls_fallback(self):
        with (
            patch.object(native, "native_search", AsyncMock(side_effect=asyncio.CancelledError())) as search,
            self.assertRaises(asyncio.CancelledError),
        ):
            await web_search.search("query")
        self.assertEqual(search.await_count, 1)

    async def test_legacy_search_uses_native_gpt_not_chat_completions(self):
        with (
            patch.object(native, "native_search", AsyncMock(return_value=native.parse_search_response(completed()))),
            patch.object(models.model_manager, "get_or_build_adapter") as adapter,
        ):
            result = await models.model_manager.chat([models.Message(role="user", content="query")], role="search")
        adapter.assert_not_called()
        self.assertIn(URL, result)
        self.assertEqual(result.model_spec, GPT)
        self.assertEqual(result.usage["total_tokens"], 14)

    async def test_legacy_gpt_failure_keeps_qwen_fallback(self):
        adapter = type("Adapter", (), {"chat": AsyncMock(return_value=models.ChatResponse("Qwen result"))})()
        with (
            patch.object(native, "native_search", AsyncMock(side_effect=RuntimeError("no search evidence"))),
            patch.object(models.model_manager, "get_or_build_adapter", return_value=adapter) as build,
        ):
            result = await models.model_manager.chat([models.Message(role="user", content="query")], role="search")
        build.assert_called_once_with("dashscope:test-search")
        self.assertEqual(result.model_spec, "dashscope:test-search")
        self.assertTrue(adapter.chat.await_args.kwargs["extra_body"]["enable_search"])

    async def test_news_selects_from_native_date_evidence_without_downloading_sources(self):
        payload = native.parse_search_response(completed())
        payload["content"] = json.dumps({"findings": [{
            "url": URL, "published_at": "2026-09-15", "date_confirmed": True,
            "date_quote": "Published September 15, 2026", "summary": "Actual update",
        }]})

        selection = (
            '{"items":[{"source_index":1,"published_at":"1999-01-01","summary":"Actual update","newsworthiness":5}]}'
        )
        with (
            patch.object(native, "native_search", AsyncMock(return_value=copy.deepcopy(payload))) as search,
            patch.object(config, "news_search_request_timeout", 45.0),
            patch.object(news.model_manager, "chat", AsyncMock(return_value=selection)) as choose,
            patch("src.plugins.llm_chat.news_sources.fetch_public_page", AsyncMock()) as fetch,
        ):
            result = await news.search_news_interest({"label": "AI"}, now=datetime(2026, 9, 15, tzinfo=timezone.utc))
        self.assertEqual(search.await_args.args[0], GPT)
        self.assertEqual(search.await_args.kwargs["request_timeout"], 45.0)
        self.assertIn('"index": 1', choose.await_args.args[0][0].content)
        self.assertIn(URL, choose.await_args.args[0][0].content)
        self.assertFalse(choose.await_args.kwargs["allow_fallback"])
        self.assertEqual(result["items"][0]["url"], URL)
        self.assertEqual(result["items"][0]["published_at"], "2026-09-15")
        self.assertEqual(result["items"][0]["date_evidence"], "native_search_citation")
        self.assertEqual(result["items"][0]["date_quote"], "Published September 15, 2026")
        fetch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
