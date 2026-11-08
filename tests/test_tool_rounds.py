import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import models, tools
from src.plugins.llm_chat.config import config


class FakeToolMessage:
    def __init__(self, tool_calls, content=None):
        self.tool_calls = tool_calls
        self.content = content
        self.reasoning_details = None

    def model_dump(self, exclude_none=True):
        data = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }
        if exclude_none and data["content"] is None:
            data.pop("content")
        return data


def make_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def make_response(message):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=None,
    )


class MultiRoundToolTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()
        tools.clear_voice_context()

    async def test_recall_and_media_in_same_batch_defers_media_until_next_round(self):
        recall_call = make_tool_call(
            "call-recall",
            "recall_history",
            {"keyword": "喜欢的衣服", "role": "assistant"},
        )
        premature_image_call = make_tool_call(
            "call-image-early",
            "generate_image",
            {"prompt": "未使用召回结果的图片"},
        )
        recalled_image_call = make_tool_call(
            "call-image-final",
            "generate_image",
            {"prompt": "根据召回结果，穿蓝色汉服的磷叶石"},
        )
        create = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([recall_call, premature_image_call])),
                make_response(FakeToolMessage([recalled_image_call])),
                make_response(FakeToolMessage([], content="喏，按你喜欢的样子画出来了。")),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:gpt-5.6-luna",
            api_key="test-key",
            base_url="https://proxy.test/v1",
            model_id="gpt-5.6-luna",
        )
        adapter.client = fake_client
        tools.set_image_context(group_id=None, user_id=42, bot=object())
        executed = []

        async def fake_execute_tool(name, args):
            executed.append((name, args))
            if name == "recall_history":
                return "历史偏好：喜欢蓝色汉服"
            if name == "generate_image":
                context = tools._image_context.get()
                context["sent_in_round"] = True
                context["last_sent"] = {
                    "prompt": args["prompt"],
                    "description": "穿蓝色汉服的磷叶石",
                    "model": "cliproxyapi:gpt-image-2",
                    "size": "1024x1536",
                }
                tools._image_context.set(context)
                return "图片已生成并发送"
            return "unexpected"

        with (
            patch.object(config, "llm_max_tool_rounds", 3),
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(tools, "execute_tool", side_effect=fake_execute_tool),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="按以前喜欢的衣服画一张自拍"),
                ],
                tools_enabled=True,
                required_tool="generate_image",
            )

        self.assertEqual(str(response), "喏，按你喜欢的样子画出来了。")
        self.assertEqual(
            [name for name, _ in executed],
            ["recall_history", "generate_image"],
        )
        self.assertEqual(
            executed[1][1]["prompt"],
            "根据召回结果，穿蓝色汉服的磷叶石",
        )
        self.assertEqual(create.await_count, 3)
        second_request = create.await_args_list[1].kwargs
        self.assertIn("tools", second_request)
        tool_results = [
            item["content"]
            for item in second_request["messages"]
            if item.get("role") == "tool"
        ]
        self.assertTrue(any("历史偏好：喜欢蓝色汉服" in item for item in tool_results))
        self.assertTrue(any("媒体调用已暂缓" in item for item in tool_results))
        self.assertNotIn("tools", create.await_args_list[2].kwargs)

    async def test_required_media_tool_is_forced_then_followed_by_tool_free_reply(self):
        image_call = make_tool_call(
            "call-image",
            "generate_image",
            {"prompt": "窗边自拍", "orientation": "portrait"},
        )
        create = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([image_call])),
                make_response(FakeToolMessage([], content="只给你看一次，别太得意。")),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:gpt-5.6-luna",
            api_key="test-key",
            base_url="https://proxy.test/v1",
            model_id="gpt-5.6-luna",
        )
        adapter.client = fake_client
        tools.set_image_context(group_id=None, user_id=42, bot=object())
        executed = []

        async def fake_execute_tool(name, args):
            executed.append((name, args))
            context = tools._image_context.get()
            context["sent_in_round"] = True
            context["last_sent"] = {
                "prompt": args["prompt"],
                "description": "磷叶石站在窗边自拍。",
                "model": "cliproxyapi:gpt-image-2",
                "size": "1024x1536",
            }
            tools._image_context.set(context)
            return '{"status":"sent","media_type":"image"}'

        with (
            patch.object(config, "llm_max_tool_rounds", 3),
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(tools, "execute_tool", side_effect=fake_execute_tool),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="给我生成一张自拍"),
                ],
                tools_enabled=True,
                required_tool="generate_image",
            )

        self.assertEqual(str(response), "只给你看一次，别太得意。")
        self.assertEqual([name for name, _ in executed], ["generate_image"])
        first_request = create.await_args_list[0].kwargs
        # Planner 只声明“本轮必须有图片”；Chat LLM 的第一步仍可先
        # recall/search。若它直接结束，适配层才补一次 named tool_choice。
        self.assertNotIn("tool_choice", first_request)
        first_tool_names = {
            item["function"]["name"] for item in first_request["tools"]
        }
        self.assertIn("generate_image", first_tool_names)
        self.assertNotIn("send_voice", first_tool_names)
        final_request = create.await_args_list[1].kwargs
        self.assertNotIn("tools", final_request)
        self.assertNotIn("tool_choice", final_request)

    async def test_required_media_is_forced_if_model_attempts_text_only(self):
        image_call = make_tool_call(
            "call-forced-image",
            "generate_image",
            {"prompt": "被适配层兜底触发的自拍"},
        )
        create = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([], content="我知道了")),
                make_response(FakeToolMessage([image_call])),
            ]
        )
        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:gpt-5.6-luna",
            api_key="test-key",
            base_url="https://proxy.test/v1",
            model_id="gpt-5.6-luna",
        )
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        tools.set_image_context(group_id=None, user_id=42, bot=object())

        async def fake_execute_tool(name, args):
            context = tools._image_context.get()
            context["sent_in_round"] = True
            context["last_sent"] = {
                "prompt": args["prompt"],
                "description": "磷叶石自拍。",
                "model": "cliproxyapi:gpt-image-2",
                "size": "1024x1024",
            }
            tools._image_context.set(context)
            return '{"status":"sent","media_type":"image"}'

        with (
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(tools, "execute_tool", side_effect=fake_execute_tool),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="给我一张自拍"),
                ],
                tools_enabled=True,
                required_tool="generate_image",
                media_followup=False,
            )

        self.assertEqual(str(response), "")
        self.assertNotIn("tool_choice", create.await_args_list[0].kwargs)
        self.assertEqual(
            create.await_args_list[1].kwargs["tool_choice"],
            {"type": "function", "function": {"name": "generate_image"}},
        )

    async def test_media_only_ends_after_successful_tool_call(self):
        image_call = make_tool_call(
            "call-image-only",
            "generate_image",
            {"prompt": "只发图片，不补文字"},
        )
        create = AsyncMock(
            return_value=make_response(FakeToolMessage([image_call]))
        )
        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:gpt-5.6-luna",
            api_key="test-key",
            base_url="https://proxy.test/v1",
            model_id="gpt-5.6-luna",
        )
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        tools.set_image_context(group_id=None, user_id=42, bot=object())

        async def fake_execute_tool(name, args):
            context = tools._image_context.get()
            context["sent_in_round"] = True
            context["last_sent"] = {
                "prompt": args["prompt"],
                "description": "一张无需文字说明的图片。",
                "model": "cliproxyapi:gpt-image-2",
                "size": "1024x1024",
            }
            tools._image_context.set(context)
            return '{"status":"sent","media_type":"image"}'

        with (
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(tools, "execute_tool", side_effect=fake_execute_tool),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="给我一张图"),
                ],
                tools_enabled=True,
                required_tool="generate_image",
                media_followup=False,
            )

        self.assertEqual(str(response), "")
        self.assertEqual(create.await_count, 1)
        first_request = create.await_args_list[0].kwargs
        self.assertIn(
            "媒体后面没有 text",
            first_request["messages"][0]["content"],
        )

    async def test_media_only_still_explains_a_failed_tool_call(self):
        image_call = make_tool_call(
            "call-image-failed",
            "generate_image",
            {"prompt": "会失败的图片"},
        )
        create = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([image_call])),
                make_response(FakeToolMessage([], content="这次没生成出来，稍后再试。")),
            ]
        )
        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:gpt-5.6-luna",
            api_key="test-key",
            base_url="https://proxy.test/v1",
            model_id="gpt-5.6-luna",
        )
        adapter.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )

        with (
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(
                tools,
                "execute_tool",
                AsyncMock(return_value="generate_image 生成失败：上游暂时不可用"),
            ),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="给我一张图"),
                ],
                tools_enabled=True,
                required_tool="generate_image",
                media_followup=False,
            )

        self.assertEqual(str(response), "图片生成或发送失败，请稍后再试。")
        self.assertEqual(create.await_count, 1)

    async def test_tool_round_limit_forces_a_final_request_without_tools(self):
        recall_call = make_tool_call(
            "call-recall",
            "recall_history",
            {"keyword": "旧信息"},
        )
        create = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([recall_call])),
                make_response(FakeToolMessage([], content="根据历史给出的最终回复")),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:gpt-5.6-luna",
            api_key="test-key",
            base_url="https://proxy.test/v1",
            model_id="gpt-5.6-luna",
        )
        adapter.client = fake_client

        with (
            patch.object(config, "llm_max_tool_rounds", 1),
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(
                tools,
                "execute_tool",
                AsyncMock(return_value="召回到的旧信息"),
            ),
        ):
            response = await adapter.chat(
                [
                    models.Message(role="system", content="system"),
                    models.Message(role="user", content="查一下旧信息"),
                ],
                tools_enabled=True,
            )

        self.assertEqual(str(response), "根据历史给出的最终回复")
        self.assertEqual(create.await_count, 2)
        first_tool_names = {
            item["function"]["name"]
            for item in create.await_args_list[0].kwargs["tools"]
        }
        self.assertNotIn("generate_image", first_tool_names)
        self.assertNotIn("send_voice", first_tool_names)
        self.assertNotIn("tools", create.await_args_list[1].kwargs)


class MediaAuthorizationTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()
        tools.clear_voice_context()

    async def test_fact_check_cannot_turn_invalid_image_args_into_forced_image_retry(self):
        from src.plugins.llm_chat.agent_runner import AgentRunner, current_run

        tools.set_image_context(None, 42, bot=object())
        tools.set_voice_context(None, 42, bot=object())
        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        self.addAsyncCleanup(adapter.client.close)
        for required_tool in (None, "send_voice"):
            with self.subTest(required_tool=required_tool):
                run = AgentRunner(budget=180)
                run.transcript = [{"role": "tool", "name": "search_web", "content": "不能确认截图真伪"}]
                original_deadline = run.deadline
                token = current_run.set(run)
                try:
                    create = AsyncMock(return_value=make_response(FakeToolMessage([
                        make_tool_call("wrong-image", "generate_image", {"query": "核查这封邮件", "limit": 10}),
                    ])))
                    with (
                        patch.object(run, "remaining", return_value=40),
                        patch.object(models, "model_supports_tools", return_value=True),
                        patch.object(adapter, "_create_completion", create),
                        patch.object(tools, "execute_tool", AsyncMock()) as execute,
                        self.assertRaisesRegex(RuntimeError, "未授权的工具"),
                    ):
                        await adapter.chat([models.Message("user", "核查一下")], tools_enabled=True,
                                           required_tool=required_tool)
                    execute.assert_not_awaited()
                    create.assert_awaited_once()
                    request = create.await_args.args[1]
                    self.assertNotIn("generate_image", {t["function"]["name"] for t in request.get("tools", [])})
                    self.assertNotIn("tool_choice", request)
                    self.assertEqual(run.deadline, original_deadline)
                    self.assertEqual(run.actions, {})
                    self.assertIsNone(run.authorized_media_tool)
                finally:
                    current_run.reset(token)

    async def test_explicit_media_can_search_then_execute_after_research_closes(self):
        from copy import deepcopy
        from src.plugins.llm_chat.agent_runner import AgentRunner, current_run
        from src.plugins.llm_chat.capabilities import MEDIA_TOOLS, READ_TOOLS

        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        self.addAsyncCleanup(adapter.client.close)
        for name, arguments, context in (
            ("generate_image", {"prompt": "根据查证结果画一只鸽子"}, tools._image_context),
            ("send_voice", {"text": "查证后的天气预报"}, tools._voice_context),
        ):
            with self.subTest(tool=name):
                tools.set_image_context(None, 42, bot=object())
                tools.set_voice_context(None, 42, bot=object())
                run = AgentRunner(budget=180)
                token = current_run.set(run)
                closing = False
                requests = []
                responses = [
                    make_response(FakeToolMessage([make_tool_call("search", "search_web", {"query": "参考资料"})])),
                    make_response(FakeToolMessage([make_tool_call("media", name, arguments)])),
                ]

                async def create(client, kwargs):
                    requests.append(deepcopy(kwargs))
                    return responses[len(requests) - 1]

                async def execute(tool, args):
                    nonlocal closing
                    if tool == "search_web":
                        closing = True
                        return '{"status":"ok","content":"已取得参考资料"}'
                    self.assertEqual(tool, name)
                    context.get()["sent_in_round"] = True
                    return '{"delivery_status":"sent"}'

                try:
                    with (
                        patch.object(run, "remaining", side_effect=lambda: 40 if closing else 170),
                        patch.object(models, "model_supports_tools", return_value=True),
                        patch.object(adapter, "_create_completion", side_effect=create),
                        patch.object(tools, "execute_tool", AsyncMock(side_effect=execute)) as executed,
                    ):
                        response = await adapter.chat([models.Message("user", "先查资料，再按计划生成媒体")],
                                                      tools_enabled=True, required_tool=name, media_followup=False)
                    self.assertEqual(str(response), "")
                    self.assertEqual([call.args[0] for call in executed.await_args_list], ["search_web", name])
                    self.assertEqual(len(requests), 2)
                    names = {t["function"]["name"] for t in requests[1]["tools"]}
                    self.assertEqual(names & MEDIA_TOOLS, {name})
                    self.assertFalse(names & READ_TOOLS)
                    self.assertGreaterEqual(run.deadline, run.started_at + config.chat_media_turn_timeout)
                    self.assertIsNone(run.authorized_media_tool)
                finally:
                    current_run.reset(token)

    async def test_provider_fallback_keeps_media_plan_and_search_observations(self):
        tools.set_image_context(None, 42, bot=object())
        first = models.OpenAICompatibleAdapter("openai:one", "dummy", "https://example.test/v1", "one")
        second = models.OpenAICompatibleAdapter("openai:two", "dummy", "https://example.test/v1", "two")
        self.addAsyncCleanup(first.client.close)
        self.addAsyncCleanup(second.client.close)
        manager = models.ModelManager()

        async def execute(name, args):
            if name == "search_web":
                return '{"status":"ok","content":"参考资料：鸽子的翅膀"}'
            self.assertEqual(name, "generate_image")
            tools._image_context.get()["sent_in_round"] = True
            return '{"delivery_status":"sent"}'

        with (
            patch.object(manager, "_resolve_candidates", return_value=("openai:one", ["openai:one", "openai:two"])),
            patch.object(manager, "get_or_build_adapter", side_effect=[first, second]),
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(first, "_create_completion", AsyncMock(side_effect=[
                make_response(FakeToolMessage([make_tool_call("search", "search_web", {"query": "鸽子"})])),
                RuntimeError("provider unavailable"),
            ])),
            patch.object(second, "_create_completion", AsyncMock(return_value=make_response(FakeToolMessage([
                make_tool_call("image", "generate_image", {"prompt": "根据参考资料画鸽子"}),
            ])))) as create,
            patch.object(tools, "execute_tool", AsyncMock(side_effect=execute)) as executed,
        ):
            response = await manager.run_turn([models.Message("user", "查一下鸽子，再画张图")],
                                              required_tool="generate_image", media_followup=False)
        self.assertEqual(str(response), "")
        self.assertEqual([call.args[0] for call in executed.await_args_list], ["search_web", "generate_image"])
        request = create.await_args.args[1]
        self.assertTrue(any(m.get("name") == "search_web" and "鸽子的翅膀" in m.get("content", "")
                            for m in request["messages"]))
        self.assertIsNone(response.agent_run.authorized_media_tool)


class MediaValidationRetryTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()
        tools.clear_voice_context()

    async def run_media(self, responses, execute, *, rounds=3, followup=False):
        from copy import deepcopy

        requests = []

        async def create(**kwargs):
            requests.append(deepcopy(kwargs))
            response = responses[len(requests) - 1]
            if isinstance(response, Exception):
                raise response
            response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
            return response

        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:grok-4.6", model_id="grok-4.6",
            api_key="test-key", base_url="https://proxy.test/v1",
        )
        await adapter.client.close()
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with (
            patch.object(config, "llm_max_tool_rounds", rounds),
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(tools, "execute_tool", execute),
        ):
            result = await adapter.chat(
                [models.Message(role="user", content="生成今日群聊漫画")],
                tools_enabled=True, required_tool="generate_image", media_followup=followup,
            )
        self.assertTrue(all(request["stream"] is False for request in requests))
        self.assertEqual(result.usage["total_tokens"], len(requests) * 12)
        return result, requests

    def image_response(self, prompt="群聊漫画", *, duplicate=False):
        calls = [make_tool_call("image-1", "generate_image", {"prompt": prompt})]
        if duplicate:
            calls.append(make_tool_call("image-2", "generate_image", {"prompt": prompt}))
        return make_response(FakeToolMessage(calls))

    async def test_long_prompt_is_corrected_once_with_real_tool_then_sent(self):
        tools.set_image_context(group_id=None, user_id=42, bot=object())
        real_execute = tools.execute_tool

        async def execute(name, args):
            if len(args["prompt"]) > 1000:
                return await real_execute(name, args)
            tools._image_context.get()["sent_in_round"] = True
            return '{"status":"sent"}'

        mocked = AsyncMock(side_effect=execute)
        with patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}), patch.object(config, "image_gen_max_prompt_chars", 1000):
            result, requests = await self.run_media(
                [self.image_response("画" * 1001), self.image_response()], mocked,
            )
        self.assertEqual(str(result), "")
        self.assertEqual(mocked.await_count, 2)
        self.assertEqual(len(requests), 2)
        retry = requests[1]
        self.assertEqual(retry["tool_choice"]["function"]["name"], "generate_image")
        self.assertEqual([tool["function"]["name"] for tool in retry["tools"]], ["generate_image"])
        self.assertIn("超过 1000 字", retry["messages"][-1]["content"])
        self.assertEqual(retry["tools"][0]["function"]["parameters"]["properties"]["prompt"]["maxLength"], 1000)

    async def test_repeated_invalid_calls_stop_after_one_correction_even_in_same_batch(self):
        execute = AsyncMock(return_value=tools.MediaValidationFailure("超过 1000 字"))
        result, requests = await self.run_media(
            [self.image_response(duplicate=True), self.image_response(duplicate=True)], execute,
        )
        self.assertEqual(execute.await_count, 2)
        self.assertEqual(len(requests), 2)
        self.assertIn("超过 1000 字", str(result))
        self.assertNotIn("稍后", str(result))

    async def test_validation_retry_respects_overall_round_cap(self):
        execute = AsyncMock(return_value=tools.MediaValidationFailure("超过 1000 字"))
        result, requests = await self.run_media([self.image_response()], execute, rounds=1)
        self.assertEqual(len(requests), 1)
        execute.assert_awaited_once()
        self.assertIn("失败", str(result))

    async def test_fake_json_after_validation_failure_never_reaches_qq(self):
        from src.plugins.llm_chat.reply_policy import enforce_reply_budget
        from src.plugins.llm_chat.utils import clean_response

        fake = '```json\n{"name":"generate_image","parameters":{"prompt":"漫画"}}\n```\nWait, I need to use the actual function call format.'
        execute = AsyncMock(return_value=tools.MediaValidationFailure("超过 1000 字"))
        result, requests = await self.run_media(
            [self.image_response(), make_response(FakeToolMessage([], content=fake))], execute,
        )
        self.assertEqual(len(requests), 2)
        execute.assert_awaited_once()
        self.assertIn("超过 1000 字", enforce_reply_budget(clean_response(result)))

    async def test_generation_failure_is_not_retried_or_rephrased_by_model(self):
        for error in ["generate_image 生成失败：上游超时", "generate_image 今日进程内额度已用完", "generate_image 暂时冷却中"]:
            with self.subTest(error=error):
                execute = AsyncMock(return_value=error)
                result, requests = await self.run_media([self.image_response(duplicate=True)], execute)
                execute.assert_awaited_once()
                self.assertEqual(len(requests), 1)
                self.assertIn("失败", str(result))

    async def test_retry_request_error_does_not_restart_model_fallback_chain(self):
        execute = AsyncMock(return_value=tools.MediaValidationFailure("超过 1000 字"))
        # Usage only exists for the first completed request.
        adapter = models.OpenAICompatibleAdapter(
            model_name="cliproxyapi:grok-4.6", model_id="grok-4.6",
            api_key="test-key", base_url="https://proxy.test/v1",
        )
        self.addAsyncCleanup(adapter.client.close)
        create = AsyncMock(side_effect=[self.image_response(), RuntimeError("timeout")])
        with patch.object(adapter, "_create_completion", create), patch.object(tools, "execute_tool", execute), patch.object(models, "model_supports_tools", return_value=True):
            result = await adapter.chat([models.Message(role="user", content="画图")], tools_enabled=True, required_tool="generate_image")
        self.assertIn("失败", str(result))
        execute.assert_awaited_once()
        self.assertEqual(create.await_count, 2)

    async def test_sent_or_unconfirmed_media_cannot_trigger_another_tool(self):
        for delivery in ("sent", "unconfirmed"):
            for final in (self.image_response(), make_response(FakeToolMessage([], content='{"name":"generate_image","parameters":{"prompt":"again"}}'))):
                with self.subTest(delivery=delivery, final=final):
                    tools.set_image_context(group_id=None, user_id=42, bot=object())

                    async def execute(name, args):
                        tools._image_context.get()["sent_in_round"] = True
                        tools._image_context.get()["last_sent"] = {"delivery_status": delivery}
                        return json.dumps({"status": delivery})

                    mocked = AsyncMock(side_effect=execute)
                    result, requests = await self.run_media([self.image_response(duplicate=True), final], mocked, followup=True)
                    self.assertEqual(str(result), "")
                    mocked.assert_awaited_once()
                    self.assertNotIn("tools", requests[1])


if __name__ == "__main__":
    unittest.main()
