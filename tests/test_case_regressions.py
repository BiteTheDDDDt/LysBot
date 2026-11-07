"""Regressions for media routing, lost search context and vision fallback deadlines."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import agent_runner, chat, context_builder, models, tools
from src.plugins.llm_chat.config import config
from test_tool_rounds import FakeToolMessage, make_response, make_tool_call


class CaseRegressions(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_image_context()
        tools.clear_voice_context()

    async def test_media_output_plan_authorizes_image_through_run_turn(self):
        from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart

        decision = PlannerDecision(parts=[ReplyPart("image")])
        tools.set_image_context(99, 42, bot=SimpleNamespace())
        create = AsyncMock(return_value=make_response(FakeToolMessage([
            make_tool_call("image", "generate_image", {"prompt": "一只鸽子"}),
        ])))
        adapter = models.OpenAICompatibleAdapter(
            "openai:gpt-5.6-sol", "test-key", "https://example.com/v1", "gpt-5.6-sol",
        )
        adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        async def send(name, arguments):
            self.assertEqual(name, "generate_image")
            self.assertEqual(arguments["prompt"], "一只鸽子")
            tools._image_context.get()["sent_in_round"] = True
            await agent_runner.record_tool_delivery("sent", {"message_id": 1})
            return "图片已发送"

        manager = models.ModelManager()
        with (
            patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}),
            patch.object(agent_runner, "_runtime_path", None),
            patch.object(models, "model_supports_tools", return_value=True),
            patch.object(manager, "_resolve_candidates", return_value=(adapter.model_name, [adapter.model_name])),
            patch.object(manager, "get_or_build_adapter", return_value=adapter),
            patch.object(tools, "execute_tool", AsyncMock(side_effect=send)) as execute,
        ):
            response = await manager.run_turn(
                [models.Message("user", "那发一张鸽子的图片")],
                session_id="user_42", required_tool=decision.required_tool,
                media_followup=decision.media_followup,
            )
        self.assertEqual(str(response), "")
        create.assert_awaited_once()
        execute.assert_awaited_once()
        self.assertIn("generate_image", {tool["function"]["name"] for tool in create.await_args.kwargs["tools"]})
        self.assertEqual(list(response.agent_run.actions.values())[0]["status"], "sent")
        self.assertGreaterEqual(response.agent_run.deadline - response.agent_run.started_at,
                                config.chat_media_turn_timeout)
        self.assertIsNone(response.agent_run.authorized_media_tool)

    async def test_media_context_does_not_authorize_an_unplanned_media_tool(self):
        run = agent_runner.AgentRunner("test")
        token = agent_runner.current_run.set(run)
        try:
            with patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}), patch.dict("os.environ", {"MODEL_TTS": "google:test-tts"}):
                tools.set_image_context(99, 42, bot=SimpleNamespace())
                tools.set_voice_context(99, 42, bot=SimpleNamespace())
                for required in (None, "generate_image", "send_voice"):
                    selected = {t["function"]["name"] for t in tools.get_chat_tools(required)}
                    self.assertEqual(selected & {"generate_image", "send_voice"}, {required} if required else set())
        finally:
            agent_runner.current_run.reset(token)

    async def test_unplanned_media_cannot_extend_outer_deadline_or_enter_validation_retry(self):
        manager = models.ModelManager()
        async def stages(*args, **kwargs):
            run = agent_runner.current_run.get()
            execute = AsyncMock()
            first_deadline = run.deadline
            for name in ("generate_image", "send_voice"):
                for raw_args in ('{"query":"核查这封邮件"}', 'invalid json'):
                    result = await run.execute(name, raw_args, execute)
                    self.assertEqual(json.loads(result)["error"]["code"], "media_not_authorized")
                    self.assertNotIsInstance(result, tools.MediaValidationFailure)
                    self.assertEqual(run.deadline, first_deadline)
            execute.assert_not_awaited()
            self.assertEqual(run.actions, {})
            self.assertEqual(run.tool_calls, 0)
            return models.ChatResponse("ok")
        with (
            patch.object(config, "chat_turn_timeout", 0.1),
            patch.object(config, "chat_media_turn_timeout", 0.6),
            patch.object(agent_runner, "_runtime_path", None),
            patch.object(manager, "_run_stages", side_effect=stages),
        ):
            self.assertEqual(str(await manager.run_turn([], session_id="user_42")), "ok")

    async def test_vision_times_out_one_candidate_and_tries_next(self):
        manager = models.ModelManager()
        async def hang(*args, **kwargs):
            await asyncio.Event().wait()
        first = SimpleNamespace(chat_with_images=AsyncMock(side_effect=hang))
        second = SimpleNamespace(chat_with_images=AsyncMock(return_value=models.ChatResponse("一只鸽子")))
        with (
            patch.object(config, "vision_request_timeout", 0.1),
            patch.object(manager, "_resolve_candidates", return_value=("openai:slow", ["openai:slow", "openai:fast"])),
            patch.object(manager, "get_or_build_adapter", side_effect=[first, second]),
        ):
            result = await asyncio.wait_for(manager._chat_with_images_impl([], ["image"]), timeout=1)
        self.assertEqual(str(result), "一只鸽子")
        first.chat_with_images.assert_awaited_once()
        second.chat_with_images.assert_awaited_once()

    async def test_request_timeout_is_not_overwritten_by_generic_step_timeout(self):
        run = agent_runner.AgentRunner("user_42", budget=10)
        backend = SimpleNamespace(model_name="openai:vision", _create_completion=AsyncMock(return_value="ok"))
        await run.step(backend, None, {"messages": [], "timeout": 0.25})
        self.assertEqual(backend._create_completion.await_args.args[1]["timeout"], 0.25)

    async def test_empty_meme_selection_is_visible_without_changing_literal_user_content(self):
        from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
        question = '解释代码里的 "[表情: 捂脸笑]"'
        with (
            patch.object(config, "enable_sentiment_analysis", False),
            patch.object(context_builder.memory_manager, "get_fixed_context_memory", AsyncMock(return_value={})),
        ):
            messages = await context_builder.build_context_messages(
                history=[], session_id="user_42", query="", user_id=42, group_id=None,
                user_content=question, planner_decision=PlannerDecision(parts=[ReplyPart("meme", "cute")]),
                planned_meme=None,
            )
        self.assertIn("本轮表情预选结果", messages[0].content)
        self.assertIn("不代表没有图库", messages[0].content)
        self.assertEqual(messages[-1].content, question)

    async def test_received_face_is_described_but_literal_format_examples_are_preserved(self):
        literal = '代码示例："[表情: 捂脸笑]"'
        text, images = await chat._render_message_value(
            SimpleNamespace(),
            [{"type": "face", "data": {"id": "14"}}, {"type": "text", "data": {"text": literal}}],
            chat._ExpansionBudget(), depth=0,
        )
        self.assertIn("[收到QQ表情：", text)
        self.assertIn(literal, text)
        self.assertEqual(images, [])
