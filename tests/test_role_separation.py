import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import models
from test_tool_rounds import FakeToolMessage, make_response


class RoleSeparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_social_handoff_treats_boundary_notes_as_internal_not_spoken_lines(self):
        # Reproduce both structured plans and the plain-text plan from the
        # 2026-10-21 banter case. The latter must not bypass delivery guidance.
        for structured in (True, False):
            with self.subTest(structured=structured):
                manager = models.ModelManager()
                planner = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
                note = "前面42分是随口报的；顺着玩笑给个算式，不代表真实计算过程。"
                requirements = ["不能把接梗算式说成真实算法"]
                payload = json.dumps({"response_mode": "social", "verify_group_identity": False,
                                      "content": note, "reply_requirements": requirements}) if structured else note
                chat = AsyncMock()
                chat.chat.return_value = models.ChatResponse("四十是实力，两分是我嘴硬")
                try:
                    with (
                        patch.object(manager, "_resolve_candidates", side_effect=lambda **kw: (
                            "openai:" + kw["role"], ["openai:" + kw["role"]])),
                        patch.object(manager, "get_or_build_adapter", side_effect=[planner, chat]),
                        patch.object(planner, "_create_completion", AsyncMock(return_value=make_response(
                            FakeToolMessage([], content=payload)))),
                    ):
                        result = await manager.run_turn([
                            models.Message("assistant", "我？42分，稳得很！"),
                            models.Message("user", "我不信，计算过程给一下"),
                        ], role="chat")
                    handoff = chat.chat.await_args.args[0][-1].content
                    self.assertIn(note, handoff)
                    self.assertIn("核验备注和内部顾虑不作为答复要点", handoff)
                    self.assertIn("普通玩笑直接接话", handoff)
                    self.assertNotIn("本轮必须保留的要求", handoff)
                    self.assertEqual(result.agent_run.planning_result, note)
                    self.assertEqual(str(result), "四十是实力，两分是我嘴硬")
                finally:
                    await planner.client.close()

    async def test_serious_source_explanation_is_preserved_without_keyword_filtering(self):
        from src.plugins.llm_chat.agent_runner import current_run
        manager = models.ModelManager()
        answer = "刚才的算式是瞎编的，和程序实际计算没有关系"
        async def call(messages, **kwargs):
            if kwargs["role"] == "planner":
                run = current_run.get()
                run.response_mode = "factual"
                run.reply_requirements = ["不能把玩笑算式说成真实算法"]
                return models.ChatResponse("认真追问实际算法，需要澄清前述算式并非程序计算。")
            handoff = messages[-1].content
            self.assertIn("不能把玩笑算式说成真实算法", handoff)
            self.assertIn("事实回答仍保留必要来源", handoff)
            return models.ChatResponse(answer)
        with patch.object(manager, "_chat_impl", side_effect=call):
            result = await manager.run_turn([models.Message("user", "认真问，程序真的按这个算的吗？")])
        self.assertEqual(str(result), answer)

    async def test_explicit_quota_cooldown_skips_and_then_recovers_primary(self):
        from types import SimpleNamespace
        manager = models.ModelManager()
        upstream = RuntimeError("quota")
        upstream.status_code = 429
        upstream.body = {"error": {"code": "model_cooldown", "reset_seconds": 60}}
        wrapped = RuntimeError("API error")
        wrapped.__cause__ = upstream
        primary = SimpleNamespace(chat=AsyncMock(side_effect=[wrapped, models.ChatResponse("恢复")]))
        backup = SimpleNamespace(chat=AsyncMock(return_value=models.ChatResponse("备用")))
        with (
            patch.object(manager, "_model_cooldowns", {}),
            patch.object(models.time, "monotonic", return_value=100) as clock,
            patch.object(manager, "_resolve_candidates", return_value=("openai:primary", ["openai:primary", "openai:backup"])),
            patch.object(manager, "get_or_build_adapter", side_effect=lambda spec: primary if spec.endswith("primary") else backup),
        ):
            self.assertEqual(str(await manager.chat([], role="chat")), "备用")
            self.assertEqual(str(await manager.chat([], role="chat")), "备用")
            self.assertEqual(primary.chat.await_count, 1)
            clock.return_value = 161
            self.assertEqual(str(await manager.chat([], role="chat")), "恢复")
            self.assertEqual(primary.chat.await_count, 2)

    def test_quota_cooldown_does_not_guess_from_generic_errors(self):
        self.assertEqual(models._quota_reset_delay(RuntimeError("429 reset_seconds: 60")), 0)
        error = RuntimeError("quota")
        error.status_code = 429
        for code, delay in (("rate_limit", 60), ("model_cooldown", -1), ("model_cooldown", "60"), ("model_cooldown", float("nan"))):
            error.body = {"error": {"code": code, "reset_seconds": delay}}
            self.assertEqual(models._quota_reset_delay(error), 0)

    async def test_proactive_can_skip_participation_but_direct_question_still_renders(self):
        from src.plugins.llm_chat.agent_runner import current_run
        from src.plugins.llm_chat.conversation_style import unpack_plan
        from src.plugins.llm_chat.reply_policy import PROACTIVE_REPLY, SHORT_REPLY
        parsed = unpack_plan(json.dumps({"response_mode": "factual", "verify_group_identity": False,
                                         "content": "缺少出处，无法提供有用解释", "should_reply": False}))
        self.assertFalse(parsed["should_reply"])
        for policy in (PROACTIVE_REPLY, SHORT_REPLY):
            manager = models.ModelManager()
            async def call(messages, **kwargs):
                if kwargs["role"] == "planner":
                    current_run.get().should_reply = parsed["should_reply"]
                    return models.ChatResponse(parsed["content"])
                return models.ChatResponse("具体指哪个角色？")
            with patch.object(manager, "_chat_impl", side_effect=call) as calls:
                result = await manager.run_turn([models.Message("user", "这是谁？")], reply_policy=policy)
            self.assertEqual(str(result), "" if policy == PROACTIVE_REPLY else "具体指哪个角色？")
            self.assertEqual(calls.call_count, 1 if policy == PROACTIVE_REPLY else 2)

    async def test_social_renderer_keeps_dialogue_instead_of_reference_payloads(self):
        from src.plugins.llm_chat.agent_runner import current_run
        manager = models.ModelManager()
        question = "[引用回复] 雷鸣解释过这个梗 [当前消息] 雷军亲自解释"
        messages = [models.Message("user", "雷鸣是我的群昵称")]
        messages.extend(models.Message("user", "无关人物记忆", is_reference=True) for _ in range(8))
        messages.append(models.Message("user", question))
        async def call(received, **kwargs):
            if kwargs["role"] == "planner":
                current_run.get().response_mode = "social"
                return models.ChatResponse("不要把昵称当成错字")
            recent = next(m.content for m in received if m.content.startswith("最近对话"))
            self.assertIn("雷鸣是我的群昵称", recent)
            self.assertNotIn("无关人物记忆", recent)
            self.assertIn("用户本轮问题：" + question, [m.content for m in received])
            return models.ChatResponse("原来是雷鸣的梗")
        with patch.object(manager, "_chat_impl", side_effect=call):
            await manager.run_turn(messages)

    async def test_factual_and_mixed_renderers_receive_handoff_without_raw_history(self):
        from src.plugins.llm_chat.agent_runner import current_run
        for mode in ("factual", "mixed"):
            with self.subTest(mode=mode):
                manager = models.ModelManager()
                question = "这个说法确认了吗？"
                draft = "目前只有一个渠道报价，还不能确认普遍行情。"
                messages = [models.Message("system", "保留原人设"),
                            models.Message("user", "群友猜测已经普遍涨价"),
                            models.Message("assistant", "之前错误断言：最低价7500"),
                            models.Message("user", question)]
                captured = {}
                async def call(received, **kwargs):
                    captured[kwargs["role"]] = received
                    if kwargs["role"] == "planner":
                        run = current_run.get()
                        run.response_mode = mode
                        run.answer_draft = draft
                        return models._attach_response_context(
                            models.ChatResponse("渠道报价不能代表普遍行情"),
                            model_spec="openai:planner", primary_spec="openai:planner", fallback_attempt=0,
                            role="planner", caller="test:plan", session_id="_unknown_",
                        )
                    return models.ChatResponse(draft)
                with patch.object(manager, "_chat_impl", side_effect=call):
                    result = await manager.run_turn(messages)
                self.assertEqual(captured["planner"][:-1], messages)
                rendered = captured["chat"]
                user_inputs = [m.content for m in rendered if m.role == "user"]
                self.assertEqual(user_inputs, ["用户本轮问题：" + question])
                self.assertEqual(rendered[0].content, "保留原人设")
                payload = json.loads(rendered[-1].content.splitlines()[2])
                self.assertEqual(payload["answer_draft"], draft)
                self.assertNotIn("之前错误断言", "\n".join(m.content for m in rendered))
                self.assertEqual(str(result), draft)

    async def test_execution_plan_can_shorten_selected_detail_with_one_hint(self):
        from src.plugins.llm_chat.agent_runner import current_run
        from src.plugins.llm_chat.reply_policy import DETAILED_REPLY
        manager = models.ModelManager()
        async def call(messages, **kwargs):
            if kwargs["role"] == "planner":
                current_run.get().reply_mode = "sentence"
                return models.ChatResponse("简短回答即可")
            prompt = "\n".join(m.content for m in messages if m.role == "system")
            self.assertIn("通常10—30字", prompt)
            self.assertNotIn("500个字符", prompt)
            self.assertEqual(prompt.count("【本轮回复规则"), 1)
            return models.ChatResponse("好了")
        with patch.object(manager, "_chat_impl", side_effect=call):
            result = await manager.run_turn([
                models.Message("system", DETAILED_REPLY.system_hint() + "保留人设"), models.Message("user", "好了？")
            ], reply_policy=DETAILED_REPLY)
        self.assertFalse(result.reply_policy.detailed)
    async def test_prepared_factual_answer_survives_renderer_failure_or_deadline(self):
        from src.plugins.llm_chat.agent_runner import current_run
        from src.plugins.llm_chat.reply_policy import SHORT_REPLY
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                manager = models.ModelManager()
                async def call(messages, **kwargs):
                    if kwargs["role"] == "planner":
                        runner = current_run.get()
                        runner.response_mode = "factual"
                        runner.answer_draft = "第二赛段已开播，尚未全部上线"
                        runner.reply_mode = "detailed"
                        return models._attach_response_context(
                            models.ChatResponse("已核实；此处包含内部回应方向", usage={"total_tokens": 7}),
                            model_spec="openai:planner", primary_spec="openai:planner", fallback_attempt=0,
                            role="planner", caller="test:plan", session_id="_unknown_",
                        )
                    if timeout:
                        await asyncio.sleep(10)
                    raise RuntimeError("renderer unavailable")
                with patch.object(manager, "_chat_impl", side_effect=call), \
                     patch.object(models.config, "chat_turn_timeout", 0.1):
                    result = await manager.run_turn([models.Message("user", "更新了吗")], reply_policy=SHORT_REPLY)
                self.assertEqual(str(result), "第二赛段已开播，尚未全部上线")
                self.assertEqual(result.agent_run.status, "complete")
                self.assertTrue(result.reply_policy.detailed)
                self.assertEqual(result.usage["total_tokens"], 7)
                self.assertTrue(result.caller.endswith(":render_fallback"))

    async def test_internal_plan_is_never_used_as_fallback(self):
        manager = models.ModelManager()
        with patch.object(manager, "_chat_impl", side_effect=[models.ChatResponse("内部规划：先确认年份"), RuntimeError("failed")]):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await manager.run_turn([models.Message("user", "跌了多少")])

    async def test_execution_mode_expands_chat_but_not_proactive(self):
        from src.plugins.llm_chat.agent_runner import current_run
        from src.plugins.llm_chat.reply_policy import SHORT_REPLY, PROACTIVE_REPLY
        for policy in (SHORT_REPLY, PROACTIVE_REPLY):
            manager = models.ModelManager()
            async def call(messages, **kwargs):
                if kwargs["role"] == "planner":
                    current_run.get().reply_mode = "detailed"
                    return models.ChatResponse("三个机制搭配")
                prompt = "\n".join(m.content for m in messages if m.role == "system")
                self.assertIn("500个字符" if policy == SHORT_REPLY else "一两句自然短话", prompt)
                return models.ChatResponse("第一种搭配。\n第二种搭配。\n第三种搭配。")
            with patch.object(manager, "_chat_impl", side_effect=call):
                result = await manager.run_turn([models.Message("user", "来几个搭配")], reply_policy=policy)
            self.assertEqual(result.reply_policy.detailed, policy == SHORT_REPLY)

    async def test_decoded_semantic_handoff_reaches_chat_and_fallback_unchanged(self):
        # These check transport/priority, not whether a live model will obey it.
        for mode, question, conclusion, requirements, draft in (
            ("factual", "创业板三季度跌了多少", "年份未明确，需要先澄清。",
             ["询问哪一年的第三季度", "不能声称查不了或让用户自己查"], "指哪一年的第三季度？"),
            ("mixed", "我很疲惫，困", "有空可以休息。", ["不要把疲惫归因于游戏"], ""),
            ("factual", "昨天最丐的70都要7500了", "更正前面的报价，把用户报价和市场结论分开。",
             ["7500是用户提供的报价，最低成交价尚未确认"],
             "前面的七千只是参考报价；你说的7500是昨天的渠道报价，我还没确认是不是普遍价。"),
            ("social", "学学大肥鱼", "用户指刚才截图里的AI说话风格，要求模仿这种夸张吐槽。",
             ["接的是模仿风格请求，对象已明确"], ""),
            ("mixed", "我觉得少吃", "区分少吃和硬扛；需要用药时按说明书，留意重复成分。",
             ["少吃不等于一味硬扛", "提醒留意重复成分"], ""),
        ):
            with self.subTest(question=question):
                manager = models.ModelManager()
                planner = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
                payload = json.dumps({"response_mode": mode, "verify_group_identity": False,
                                      "content": conclusion, "reply_requirements": requirements,
                                      "reply_mode": "detailed", "answer_draft": draft})
                chat_one, chat_two = AsyncMock(), AsyncMock()
                chat_one.chat.side_effect = RuntimeError("unavailable")
                chat_two.chat.return_value = models.ChatResponse("自然的回复", usage={"total_tokens": 3})

                def resolve(**kwargs):
                    chain = ["openai:planner"] if kwargs["role"] == "planner" else ["openai:one", "openai:two"]
                    return chain[0], chain

                try:
                    with (
                        patch.object(manager, "_resolve_candidates", side_effect=resolve),
                        patch.object(manager, "get_or_build_adapter", side_effect=[planner, chat_one, chat_two]),
                        patch.object(planner, "_create_completion", AsyncMock(return_value=make_response(
                            FakeToolMessage([], content=payload)))),
                    ):
                        result = await manager.run_turn([models.Message("user", question)], role="chat")
                    self.assertEqual(result.agent_run.reply_requirements, requirements)
                    self.assertEqual(result.agent_run.planning_result, conclusion)
                    self.assertEqual(result.agent_run.reply_mode, "detailed")
                    self.assertEqual(result.agent_run.answer_draft, draft)
                    handoffs = []
                    for adapter in (chat_one, chat_two):
                        messages = adapter.chat.await_args.args[0]
                        handoff = messages[-1]
                        self.assertEqual(handoff.role, "system")
                        data = json.loads(handoff.content.splitlines()[2])
                        self.assertEqual(data["response_mode"], mode)
                        self.assertEqual(data["content"], conclusion)
                        self.assertEqual(data["reply_requirements"], requirements)
                        if draft:
                            self.assertEqual(data["answer_draft"], draft)
                        else:
                            self.assertNotIn("answer_draft", data)
                        handoffs.append(handoff.content)
                        self.assertFalse(adapter.chat.await_args.kwargs["tools_enabled"])
                    self.assertEqual(handoffs[0], handoffs[1])
                    self.assertEqual(str(result), "自然的回复")
                finally:
                    await planner.client.close()

    async def asyncTearDown(self):
        from src.plugins.llm_chat.llm_logger import drain_usage_tasks
        from src.plugins.llm_chat.usage_db import close_db

        await drain_usage_tasks()
        await close_db()

    async def test_planner_owns_tools_and_chat_fallback_does_not_repeat_planning(self):
        manager = models.ModelManager()
        planner = AsyncMock()
        planner.chat.return_value = models.ChatResponse("已核实结论")
        chat_one = AsyncMock()
        chat_one.chat.side_effect = RuntimeError("unavailable")
        chat_two = AsyncMock()
        chat_two.chat.return_value = models.ChatResponse("聊天表达")
        routes = []

        def resolve(**kwargs):
            routes.append(kwargs)
            chain = ["openai:planner"] if kwargs["role"] == "planner" else ["openai:chat-one", "openai:chat-two"]
            return chain[0], chain

        with (
            patch.object(manager, "_resolve_candidates", side_effect=resolve),
            patch.object(manager, "get_or_build_adapter", side_effect=[planner, chat_one, chat_two]),
        ):
            result = await manager.run_turn(
                [models.Message("user", "帮我核实一下")], model="openai:user-chat", role="chat"
            )
        self.assertEqual(str(result), "聊天表达")
        self.assertEqual([r["role"] for r in routes], ["planner", "chat"])
        self.assertIsNone(routes[0]["model"])
        self.assertEqual(routes[1]["model"], "openai:user-chat")
        planner.chat.assert_awaited_once()
        self.assertTrue(planner.chat.await_args.kwargs["tools_enabled"])
        for adapter in (chat_one, chat_two):
            self.assertFalse(adapter.chat.await_args.kwargs["tools_enabled"])
            self.assertIsNone(adapter.chat.await_args.kwargs["required_tool"])
        self.assertEqual(result.role, "chat")
        self.assertEqual(result.agent_run.status, "complete")

    async def test_direct_chat_is_text_only(self):
        manager = models.ModelManager()
        adapter = AsyncMock()
        adapter.chat.return_value = models.ChatResponse("你好")
        with (
            patch.object(manager, "_resolve_candidates", return_value=("openai:chat", ["openai:chat"])),
            patch.object(manager, "get_or_build_adapter", return_value=adapter),
        ):
            result = await manager.chat([models.Message("user", "你好")], role="chat")
        self.assertFalse(adapter.chat.await_args.kwargs["tools_enabled"])
        self.assertFalse(hasattr(result, "agent_run"))

    async def test_media_only_turn_does_not_invent_a_text_followup(self):
        manager = models.ModelManager()
        with patch.object(manager, "_chat_impl", AsyncMock(return_value=models.ChatResponse(""))) as call:
            result = await manager.run_turn([], required_tool="generate_image", media_followup=False)
        self.assertEqual(str(result), "")
        call.assert_awaited_once()
        self.assertEqual(call.await_args.kwargs["role"], "planner")
        self.assertEqual(call.await_args.kwargs["required_tool"], "generate_image")
