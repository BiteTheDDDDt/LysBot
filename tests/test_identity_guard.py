import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import agent_runner, models, tools
from src.plugins.llm_chat.identity_guard import UNCERTAIN, is_identity_question, verify_identity
from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.retrieval import HistorySearch


def response(payload):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))], usage=None)


class IdentityGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.tmp.name) / "history.db"
        self.manager._closing = True
        self.db = await self.manager._get_db()
        await self.db.executemany(
            "INSERT INTO messages_detail(session_id,role,name,sender_id,content,timestamp) VALUES(?,?,?,?,?,?)",
            [
                ("group_7", "user", "甲", 10101, "[当前消息]\n@RiverFox(QQ:20202) R姐要这个吗", "2026-09-01T10:00:00"),
                ("group_7", "user", "RiverFox", 20202, "我想要另一个", "2026-09-01T10:00:01"),
                ("group_7", "user", "WrongBot", 30303, "我是第三方bot", "2026-09-01T10:00:02"),
                (
                    "group_7",
                    "user",
                    "甲",
                    10101,
                    "[引用回复]\n被回复者：RiverFox\n原消息：想吃饭\n[当前消息]\nR姐下楼吃吧",
                    "2026-09-02T10:00:00",
                ),
                ("group_8", "user", "Foreign", 40404, "外群秘密", "2026-09-02T10:00:00"),
            ],
        )
        await self.db.execute(
            "INSERT INTO important_info(session_id,user_id,info_type,content,created_at) VALUES(?,?,?,?,?)",
            ("group_7", 30303, "nickname", "旧记忆错误断言：R姐就是WrongBot", "2026-09-03T10:00:00"),
        )
        await self.db.commit()
        self.run = agent_runner.AgentRunner("group_7")
        self.run.memory_path = self.manager._db_path
        self.run.memory_generation = 0
        self.run.user_question = "R姐是谁"
        self.run.identity_question = True
        self.backend = SimpleNamespace(model_name="test:reviewer")
        self.verdict = {"person_id": "20202", "evidence_ids": ["h:0:messages:1"], "relation": "direct_address"}
        self.run.step = AsyncMock(return_value=response(self.verdict))

    async def asyncTearDown(self):
        from src.plugins.llm_chat.llm_logger import drain_usage_tasks
        from src.plugins.llm_chat.usage_db import close_db

        await self.manager.close()
        await drain_usage_tasks()
        await close_db()
        MemoryManager._instance = None
        self.tmp.cleanup()

    async def lead(self, ref="h:0:messages:1"):
        found = await HistorySearch(self.manager._db_path).read("group_7", ids=[ref])
        self.run.evidence.update({i["id"]: i for i in found["items"]})

    async def verify(self):
        return (await verify_identity(self.run, "一般指WrongBot", self.backend, None, {}))[0]

    async def test_wrong_draft_and_legacy_assertion_cannot_supply_review_evidence(self):
        await self.lead()
        await self.lead("h:0:important_info:1")
        self.assertEqual(await self.verify(), "是RiverFox。")
        sent = self.run.step.await_args.args[2]
        payload = json.loads(sent["messages"][1]["content"])
        self.assertNotIn("旧记忆错误断言", json.dumps(payload["evidence"], ensure_ascii=False))
        self.assertNotIn("tools", sent)
        self.assertEqual(self.run.identity_verdict["person_id"], "20202")

    async def test_reply_target_resolves_to_account_without_hardcoded_alias(self):
        await self.lead("h:0:messages:4")
        self.run.step.return_value = response(
            {**self.verdict, "evidence_ids": ["h:0:messages:4"], "relation": "reply_address", "answer": "WrongBot"}
        )
        self.assertEqual(await self.verify(), "是RiverFox。")
        self.run.user_question = "R姐的QQ号是谁"
        self.assertEqual(await self.verify(), "是 QQ 20202。")

    async def test_invalid_or_unrelated_account_and_citation_fail_closed(self):
        await self.lead()
        for verdict in [
            {**self.verdict, "person_id": "30303"},
            {**self.verdict, "evidence_ids": ["h:0:important_info:1"]},
            {**self.verdict, "evidence_ids": ["h:0:messages:5"]},
            {**self.verdict, "evidence_ids": []},
            {**self.verdict, "relation": "mentioned_someone"},
            [],
        ]:
            self.run.step.return_value = response(verdict)
            self.assertEqual(await self.verify(), UNCERTAIN)

    async def test_only_plain_mentions_or_deleted_sources_cannot_verify(self):
        await self.lead("h:0:messages:2")
        self.assertEqual(await self.verify(), UNCERTAIN)
        self.run.step.assert_not_awaited()
        await self.lead()
        await self.db.execute("DELETE FROM messages_detail WHERE id=1")
        await self.db.commit()
        self.assertEqual(await self.verify(), UNCERTAIN)

    async def test_ambiguous_reply_name_does_not_select_an_account(self):
        await self.lead("h:0:messages:4")
        await self.db.execute(
            "INSERT INTO messages_detail(session_id,role,name,sender_id,content,timestamp) VALUES(?,?,?,?,?,?)",
            ("group_7", "user", "RiverFox", 90909, "同名用户", "2026-09-03T10:00:00"),
        )
        await self.db.commit()
        self.run.step.return_value = response(
            {**self.verdict, "evidence_ids": ["h:0:messages:4"], "relation": "reply_address"}
        )
        self.assertEqual(await self.verify(), UNCERTAIN)

    async def test_final_adapter_path_replaces_wrong_identity_with_verified_account(self):
        from test_tool_rounds import FakeToolMessage, make_response, make_tool_call

        found = await HistorySearch(self.manager._db_path).read("group_7", ids=["h:0:messages:1"])
        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        manager = models.ModelManager()
        completions = AsyncMock(
            side_effect=[
                make_response(FakeToolMessage([make_tool_call("a", "recall_history", {"exact_phrase": "R姐"})])),
                make_response(FakeToolMessage([], content="一般指WrongBot")),
                response(self.verdict),
            ]
        )
        try:
            with (
                patch.object(agent_runner, "_runtime_path", self.manager._db_path),
                patch.object(manager, "_resolve_candidates", return_value=("openai:test", ["openai:test"])),
                patch.object(manager, "get_or_build_adapter", return_value=adapter),
                patch.object(adapter, "_create_completion", completions),
                patch.object(tools, "execute_tool", AsyncMock(return_value=json.dumps(found))),
            ):
                completions.side_effect = [*completions.side_effect, make_response(FakeToolMessage([], content="又猜错人"))]
                answer = await manager.run_turn(
                    [models.Message(role="user", content="R姐是谁", sender_id=10101)], role="chat", session_id="group_7"
                )
            self.assertEqual(str(answer), "是RiverFox。")
            self.assertEqual(answer.agent_run.identity_verdict["status"], "verified")
            self.assertEqual(completions.await_count, 4)
        finally:
            await adapter.client.close()

    async def test_quoted_display_name_does_not_replace_successful_research(self):
        from test_tool_rounds import FakeToolMessage, make_response, make_tool_call
        from src.plugins.llm_chat.chat import ResolvedIncomingMessage

        question = "【群员-Clam】 " + ResolvedIncomingMessage(
            plain_text="什么派 我了解下",
            context_blocks=[
                "[引用回复]\n被回复者：是谁还没发roc...\n"
                "原消息：实际上政治学研究有一个专门的流派就是通过软件模拟的。。"
            ],
        ).compose()
        draft = "这是计算政治学中的基于主体建模（ABM），是一种计算模拟方法，并非单一学派。"
        adapter = models.OpenAICompatibleAdapter("openai:test", "dummy", "https://example.test/v1", "test")
        manager = models.ModelManager()
        completions = AsyncMock(side_effect=[
            make_response(FakeToolMessage([make_tool_call("search", "search_web", {
                "query": "political science agent-based modeling computational simulation", "limit": 5,
            })])),
            make_response(FakeToolMessage([], content=draft)),
            make_response(FakeToolMessage([], content=draft)),
        ])
        search = json.dumps({
            "status": "ok", "items": [],
            "search_summary": {"content": draft, "provenance": "native_search_answer"},
        })
        try:
            with (
                patch.object(agent_runner, "_runtime_path", self.manager._db_path),
                patch.object(manager, "_resolve_candidates", return_value=("openai:test", ["openai:test"])),
                patch.object(manager, "get_or_build_adapter", return_value=adapter),
                patch.object(adapter, "_create_completion", completions),
                patch.object(tools, "execute_tool", AsyncMock(return_value=search)) as execute,
                patch("src.plugins.llm_chat.identity_guard.verify_identity", wraps=verify_identity) as verify,
            ):
                answer = await manager.run_turn(
                    [models.Message(role="user", content=question, sender_id=10101)],
                    role="chat", session_id="group_7",
                )
            self.assertEqual(str(answer), draft)
            self.assertEqual(answer.agent_run.planning_result, draft)
            self.assertFalse(answer.agent_run.identity_question)
            self.assertIsNone(answer.agent_run.identity_verdict)
            verify.assert_not_called()
            self.assertEqual(execute.await_args.args[0], "search_web")
            self.assertEqual(completions.await_count, 3)
        finally:
            await adapter.client.close()

    async def test_identity_correction_keeps_previous_question_and_evidence(self):
        with patch.object(agent_runner, "_runtime_path", self.manager._db_path):
            previous = agent_runner.AgentRunner("group_7")
            await previous.start([models.Message("user", "本群R姐是谁", sender_id=10101)])
            found = await HistorySearch(self.manager._db_path).read("group_7", ids=["h:0:messages:1"])
            previous.evidence = {item["id"]: item for item in found["items"]}
            previous.tool_calls = 1
            previous.planning_result = "需要核对R姐对应的群成员"
            await previous.finish("complete")
            current = agent_runner.AgentRunner("group_7")
            try:
                await current.start([models.Message("user", "我是问本群那个", sender_id=10101)])
                self.assertTrue(current.identity_question)
                self.assertIn("本群R姐是谁", current.user_question)
                self.assertIn("本轮纠正：我是问本群那个", current.user_question)
                self.assertIn("h:0:messages:1", current.evidence)
            finally:
                await current.finish("complete")

    def test_identity_trigger_does_not_match_ordinary_search_or_statistics(self):
        self.assertTrue(is_identity_question("【群员-A】 本群R姐是谁"))
        self.assertFalse(is_identity_question("统计本群今天的消息数"))
        self.assertFalse(is_identity_question("查一下上次推荐的显卡价格"))

    def test_identity_fallback_uses_current_utterance_not_author_or_quoted_question(self):
        for text in (
            "【群员-是谁还没发roc... QQ:10101】 什么派 我了解下",
            "【群员-A】 [引用回复]\n被回复者：是谁还没发roc...\n原消息：软件模拟研究\n\n[当前消息]\n什么派 我了解下",
            "【群员-A】 [引用回复]\n被回复者：乙\n原消息：R姐是谁\n\n[当前消息]\n你为什么这么说",
            "【群员-A】 [引用回复]\n被回复者：是谁还没发roc...\n原消息：软件模拟研究",
            "【群员-A】 [合并转发]\nQQ用户：R姐是谁",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_identity_question(text))
        for text in (
            "【群员-是谁还没发roc...】 本群R姐是谁",
            "【群员-A】 [引用回复]\n被回复者：乙\n原消息：R姐说过\n\n[当前消息]\n她是哪个群友",
            "【群员-A】 [引用回复]\n被回复者：乙\n原消息：旧问题\r\n\r\n[当前消息]\r\n本群谁叫R姐",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_identity_question(text))
