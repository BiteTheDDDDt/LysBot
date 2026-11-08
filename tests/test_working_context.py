import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import agent_runner, models
from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.memory import MemoryManager
from src.plugins.llm_chat.retrieval import HistorySearch
from src.plugins.llm_chat.working_context import restore_context


class WorkingContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_jrrp_delivery_and_basis_are_restored_for_followup(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from src.plugins.llm_chat import commands, followups
        class GroupEvent(SimpleNamespace):
            pass
        event = GroupEvent(user_id=101, group_id=7, sender=SimpleNamespace(card='', nickname='甲'))
        async def generate(messages, **kwargs):
            run = agent_runner.AgentRunner(kwargs['session_id'])
            await run.start(messages)
            run.planning_result = '用本人旧评价“模型凑合用”作玩笑，与49分没有因果关系'
            await run.finish('complete')
            result = models.ChatResponse('49，跟那个模型一样凑合用', model_spec='test:actual-fallback')
            result.agent_run = run
            return result
        with (
            patch.object(agent_runner, '_runtime_path', self.manager._db_path),
            patch.object(commands, 'GroupMessageEvent', GroupEvent),
            patch.object(commands, '_calculate_2bot_jrrp_value', return_value=49),
            patch.object(commands, '_build_jrrp_memory_context', AsyncMock(return_value='旧记忆：本人说模型凑合用')),
            patch.object(commands, '_record_command_interaction', AsyncMock()) as record,
            patch('src.plugins.llm_chat.model_identity.update_group_card_for_response', AsyncMock()) as card,
            patch.object(commands.model_manager, 'run_turn', side_effect=generate),
            patch.object(commands.jrrp_cmd, 'send', AsyncMock(return_value={'message_id': 99})),
        ):
            bot = SimpleNamespace(self_id='42')
            await commands.handle_jrrp(event, bot)
            self.assertEqual(card.await_args.args[2].model_spec, 'test:actual-fallback')
            self.assertIs(card.await_args.args[0], bot)
            self.assertEqual(record.await_args.kwargs['model_spec'], 'test:actual-fallback')
        try:
            self.assertTrue(followups.eligible(7, 101))
            next_run = self.runner()
            await restore_context(next_run)
            note = json.loads(next_run.transcript[-1]['content'].split('\n', 1)[1])[0]
            self.assertEqual(note['answer'], '49，跟那个模型一样凑合用')
            self.assertIn('本人旧评价', note['planning_result'])
            self.assertIn('与聊天内容无因果关系', note['planning_result'])
        finally:
            followups.windows.pop(7, None)

    def test_jrrp_prefers_sender_id_over_changed_or_shared_nickname(self):
        from src.plugins.llm_chat.commands import _is_jrrp_target_group_message
        self.assertTrue(_is_jrrp_target_group_message(models.Message('user', '本人发言', name='旧昵称', sender_id=101), '新昵称', 101))
        self.assertFalse(_is_jrrp_target_group_message(models.Message('user', '别人发言', name='同名', sender_id=202), '同名', 101))

    async def test_restore_only_claims_confirmed_delivery(self):
        await self.save()
        r = self.runner()
        await restore_context(r)
        notes = json.loads(r.transcript[-1]["content"].split("\n", 1)[1])
        self.assertEqual(notes[0]["answer"], "")
        self.assertIn("直接称呼", notes[0]["planning_result"])
        state = {"final_answer": {"answer": "没发出去的全文"}, "actions": {
            "a": {"status": "sent", "text": "实际发出的短回复"},
            "b": {"status": "unknown", "text": "发送结果未知"},
        }}
        self.assertEqual(agent_runner.confirmed_reply(state), "实际发出的短回复")
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        MemoryManager._instance = None
        self.manager = MemoryManager()
        self.manager._db_path = Path(self.tmp.name) / "history.db"
        self.manager._closing = True
        self.db = await self.manager._get_db()
        await self.db.execute(
            "INSERT INTO messages_detail(session_id,role,name,sender_id,content,timestamp) VALUES(?,?,?,?,?,?)",
            ("group_7", "user", "甲", 101, "@RiverFox(QQ:20202) R姐下楼吃饭", "2026-09-01T10:00:00"),
        )
        await self.db.commit()
        found = await HistorySearch(self.manager._db_path).read("group_7", ids=["h:0:messages:1"])
        self.item = found["items"][0]

    async def asyncTearDown(self):
        await self.manager.close()
        MemoryManager._instance = None
        self.tmp.cleanup()

    async def test_reply_model_survives_reload_and_history_retrieval(self):
        from src.plugins.llm_chat.context_builder import _format_history_message
        with patch.object(self.manager, '_schedule_summary'):
            msg = await self.manager.add_message(0, 'assistant', '这是回答', 7, model_spec='test:actual-fallback')
        self.manager._cache.pop('group_7', None)
        await self.manager._load_session('group_7')
        loaded = self.manager._cache['group_7'][-1]
        self.assertEqual(loaded.model_spec, 'test:actual-fallback')
        self.assertIn('test:actual-fallback', _format_history_message(loaded).content)
        found = await HistorySearch(self.manager._db_path).read('group_7', ids=[f'h:0:messages:{msg.source_id}'])
        self.assertEqual(found['items'][0]['model_spec'], 'test:actual-fallback')
        old = _format_history_message(models.Message('assistant', '旧回答'))
        self.assertIn('未记录', old.content)

    async def save(self, rid="previous", uid=101, sid="group_7", age=0, generation=0, item=None):
        state = {
            "requester_id": uid, "user_question": "R姐是谁", "planning_result": "直接称呼支持身份，外号由来未知",
            "final_answer": {"answer": "是RiverFox。"},
            "evidence": {"h:0:messages:1": item or self.item},
            "identity_verdict": {"status": "verified", "person_id": "20202", "evidence_ids": ["h:0:messages:1"]},
            "actions": {"old": {"status": "sent"}}, "cache": {"old": "stale"},
        }
        await self.db.execute("INSERT INTO agent_runs VALUES(?,?,?,?,?,?)", (
            rid, sid, generation, "complete", json.dumps(state), time.time() - age,
        ))
        await self.db.commit()

    def runner(self):
        r = agent_runner.AgentRunner("group_7")
        r.memory_path = self.manager._db_path
        r.memory_generation = 0
        r.requester_id = 101
        return r

    async def test_static_evidence_is_revalidated_and_removed_after_file_changes(self):
        from src.plugins.llm_chat.knowledge import KnowledgeStore
        root = Path(self.tmp.name) / "knowledge"
        root.mkdir()
        path = root / "project.md"
        path.write_text("# 项目信息\n静态来源", encoding="utf-8")
        store = KnowledgeStore(root)
        doc = store.search()["items"][0]["document_id"]
        item = store.read(doc)["items"][0]
        state = {"requester_id": 101, "evidence": {item["id"]: item}, "actions": {}}
        await self.db.execute("INSERT INTO agent_runs VALUES(?,?,?,?,?,?)", (
            "static-previous", "group_7", 0, "complete", json.dumps(state), time.time(),
        ))
        await self.db.commit()
        with patch.object(config, "knowledge_dir", str(root)):
            run = self.runner()
            await restore_context(run)
            self.assertIn(item["id"], run.evidence)
            self.assertEqual(run.evidence[item["id"]]["provenance"], "static_knowledge")
            path.write_text("# 已更新\n新版本资料", encoding="utf-8")
            run = self.runner()
            await restore_context(run)
            self.assertNotIn(item["id"], run.evidence)

    async def test_plain_followup_restores_checked_evidence_without_replaying_actions(self):
        await self.save()
        with patch.object(agent_runner, "_runtime_path", self.manager._db_path):
            r = agent_runner.AgentRunner("group_7")
        await r.start([models.Message("user", "原因是什么", sender_id=101)])
        try:
            self.assertEqual(r.context_run_ids, ["previous"])
            self.assertIn("h:0:messages:1", r.evidence)
            self.assertIn("外号由来未知", r.transcript[-1]["content"])
            self.assertEqual(r.evidence["h:0:messages:1"]["read_state"], "original")
            self.assertFalse(r.actions)
            self.assertFalse(r.cache)
            self.assertEqual(r.tool_calls, 0)
            self.assertIsNone(r.identity_verdict)  # Explanation must not be locked to a bare name.
        finally:
            await r.finish("complete")

    async def test_other_user_group_expired_and_reset_generation_are_not_loaded(self):
        await self.save("other-user", uid=202)
        await self.save("other-group", sid="group_8")
        await self.save("expired", age=3600)
        await self.save("reset", generation=1)
        r = self.runner()
        await restore_context(r)
        self.assertFalse(r.transcript)
        self.assertFalse(r.evidence)

    async def test_group_followup_can_restore_another_members_delivered_reply_and_errors(self):
        await self.save("other-user", uid=202)
        row = await (await self.db.execute("SELECT state_json FROM agent_runs WHERE id='other-user'")).fetchone()
        state = json.loads(row[0])
        state["actions"] = {"reply": {"status": "sent", "text": "图片发送失败"}}
        state["api_errors"] = [{"operation": "image", "code": "invalid_size", "message": "图片尺寸超限"}]
        await self.db.execute("UPDATE agent_runs SET state_json=? WHERE id='other-user'", (json.dumps(state),))
        await self.db.commit()
        r = self.runner()
        await restore_context(r)
        notes = json.loads(r.transcript[-1]["content"].split("\n", 1)[1])
        self.assertEqual(notes[0]["answer"], "图片发送失败")
        self.assertEqual(notes[0]["api_errors"][0]["code"], "invalid_size")
        self.assertFalse(r.actions)

    async def test_changed_or_deleted_evidence_does_not_preserve_verified_identity(self):
        await self.save()
        for sql in ("UPDATE messages_detail SET content='修改后的消息' WHERE id=1", "DELETE FROM messages_detail WHERE id=1"):
            await self.db.execute(sql)
            await self.db.commit()
            r = self.runner()
            await restore_context(r)
            self.assertFalse(r.evidence)
            payload = json.loads(r.transcript[-1]["content"].split("\n", 1)[1])
            self.assertIsNone(payload[0]["identity"])
            self.assertEqual(payload[0]["sources"], [])

    async def test_context_is_bounded_and_inherited_evidence_keeps_original_expiry(self):
        for i in range(5):
            await self.save(str(i), item={**self.item, "context_origin_at": time.time() - 3600})
        r = self.runner()
        with patch.object(config, "agent_context_carry_chars", 1800):
            await restore_context(r)
        self.assertLessEqual(len(r.context_run_ids), 3)
        self.assertFalse(r.evidence)
        payload = r.transcript[-1]["content"].split("\n", 1)[1]
        self.assertLess(len(payload), 1850)

    async def test_web_followup_keeps_source_id_but_does_not_claim_freshness(self):
        await self.save()
        row = await (await self.db.execute("SELECT state_json FROM agent_runs WHERE id='previous'")).fetchone()
        state = json.loads(row[0])
        state["evidence"] = {"w:old": {"excerpt": "上轮网页内容", "read_state": "original"}}
        state["sources"] = {"w:old": {"id": "w:old", "url": "https://example.org/article",
                                          "created_at": time.time(), "body": "不应整页装入上下文"}}
        state["identity_verdict"] = None
        await self.db.execute("UPDATE agent_runs SET state_json=? WHERE id='previous'", (json.dumps(state),))
        await self.db.commit()
        r = self.runner()
        await restore_context(r)
        self.assertEqual(r.evidence["w:old"]["read_state"], "recovered_lead")
        self.assertNotIn("body", r.sources["w:old"])
        self.assertIn("previous_turn_not_refetched", r.transcript[-1]["content"])

    async def test_native_search_answer_and_all_citations_survive_followup(self):
        from unittest.mock import AsyncMock
        from src.plugins.llm_chat.web_search import register_url
        run = self.runner()
        first = register_url(run, "https://example.org/prices", title="Price table")
        second = register_url(run, "https://example.org/regions", title="Regions")
        answer = "输入每百万2美元；欧洲地区另外增加10%，这一条件来自第二个来源。"
        payload = {"status": "ok", "items": [
            {"id": first, "kind": "web", "excerpt": "Price table", "url": run.sources[first]["url"],
             "read_state": "search_snippet", "source_ids": [first]},
            {"id": second, "kind": "web", "excerpt": "Regions", "url": run.sources[second]["url"],
             "read_state": "search_snippet", "source_ids": [second]},
        ], "search_summary": {"content": answer, "query": "定价", "searched_at": "2026-10-21T00:00:00+08:00",
                              "related_source_ids": [first, second]}}
        await run.execute("search_web", '{"query":"定价"}', AsyncMock(return_value=json.dumps(payload)))
        await run.finish("complete", "输入2美元 https://example.org/regions")
        self.assertEqual(run.final_answer["evidence_ids"], [second])
        self.assertEqual(run.final_answer["read_evidence_ids"], [first])
        fresh = self.runner()
        await restore_context(fresh)
        note = json.loads(fresh.transcript[-1]["content"].split("\n", 1)[1])[0]
        summary = next(source for source in note["sources"] if source["id"] == first)
        self.assertEqual(summary["excerpt"], answer)
        self.assertEqual(summary["provenance"], "native_search_answer")
        self.assertEqual({source["id"] for source in summary["related_sources"]}, {first, second})
        self.assertEqual(fresh.evidence[first]["read_state"], "native_search")
        self.assertEqual(summary["searched_at"], payload["search_summary"]["searched_at"])
        # An older source cannot be silently detached from a composite answer.
        run.sources[second]["created_at"] = time.time() - 3600
        await run.checkpoint()
        expired = self.runner()
        await restore_context(expired)
        self.assertNotIn(first, expired.evidence)

    async def test_title_only_search_result_does_not_replace_saved_search_answer(self):
        from unittest.mock import AsyncMock
        run = self.runner()
        run.evidence["w:old"] = {"id": "w:old", "read_state": "native_search", "excerpt": "完整查证结果"}
        payload = {"status": "ok", "items": [{"id": "w:old", "read_state": "search_snippet", "excerpt": "标题"}]}
        await run.execute("search_web", '{"query":"补充"}', AsyncMock(return_value=json.dumps(payload)))
        self.assertEqual(run.evidence["w:old"]["excerpt"], "完整查证结果")
