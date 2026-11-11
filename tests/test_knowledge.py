import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import capabilities, tools
from src.plugins.llm_chat.config import config, get_tool_usage_prompt
from src.plugins.llm_chat.knowledge import KnowledgeStore, refresh_evidence


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "knowledge"
        self.root.mkdir()
        self.path = self.root / "project.md"
        self.path.write_text("# 项目介绍\nLysBot 源码：https://example.test/bot\n" + "分段正文。" * 1200, encoding="utf-8")
        (self.root / "skills").mkdir()
        (self.root / "skills" / "deploy.md").write_text("# 部署方法\n这是静态 skill，使用 Docker Compose。", encoding="utf-8")
        self.store = KnowledgeStore(self.root)

    def test_search_browse_and_read_have_static_provenance(self):
        found = self.store.search("LysBot 源码")
        self.assertEqual(found["status"], "ok")
        item = found["items"][0]
        self.assertEqual(item["path"], "project.md")
        self.assertEqual(item["provenance"], "static_knowledge")
        self.assertEqual(item["read_state"], "search_snippet")
        read = self.store.read(item["document_id"])
        self.assertEqual(read["items"][0]["read_state"], "original")
        self.assertNotIn(self.tmp.name, json.dumps(read))
        first = self.store.search(limit=1)
        second = self.store.search(limit=1, offset=first["next_offset"])
        self.assertNotEqual(first["items"][0]["document_id"], second["items"][0]["document_id"])
        self.assertIsNone(second["next_offset"])
        self.assertEqual(self.store.search("不存在的术语xyz")["items"], [])

    def test_long_unicode_document_pages_reconstruct_exact_text(self):
        doc = self.store.search("LysBot")["items"][0]["document_id"]
        offset, content, ids = 0, [], set()
        while True:
            page = self.store.read(doc, offset=offset, max_chars=97)
            item = page["items"][0]
            self.assertNotIn(item["id"], ids)
            ids.add(item["id"])
            content.append(item["excerpt"])
            if page["next_offset"] is None:
                break
            self.assertGreater(page["next_offset"], offset)
            offset = page["next_offset"]
        self.assertEqual("".join(content), self.path.read_text(encoding="utf-8"))

    def test_image_refs_only_resolve_local_document_attachments(self):
        (self.root.parent / "outside.png").write_bytes(b"private")
        (self.root / "ref.png").write_bytes(b"reference image")
        self.path.write_text("# 角色\n![合法参考](ref.png)\n![越界](../outside.png)\n![远程](https://example.test/ref.png)\n", encoding="utf-8")
        doc = self.store.search("角色")["items"][0]["document_id"]
        refs = self.store.read(doc)["items"][0]["image_refs"]
        self.assertEqual([ref["alt"] for ref in refs], ["合法参考"])
        self.assertEqual(base64.b64decode(self.store.image_base64(refs[0]["id"])), b"reference image")
        with self.assertRaises(ValueError):
            self.store.image_base64(refs[0]["id"].rsplit(":", 1)[0] + ":1")
        self.path.write_text("# 更新后的角色", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.image_base64(refs[0]["id"])

    def test_revision_and_restored_evidence_require_current_document(self):
        doc = self.store.search("LysBot")["items"][0]["document_id"]
        item = self.store.read(doc)["items"][0]
        with patch.object(config, "knowledge_dir", str(self.root)):
            self.assertIsNotNone(refresh_evidence(item))
            self.path.write_text("# 修改后的项目\n另一个地址", encoding="utf-8")
            self.assertEqual(self.store.read(doc)["error"]["code"], "document_changed_search_again")
            self.assertIsNone(refresh_evidence(item))
        self.path.unlink()
        self.assertEqual(self.store.read(doc)["error"]["code"], "document_unavailable_search_again")

    def test_read_rejects_arbitrary_paths_hidden_files_and_traversal(self):
        secret = self.root.parent / "secret.md"
        secret.write_text("PRIVATE", encoding="utf-8")
        (self.root / ".private.md").write_text("PRIVATE", encoding="utf-8")
        (self.root / ".env").write_text("PRIVATE", encoding="utf-8")
        revision = hashlib.sha256(b"PRIVATE").hexdigest()
        for path in ["../secret.md", "/secret.md", "a/../../secret.md", "a\\secret.md", ".private.md", ".env", "C:/secret.md", "project.md:stream"]:
            encoded = base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")
            result = self.store.read(f"k:{encoded}:{revision}")
            self.assertEqual(result["status"], "invalid_arguments", path)
            self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertEqual(len(self.store.search()["items"]), 2)
        self.assertEqual(self.store.read(str(secret))["status"], "invalid_arguments")

    def test_symlink_files_and_directories_are_not_read(self):
        outside = self.root.parent / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text("PRIVATE", encoding="utf-8")
        try:
            (self.root / "link.md").symlink_to(outside / "secret.md")
            (self.root / "linked").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable")
        self.assertEqual(len(self.store.search()["items"]), 2)
        revision = hashlib.sha256(b"PRIVATE").hexdigest()
        for path in ["link.md", "linked/secret.md"]:
            encoded = base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")
            self.assertEqual(self.store.read(f"k:{encoded}:{revision}")["error"]["code"], "linked_document_not_allowed")

    def test_invalid_arguments_and_scan_limits_are_explicit(self):
        for args in [{"limit": True}, {"offset": -1}, {"query": []}, {"query": "x" * 501}, {"limit": 21}]:
            self.assertEqual(self.store.search(**args)["status"], "invalid_arguments")
        doc = self.store.search()["items"][0]["document_id"]
        for args in [{"offset": True}, {"offset": -1}, {"max_chars": 6001}, {"max_chars": 0}]:
            self.assertEqual(self.store.read(doc, **args)["status"], "invalid_arguments")
        (self.root / "bad.md").write_bytes(b"\xff")
        (self.root / "binary.md").write_bytes(b"\0")
        with patch("src.plugins.llm_chat.knowledge.MAX_FILES", 1):
            result = self.store.search()
            self.assertEqual(result["status"], "partial")
            self.assertIn("knowledge_scan_limit", result["warnings"])
            self.assertEqual(len(result["items"]), 1)
        with patch("src.plugins.llm_chat.knowledge.MAX_FILE_BYTES", 20):
            self.assertIn("knowledge_document_skipped", self.store.search()["warnings"])
        missing = KnowledgeStore(self.root / "missing").search()
        self.assertIn("knowledge_directory_missing", missing["warnings"])


class KnowledgeOverviewTests(unittest.TestCase):
    def test_fixed_storage_metadata_and_configured_categories_need_no_directory_scan(self):
        with (
            patch.object(config, "knowledge_dir", "missing-knowledge-directory"),
            patch.object(config, "knowledge_overview", "数据库术语；游戏机制；故障排查方法"),
            patch.object(KnowledgeStore, "_documents", side_effect=AssertionError("no dynamic catalog")),
        ):
            prompt = get_tool_usage_prompt()
        self.assertIn("聊天原文及作者、时间、引用关系", prompt)
        self.assertIn("长期记忆及其来源", prompt)
        self.assertIn("数据库术语；游戏机制；故障排查方法", prompt)
        self.assertIn("不表示具体资料一定存在", prompt)

    def test_disabling_static_knowledge_keeps_db_metadata_without_static_categories(self):
        with patch.object(config, "knowledge_dir", ""), patch.object(config, "knowledge_overview", "custom-category"):
            prompt = get_tool_usage_prompt()
        self.assertIn("DB（动态知识", prompt)
        self.assertNotIn("custom-category", prompt)
        self.assertNotIn("search_knowledge", prompt)


class KnowledgeToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_tools_are_available_without_database_or_special_model(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "project.md").write_text("# 项目\n源码地址 example.test", encoding="utf-8")
            with patch.object(config, "knowledge_dir", directory):
                names = {t["function"]["name"] for t in tools.get_chat_tools()}
                self.assertTrue({"search_knowledge", "read_knowledge"} <= names)
                self.assertTrue({"search_knowledge", "read_knowledge"} <= capabilities.READ_TOOLS)
                self.assertTrue({"search_knowledge", "read_knowledge"}.isdisjoint(capabilities.MEMORY_READ_TOOLS))
                self.assertIn("静态知识", get_tool_usage_prompt())
                result = json.loads(await tools.execute_tool("search_knowledge", {"query": "源码"}))
                read = json.loads(await tools.execute_tool("read_knowledge", {"document_id": result["items"][0]["document_id"]}))
                self.assertEqual(read["status"], "ok")
            with patch.object(config, "knowledge_dir", ""):
                self.assertNotIn("search_knowledge", {t["function"]["name"] for t in tools.get_chat_tools()})
                result = json.loads(await tools.execute_tool("search_knowledge", {}))
                self.assertEqual(result["status"], "unavailable")
