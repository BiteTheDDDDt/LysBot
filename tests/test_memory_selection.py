import unittest
from types import SimpleNamespace

import nonebot

nonebot.init()

from src.plugins.llm_chat.memory_selection import parse_summary_selection


class MemorySelectionTests(unittest.TestCase):
    def setUp(self):
        self.messages = [SimpleNamespace(role="user", is_thinking=False) for _ in range(6)]

    def test_no_durable_fact_skips_second_read(self):
        summary, selected = parse_summary_selection('{"summary":"讨论游戏","memory_candidates":[]}', self.messages)
        self.assertEqual(summary, "讨论游戏")
        self.assertEqual(selected, [])

    def test_candidates_keep_original_objects_and_nearby_context(self):
        _, selected = parse_summary_selection('{"summary":"用户换显卡","memory_candidates":[3,3]}', self.messages)
        self.assertEqual(selected, self.messages[1:4])

    def test_malformed_response_does_not_silently_drop_memory(self):
        for text in ['{"summary":"a","memory_candidates":[90]}', '{"summary":"a"}', '{broken', '{"summary":"","memory_candidates":[]}']:
            with self.assertRaises(ValueError):
                parse_summary_selection(text, self.messages)
        self.assertEqual(parse_summary_selection("旧格式摘要", self.messages)[1], self.messages)

    def test_assistant_and_thinking_cannot_nominate_facts(self):
        self.messages[2].role = "assistant"
        self.messages[3].is_thinking = True
        _, selected = parse_summary_selection('{"summary":"a","memory_candidates":[3,4]}', self.messages)
        self.assertEqual(selected, [])
