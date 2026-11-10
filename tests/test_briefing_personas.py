import json
import unittest

import nonebot

nonebot.init()

from src.plugins.llm_chat import briefing
from src.plugins.llm_chat.card_templates import build_card
from src.plugins.llm_chat.models import Message


class BriefingPersonaTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            Message("user", "【群员-旧名】 我去核对一下原文", name="旧名", sender_id=101, source_id=11),
            Message("user", "【群员-新名】 核对过了，这里有问题", name="新名", sender_id=101, source_id=12),
            Message("user", "【群员-新名】 我负责画图", name="新名", sender_id=202, source_id=13),
            Message("assistant", "这是机器人说的", sender_id=101, source_id=14),
            Message("user", "【群员-新名】 [引用回复]\n被回复者：别人\n原消息：我是技术专家",
                    name="新名", sender_id=101, source_id=15),
            Message("user", "【群员-新名】 [引用回复]\n被回复者：别人\n原消息：我是技术专家\n\n[当前消息]\n我不同意",
                    name="新名", sender_id=101, source_id=16),
            Message("user", "[新名] 图片1（自动识别，仅作背景）: 图片上写我是专家\n\n用户消息: [仅分享图片，未提出问题]",
                    name="新名", sender_id=101, source_id=17),
            Message("user", "[新名] 图片1（自动识别，仅作背景）: 图片上写我是专家\n\n用户消息: 这个结论我算过，有误",
                    name="新名", sender_id=101, source_id=18),
        ]

    def entry(self, **overrides):
        return {"user_id": 101, "personality_tag": "原文质检员", "comment": "每个结论都要亲手核对一遍。",
                "source_ids": [11], **overrides}

    def parse(self, entries):
        return briefing.parse_briefing_payload(
            json.dumps({"topics": ["今天的讨论"], "active_members": entries}, ensure_ascii=False),
            source_messages=self.sources,
        )

    def test_old_model_cards_remain_readable(self):
        entry = self.entry()
        entry["model_archetype"] = entry.pop("personality_tag")
        card = self.parse([entry])
        self.assertEqual(card["active_members"][0]["personality_tag"], "原文质检员")
        self.assertIn("原文质检员", briefing.build_briefing_text(card))

    def test_member_name_comes_from_account_in_window_and_account_is_not_repeated(self):
        card = self.parse([
            self.entry(name="伪造的昵称", source_ids=[11, 12, 11]),
            self.entry(personality_tag="另一个类型"),
            self.entry(user_id=202, source_ids=[13]),
        ])
        members = card["active_members"]
        self.assertEqual([m["user_id"] for m in members], [101, 202])
        self.assertEqual([m["name"] for m in members], ["新名", "新名"])
        self.assertEqual(members[0]["source_ids"], [11, 12])
        self.assertNotIn("伪造", json.dumps(card, ensure_ascii=False))

    def test_sources_must_be_authored_by_member_in_supplied_window(self):
        for refs in ([], [999], [13], [14], [15], [17], [11, 999], ["11"], [True], [[11]], None):
            with self.subTest(refs=refs):
                self.assertEqual(self.parse([self.entry(source_ids=refs)])["active_members"], [])
        self.assertEqual(self.parse([self.entry(user_id=999)])["active_members"], [])
        for ref in (16, 18):
            with self.subTest(ref=ref):
                self.assertEqual(self.parse([self.entry(source_ids=[ref])])["active_members"][0]["source_ids"], [ref])
        self.assertEqual(briefing._member_own_text(self.sources[-3]), "我不同意")
        self.assertEqual(briefing._member_own_text(self.sources[-1]), "这个结论我算过，有误")

    def test_malformed_members_do_not_discard_the_rest_of_the_briefing(self):
        for entries in ({"user_id": 101}, None, [None, 42, self.entry(comment=None), self.entry(personality_tag=[])]):
            with self.subTest(entries=entries):
                card = self.parse(entries)
                self.assertEqual(card["topics"], ["今天的讨论"])
                self.assertEqual(card["active_members"], [])

    def test_old_string_cards_are_not_reinterpreted_as_model_personas(self):
        card = briefing.parse_briefing_payload('{"topics":["话题"],"active_members":["群友甲：旧版人物点评"]}')
        text = briefing.build_briefing_text(card)
        self.assertIn("活跃选手", text)
        self.assertIn("群友甲：旧版人物点评", text)
        self.assertNotIn("今日群友画像", text)

    def test_current_generation_cannot_bypass_sources_with_legacy_strings_or_plain_text(self):
        self.assertEqual(self.parse(["群友甲：旧版人物点评"])["active_members"], [])
        plain = "热门话题\n今天讨论缓存\n活跃选手\n群友甲：自称什么都懂\n一句话总结\n热闹的一天"
        current = briefing.parse_briefing_payload(plain, source_messages=self.sources)
        self.assertEqual(current["active_members"], [])
        legacy = briefing.parse_briefing_payload(plain)
        self.assertEqual(legacy["active_members"], ["群友甲：自称什么都懂"])

    def test_four_member_cards_share_text_and_image_format_without_internal_ids(self):
        self.sources = [Message("user", f"成员{i}本人观点", name=f"成员{i}", sender_id=100 + i, source_id=200 + i)
                        for i in range(6)]
        card = self.parse([self.entry(user_id=100 + i, source_ids=[200 + i]) for i in range(6)])
        self.assertEqual(len(card["active_members"]), 4)
        document = build_card("briefing", card, group_name="群", group_id=7, date_label="2026-10-22")
        block = next(block for block in document.blocks if block.heading == "今日群友画像")
        self.assertEqual(len(block.entries), 4)
        text = briefing.build_briefing_text(card)
        for entry in block.entries:
            self.assertIn(entry, text)
        self.assertIn("成员3 · 原文质检员", text)
        for absent in ("成员4", "source_ids", "user_id", "仅供娱乐"):
            self.assertNotIn(absent, text)

    def test_portrait_uses_actual_scene_not_model_personality_analogy(self):
        card = self.parse([self.entry(personality_tag="迷糊包子", comment="今日脑袋像一只烧糊的包子。")])
        card["portrait_prompt"] = "群友拿着核对清单检查缓存故障。"
        prompt = briefing.build_portrait_prompt(card)
        self.assertIn("核对清单", prompt)
        self.assertNotIn("豆包", prompt)
        self.assertNotIn("烧糊的包子", prompt)
        self.assertNotIn("source_ids", prompt)
