from dataclasses import replace
import unittest

import nonebot


nonebot.init()

from src.plugins.llm_chat.planner import PlannerDecision, ReplyPart
from src.plugins.llm_chat.turns import build_turn_items, wants_meme
from src.plugins.llm_chat.reply_policy import DETAILED_REPLY


class ResponsePartsTests(unittest.TestCase):
    def test_meme_and_text_follow_declared_order(self):
        meme = {"id": 1, "topic": "funny"}
        meme_then_text = PlannerDecision(
            parts=[ReplyPart("meme", "funny"), ReplyPart("text")]
        )
        text_then_meme = PlannerDecision(
            parts=[ReplyPart("text"), ReplyPart("meme", "funny")]
        )

        self.assertEqual(
            build_turn_items(
                text="接一句",
                decision=meme_then_text,
                planned_meme=meme,
            ),
            [("meme", meme), ("text", "接一句")],
        )
        self.assertEqual(
            build_turn_items(
                text="接一句",
                decision=text_then_meme,
                planned_meme=meme,
            ),
            [("text", "接一句"), ("meme", meme)],
        )
        self.assertTrue(wants_meme(meme_then_text))

    def test_two_text_parts_split_one_model_response(self):
        decision = PlannerDecision(
            parts=[ReplyPart("text"), ReplyPart("text")]
        )

        items = build_turn_items(
            text="第一句。\n第二句。",
            decision=decision,
            planned_meme=None,
            reply_policy=DETAILED_REPLY,
        )

        self.assertEqual(items, [("text", "第一句"), ("text", "第二句")])

    def test_execution_can_expand_or_collapse_preliminary_text_plan(self):
        text = "第一段保留\n内部换行\n\n第二段保留"
        one = PlannerDecision(parts=[ReplyPart("text")])
        two = PlannerDecision(parts=[ReplyPart("text"), ReplyPart("text")])
        self.assertEqual(build_turn_items(
            text=text, decision=one, planned_meme=None, reply_policy=replace(DETAILED_REPLY, text_parts=2),
        ), [("text", "第一段保留\n内部换行"), ("text", "第二段保留")])
        self.assertEqual(build_turn_items(
            text=text, decision=two, planned_meme=None, reply_policy=replace(DETAILED_REPLY, text_parts=1),
        ), [("text", text)])

    def test_failed_media_explanation_is_not_discarded(self):
        decision = PlannerDecision(parts=[ReplyPart("image")])

        items = build_turn_items(
            text="这次图片没生成出来。",
            decision=decision,
            planned_meme=None,
        )

        self.assertEqual(items, [("text", "这次图片没生成出来")])


if __name__ == "__main__":
    unittest.main()
