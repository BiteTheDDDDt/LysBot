import unittest
from unittest.mock import AsyncMock, patch

import nonebot


nonebot.init()

from src.plugins.llm_chat import models


class MemeAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_sentence_image_is_rejected_even_if_model_marks_it_as_meme(self):
        for count in (3, 8):
            result, _ = await self._analyze(
                '一张聊天截图。\n' +
                '{"is_meme":true,"topic":"funny","random_eligible":true,"text_sentence_count":' + str(count) + '}'
            )
            self.assertFalse(result["is_meme"])
            self.assertFalse(result["random_eligible"])
            self.assertIsNone(result["topic"])

    async def test_two_sentence_joke_is_still_allowed(self):
        result, _ = await self._analyze('两句笑话。\n{"is_meme":true,"topic":"funny","random_eligible":true,"text_sentence_count":2}')
        self.assertTrue(result["is_meme"])

    async def _analyze(self, response: str) -> tuple[dict, str]:
        manager = models.model_manager
        with patch.object(
            manager,
            "_chat_with_images_impl",
            AsyncMock(return_value=response),
        ) as analyze:
            result = await manager._analyze_image_for_meme_uncached(
                "base64-image",
                session_id="test",
                caller="test:meme_analysis",
            )
        prompt = analyze.await_args.args[0][0].content
        return result, prompt

    async def test_captioned_joke_can_be_collected_without_random_use(self):
        result, _ = await self._analyze(
            '角色头上写着好股票。\n'
            '{"is_meme": true, "topic": "funny", "random_eligible": false}'
        )
        self.assertTrue(result["is_meme"])
        self.assertFalse(result["random_eligible"])

    async def test_self_contained_text_joke_can_be_random(self):
        result, _ = await self._analyze(
            '虽然什么都没做，但还是辛苦我自己了。\n'
            '{"is_meme": true, "topic": "funny", "random_eligible": true}'
        )
        self.assertTrue(result["random_eligible"])

    async def test_accepts_only_native_true_with_allowed_topic(self):
        result, _ = await self._analyze(
            '一只猫瞪大眼睛，头顶有一个问号。\n'
            '{"is_meme": true, "topic": "shock"}'
        )

        self.assertEqual(
            result,
            {
                "description": "一只猫瞪大眼睛，头顶有一个问号。",
                "is_meme": True,
                "topic": "shock",
                "random_eligible": None,
                "meme_kind": None,
                "classification_reason": "",
                "text_sentence_count": None,
            },
        )

    async def test_rejects_string_false_instead_of_treating_it_as_truthy(self):
        result, _ = await self._analyze(
            '一张普通游戏界面截图。\n'
            '{"is_meme": "false", "topic": "funny"}'
        )

        self.assertFalse(result["is_meme"])
        self.assertIsNone(result["topic"])

    async def test_rejects_string_true_and_numeric_true(self):
        for raw_value in ('"true"', "1"):
            with self.subTest(raw_value=raw_value):
                result, _ = await self._analyze(
                    "一张普通内容图。\n"
                    f'{{"is_meme": {raw_value}, "topic": "funny"}}'
                )

                self.assertFalse(result["is_meme"])
                self.assertIsNone(result["topic"])

    async def test_false_result_never_keeps_a_topic(self):
        result, _ = await self._analyze(
            '一张六宫格分类图。\n'
            '{"is_meme": false, "topic": "funny"}'
        )

        self.assertFalse(result["is_meme"])
        self.assertIsNone(result["topic"])

    async def test_true_with_unknown_topic_is_rejected(self):
        result, _ = await self._analyze(
            '一个人物露出惊讶表情。\n'
            '{"is_meme": true, "topic": "confused", "random_eligible": true}'
        )

        self.assertFalse(result["is_meme"])
        self.assertIsNone(result["topic"])
        self.assertFalse(result["random_eligible"])


if __name__ == "__main__":
    unittest.main()
