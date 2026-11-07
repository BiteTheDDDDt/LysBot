import unittest
from unittest.mock import patch

import nonebot

nonebot.init()

from src.plugins.llm_chat import sentiment


class SentimentParticipationTests(unittest.TestCase):
    def test_annoyance_hints_reduce_participation_without_forcing_apologies(self):
        analyzer = sentiment.SentimentAnalyzer()
        state = analyzer._get_state(71)
        for score, factor in ((0, 1.0), (40, 0.6), (60, 0.3), (80, 0.1)):
            with self.subTest(score=score):
                state.annoyance_score = score
                hint = analyzer.get_mood_hint(71)
                self.assertEqual(analyzer.should_speak_less(71), (score >= 40, factor))
                for unwanted in ("道歉", "抱歉", "对不起", "谦卑", "低姿态"):
                    self.assertNotIn(unwanted, hint)
        self.assertIn("暂时不主动发言", hint)
        self.assertIn("结束当前互动", hint)

    def test_repeated_rejection_preserves_quiet_cooldown_and_expiration(self):
        analyzer = sentiment.SentimentAnalyzer()
        state = analyzer._get_state(72)
        with patch.object(sentiment.time, "time", return_value=1000):
            for _ in range(4):
                analyzer._update_state(state, sentiment.SentimentRecord(
                    content="别继续了", timestamp=1000, user_id=1,
                    sentiment_type="negative", confidence=0.9,
                ))
            self.assertTrue(analyzer.get_state_summary(72)["is_quiet_mode"])
            self.assertEqual(analyzer.should_speak_less(72), (True, 0.1))
        with patch.object(sentiment.time, "time", return_value=2801):
            self.assertFalse(analyzer.is_quiet_mode(72))
            self.assertIsNone(state.quiet_mode_until)
