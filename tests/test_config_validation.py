import unittest

import nonebot
nonebot.init()

from src.plugins.llm_chat.config import Config


class ConfigValidationTests(unittest.TestCase):
    def test_daily_schedule_time_is_normalized(self):
        self.assertEqual(Config(briefing_time="8:05").briefing_time, "08:05")
        self.assertEqual(Config(news_broadcast_time="23:30").news_broadcast_time, "23:30")

    def test_invalid_daily_schedule_time_is_rejected(self):
        for value in ("25:00", "08:60", "morning", "08:00:00", ""):
            with self.assertRaises(ValueError, msg=value):
                Config(briefing_time=value)
            with self.assertRaises(ValueError, msg=value):
                Config(news_broadcast_time=value)

    def test_removed_retry_knobs_are_gone(self):
        config = Config()
        for name in ("llm_max_retries", "vision_max_retries",
                     "image_gen_max_retries", "google_tts_max_retries",
                     # internal tuning constants, now module-level
                     "llm_log_response_max_chars", "llm_log_tool_result_max_chars",
                     "agent_context_max_chars", "max_thinking_messages",
                     "memory_embedding_batch_size", "memory_embedding_timeout",
                     "proactive_idle_threshold", "proactive_check_window",
                     "proactive_planner_cooldown", "proactive_keywords",
                     "proactive_keyword_probability",
                     "repeat_window", "repeat_min_count", "repeat_base_probability",
                     "repeat_probability_increment", "repeat_cooldown",
                     "repeat_text_only", "repeat_exclude_self",
                     "reply_judge_window", "briefing_min_messages"):
            self.assertFalse(hasattr(config, name), name)
