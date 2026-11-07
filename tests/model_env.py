"""Explicit dummy model routes for offline tests; never used by the Bot."""

TEST_MODEL_ENV = {
    "MODEL_SEARCH": "dashscope:test-search",
    "MODEL_VISION": "google:gemma-test",
    "MODEL_TTS": "google:test-tts",
    "MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test",
    "MODEL_CHAT": "openai:test-chat",
    "MODEL_PLANNER": "openai:test-planner",
    "MODEL_SUMMARY": "openai:test-summary",
    "MODEL_IMPORTANT_INFO": "openai:test-important-info",
    "MODEL_GENERAL": "cliproxyapi:test-general",
    "MODEL_FALLBACK": "openai:test-fallback",
}
