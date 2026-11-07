import unittest
from unittest.mock import patch

import nonebot


nonebot.init()

from src.plugins.llm_chat.config import config
from src.plugins.llm_chat.config import Config
from src.plugins.llm_chat.providers import (
    PROVIDERS,
    model_supports_tools,
    parse_spec,
    provider_configured,
    supports_vision,
)


class CLIProxyAPIProviderTests(unittest.TestCase):
    def test_legacy_specs_and_credentials_remain_compatible(self):
        self.assertEqual(parse_spec("codexproxy:gpt-6-luna"), ("cliproxyapi", "gpt-6-luna"))
        legacy = Config(codexproxy_api_key="legacy", codexproxy_base_url="https://legacy.test/v1")
        self.assertEqual(legacy.cliproxyapi_api_key, "legacy")
        self.assertEqual(legacy.cliproxyapi_base_url, "https://legacy.test/v1")
        explicit = Config(codexproxy_api_key="old", cliproxyapi_api_key="new", codexproxy_base_url="https://old.test/v1", cliproxyapi_base_url="https://new.test/v1")
        self.assertEqual(explicit.cliproxyapi_api_key, "new")
        self.assertEqual(explicit.cliproxyapi_base_url, "https://new.test/v1")

    def test_provider_uses_local_openai_compatible_endpoint(self):
        provider = PROVIDERS["cliproxyapi"]

        self.assertEqual(provider.adapter_kind, "openai")
        self.assertEqual(provider.api_key_attr, "cliproxyapi_api_key")
        self.assertEqual(provider.base_url_attr, "cliproxyapi_base_url")
        self.assertEqual(config.cliproxyapi_base_url, "http://127.0.0.1:8317/v1")

    def test_provider_can_be_used_in_spec_chains(self):
        self.assertEqual(
            parse_spec("cliproxyapi:gpt-5.6-sol"),
            ("cliproxyapi", "gpt-5.6-sol"),
        )
        self.assertTrue(model_supports_tools("cliproxyapi:gpt-5.6-sol"))
        self.assertTrue(supports_vision("cliproxyapi:gpt-5.6-sol"))
        self.assertTrue(supports_vision("cliproxyapi:gpt-5.4-mini"))
        self.assertTrue(supports_vision("cliproxyapi:gpt-5.5"))
        self.assertTrue(supports_vision("cliproxyapi:gpt-6-luna"))
        self.assertTrue(supports_vision("cliproxyapi:gpt-6-luna-2026-09-01"))
        self.assertFalse(supports_vision("cliproxyapi:gpt-6-luna-image"))
        self.assertTrue(supports_vision("cliproxyapi:gemini-3.1-pro"))
        self.assertTrue(supports_vision("cliproxyapi:claude-sonnet-4-6"))
        self.assertFalse(supports_vision("cliproxyapi:gpt-image-2"))
        self.assertFalse(supports_vision("cliproxyapi:gemini-3-pro-image"))

    def test_provider_is_configured_by_local_proxy_key(self):
        with patch.object(config, "cliproxyapi_api_key", "local-test-key"):
            self.assertTrue(provider_configured("cliproxyapi"))


if __name__ == "__main__":
    unittest.main()
