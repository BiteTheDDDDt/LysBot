import hashlib
import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts.doctor import diagnose
from scripts.napcat import NapCatClient, NapCatError, connection_config


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parents[1]
        (self.root / "src/plugins/llm_chat").mkdir(parents=True)
        for name in ("local_embedding.py", "providers.py"):
            shutil.copy2(source / "src/plugins/llm_chat" / name, self.root / "src/plugins/llm_chat" / name)
        shutil.copytree(source / "personas", self.root / "personas")

    def diagnose(self, **overrides):
        values = {"MASTER_QQ": "42", "ACCESS_TOKEN": "private-onebot-value",
                  "MODEL_GENERAL": "openai:test-general", "OPENAI_API_KEY": "private-model-value", **overrides}
        with patch("scripts.doctor.importlib.util.find_spec", return_value=object()):
            return diagnose(self.root, values)

    def test_missing_vector_is_warning_and_report_does_not_leak_credentials(self):
        result = self.diagnose()
        self.assertTrue(result["ok"])
        self.assertFalse(result["network_verified"])
        self.assertFalse(result["capabilities"]["vision"])
        self.assertTrue(any(item["status"] == "warning" for item in result["checks"]))
        output = json.dumps(result)
        self.assertNotIn("private-onebot-value", output)
        self.assertNotIn("private-model-value", output)
        self.assertFalse((self.root / "data").exists())

    def test_invalid_provider_and_incompatible_optional_model_are_errors(self):
        result = self.diagnose(MODEL_GENERAL="unknown:example", MODEL_TTS="openai:text-only")
        codes = {item["code"] for item in result["checks"]}
        self.assertFalse(result["ok"])
        self.assertIn("invalid_model_spec", codes)
        self.assertIn("incompatible_model_capability", codes)

    def test_news_dependency_and_example_configuration_are_reported(self):
        result = self.diagnose(ENABLE_DAILY_NEWS_BROADCAST="true", OPENAI_API_KEY="your_api_key")
        codes = {item["code"] for item in result["checks"]}
        self.assertIn("news_dependencies_missing", codes)
        self.assertIn("provider_credentials_missing_or_placeholder", codes)


class NapCatOperationsTests(unittest.TestCase):
    def test_connection_configuration_preserves_others_and_is_idempotent(self):
        original = {"network": {"httpServers": [{"name": "other-http"}],
                                "websocketClients": [{"name": "other", "url": "ws://other/"}]},
                    "parseMultMsg": True}
        before = deepcopy(original)
        desired = connection_config(original, "ws://bot:8080/onebot/v11/ws", "onebot-secret")
        self.assertEqual(original, before)
        self.assertEqual(desired["network"]["websocketClients"][0], original["network"]["websocketClients"][0])
        self.assertEqual(desired, connection_config(desired, "ws://bot:8080/onebot/v11/ws", "onebot-secret"))
        self.assertTrue(desired["parseMultMsg"])

    def test_api_authentication_and_config_write_are_verified(self):
        state = {"config": {"network": {"websocketClients": []}}, "writes": 0}

        def transport(request):
            body = json.loads(request.content)
            route = request.url.path
            if route == "/api/auth/login":
                self.assertEqual(body["hash"], hashlib.sha256(b"webui-secret.napcat").hexdigest())
                data = {"Credential": "private-credential"}
            else:
                self.assertEqual(request.headers["Authorization"], "Bearer private-credential")
                if route.endswith("CheckLoginStatus"):
                    data = {"isLogin": True, "coreReady": True, "qrcodeurl": "must-not-return-this"}
                elif route.endswith("SetConfig"):
                    state["config"] = json.loads(body["config"])
                    state["writes"] += 1
                    data = None
                else:
                    data = state["config"]
            return httpx.Response(200, json={"code": 0, "data": data})

        client = NapCatClient("http://napcat:6099", "webui-secret", transport=httpx.MockTransport(transport))
        self.addCleanup(client.close)
        client.login()
        self.assertNotIn("qrcodeurl", client.status())
        self.assertTrue(client.connect("ws://bot:8080/onebot/v11/ws", "onebot-secret")["changed"])
        self.assertFalse(client.connect("ws://bot:8080/onebot/v11/ws", "onebot-secret")["changed"])
        self.assertEqual(state["writes"], 1)

    def test_login_required_prevents_configuration_write(self):
        client = NapCatClient("http://napcat:6099", "webui-secret", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"code": 0, "data": {"isLogin": False}})
        ))
        self.addCleanup(client.close)
        result = client.connect("ws://bot:8080/onebot/v11/ws", "onebot-secret")
        self.assertEqual(result["needs_user_action"], "scan_in_qq")
        self.assertFalse(result["changed"])

    def test_qr_is_written_privately_without_returning_its_url(self):
        client = NapCatClient("http://napcat:6099", "webui-secret", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"code": 0, "data": {"qrcode": "https://example.com/login-test"}})
        ))
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private/qr.png"
            result = client.qr(output)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertNotIn("https://example.com", json.dumps(result))

    def test_invalid_configuration_is_rejected(self):
        for current, url, token in (([], "ws://bot/", "secret"), ({}, "http://bot/", "secret"),
                                    ({}, "ws://bot/", "replace-with-token")):
            with self.assertRaises(NapCatError):
                connection_config(current, url, token)
