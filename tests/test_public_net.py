import unittest
from unittest.mock import AsyncMock, patch

import nonebot
nonebot.init()

from src.plugins.llm_chat import public_net


class _FakeResponse:
    def __init__(self, body=b"", content_type="image/png", redirect=None, declared=None):
        self._body = body
        self._redirect = redirect
        self.is_redirect = bool(redirect)
        self.headers = {"content-type": content_type}
        if redirect:
            self.headers["location"] = redirect
        if declared is not None:
            self.headers["content-length"] = str(declared)

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield self._body


class _FakeStream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def stream(self, method, url, *, headers=None, extensions=None, follow_redirects=True):
        self.requests.append((str(url), dict(headers or {}), dict(extensions or {})))
        return _FakeStream(self.responses.pop(0))


class PublicAddressTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_non_public_destinations(self):
        client = AsyncMock()
        for url in (
            "http://127.0.0.1/x.png",
            "http://localhost/x.png",
            "http://169.254.169.254/latest/meta-data",
            "http://10.0.0.5/x.png",
            "http://[::1]/x.png",
            "file:///etc/passwd",
            "http://user:pass@example.com/x.png",
            "http://example.com:8080/x.png",
            "http://metadata.internal/x.png",
        ):
            with self.assertRaises(ValueError, msg=url):
                await public_net.resolve_public_address(url, client)

    async def test_accepts_public_ip_literal(self):
        address = await public_net.resolve_public_address("https://93.184.216.34/x.png", AsyncMock())
        self.assertEqual(address, "93.184.216.34")

    async def test_dns_answer_must_be_global(self):
        client = AsyncMock()
        with patch.object(public_net.socket, "getaddrinfo",
                          return_value=[(None, None, None, None, ("10.0.0.7", 443))]):
            with self.assertRaises(ValueError):
                await public_net.resolve_public_address("https://evil.example/x.png", client)
        with patch.object(public_net.socket, "getaddrinfo",
                          return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
            self.assertEqual(
                await public_net.resolve_public_address("https://example.com/x.png", client),
                "93.184.216.34",
            )

    async def test_fetch_pins_validated_address_and_keeps_host_identity(self):
        client = _FakeClient([_FakeResponse(body=b"image")])
        with patch.object(public_net.socket, "getaddrinfo",
                          return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
            data, content_type = await public_net.fetch_pinned_bytes(
                "https://example.com/a.png", client=client, max_bytes=64,
            )
        self.assertEqual(data, b"image")
        self.assertEqual(content_type, "image/png")
        requested_url, headers, extensions = client.requests[0]
        self.assertEqual(requested_url, "https://93.184.216.34/a.png")
        self.assertEqual(headers["Host"], "example.com")
        self.assertEqual(extensions["sni_hostname"], "example.com")

    async def test_each_redirect_hop_is_revalidated(self):
        client = _FakeClient([_FakeResponse(redirect="http://127.0.0.1/secret")])
        with patch.object(public_net.socket, "getaddrinfo",
                          return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
            with self.assertRaises(ValueError):
                await public_net.fetch_pinned_bytes("https://example.com/a.png", client=client, max_bytes=64)
        self.assertEqual(len(client.requests), 1)

    async def test_size_cap_is_enforced_while_streaming(self):
        client = _FakeClient([_FakeResponse(body=b"x" * 32)])
        with patch.object(public_net.socket, "getaddrinfo",
                          return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
            with self.assertRaises(ValueError):
                await public_net.fetch_pinned_bytes("https://example.com/a.png", client=client, max_bytes=16)
