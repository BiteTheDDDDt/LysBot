"""Validate outbound destinations before fetching untrusted URLs.

Message attachments carry attacker-controlled URLs, so a fetch must never reach
loopback, private, link-local or otherwise non-global addresses. The connection
is pinned to the address that was checked, which also removes the second DNS
lookup a redirect or a rebinding answer could exploit.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx

FAKE_IP_RANGE = ipaddress.ip_network("198.18.0.0/15")
DOH_ENDPOINT = "https://dns.google/resolve"
ALLOWED_PORTS = {None, 80, 443}


def _check_url(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("invalid_url")
    host = parsed.hostname.rstrip(".")
    if host == "localhost" or "." not in host or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("local_hostname")
    if parsed.port not in ALLOWED_PORTS:
        raise ValueError("blocked_port")
    return parsed.scheme, host, parsed.port or 443


async def resolve_public_address(url: str, client: httpx.AsyncClient) -> str:
    """Resolve and pin a public destination, including hosts behind Fake-IP DNS."""
    _, host, port = _check_url(url)
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise ValueError("non_public_address")
        return str(literal)
    resolved = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(ipaddress.ip_address(item[4][0]) for item in resolved))
    if addresses and all(address.is_global for address in addresses):
        return str(sorted(addresses, key=lambda address: address.version)[0])
    if not addresses or not all(address in FAKE_IP_RANGE for address in addresses):
        raise ValueError("non_public_address")
    # Only the known Fake-IP range takes this path. Never allow that range as a
    # destination: resolve the hostname independently and connect to the real IP.
    response = await client.get(DOH_ENDPOINT, params={"name": host, "type": "A"}, follow_redirects=False)
    response.raise_for_status()
    data = response.json()
    addresses = [ipaddress.ip_address(item["data"]) for item in data.get("Answer", []) if item.get("type") == 1]
    if data.get("Status") != 0 or not addresses or not all(address.is_global for address in addresses):
        raise ValueError("no_public_dns_answer")
    return str(addresses[0])


async def fetch_pinned_bytes(
    url: str,
    *,
    client: httpx.AsyncClient,
    max_bytes: int,
    content_type_prefixes: tuple[str, ...] = (),
    max_redirects: int = 4,
    headers: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    """Stream a public URL with a hard size cap, validating every redirect hop."""
    from urllib.parse import urljoin

    current = url
    for _ in range(max_redirects):
        address = await resolve_public_address(current, client)
        original = httpx.URL(current)
        request_headers = {"Host": original.netloc.decode("ascii"), "Connection": "close", **(headers or {})}
        async with client.stream(
            "GET",
            original.copy_with(host=address),
            headers=request_headers,
            extensions={"sni_hostname": original.host},
            follow_redirects=False,
        ) as response:
            if response.is_redirect:
                current = urljoin(current, response.headers.get("location", ""))
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type_prefixes and content_type and not content_type.startswith(content_type_prefixes):
                raise ValueError(f"unexpected content type: {content_type}")
            declared = int(response.headers.get("content-length") or 0)
            if declared > max_bytes:
                raise ValueError("response exceeds size limit")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError("response exceeds size limit")
            return bytes(data), content_type
    raise ValueError("too_many_redirects")
