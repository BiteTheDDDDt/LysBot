"""Read publication metadata without treating model-written dates as evidence."""
import asyncio
import json
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from .config import config
from .public_net import FAKE_IP_RANGE, DOH_ENDPOINT, resolve_public_address  # noqa: F401 - re-exported for callers


def publication_date(value) -> str:
    try:
        return date.fromisoformat(str(value or "").strip()[:10]).isoformat()
    except ValueError:
        return ""


class PublicationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.dates = []
        self._script = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        key = str(attributes.get("property") or attributes.get("name") or attributes.get("itemprop") or "").lower()
        if tag == "meta" and key in {"article:published_time", "datepublished", "parsely-pub-date", "dc.date.issued", "publishdate", "pubdate"}:
            self.dates.append(attributes.get("content"))
        if tag == "time" and key == "datepublished":
            self.dates.append(attributes.get("datetime"))
        if tag == "script" and attributes.get("type") == "application/ld+json":
            self._script = []

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag):
        if tag != "script" or self._script is None:
            return
        try:
            self._walk(json.loads("".join(self._script)))
        except (ValueError, RecursionError):
            pass
        self._script = None

    def _walk(self, value):
        if isinstance(value, list):
            for item in value:
                self._walk(item)
        elif isinstance(value, dict):
            if value.get("datePublished"):
                self.dates.append(value["datePublished"])
            if "@graph" in value:
                self._walk(value["@graph"])


def extract_publication_date(html: str) -> str:
    parser = PublicationParser()
    parser.feed(html)
    dates = {publication_date(value) for value in parser.dates} - {""}
    # Conflicting article metadata is ambiguous; don't guess the freshest date.
    return next(iter(dates)) if len(dates) == 1 else ""


async def _public_address(url: str, client: httpx.AsyncClient) -> str:
    """Resolve and pin a public destination, including hosts behind Fake-IP DNS."""
    return await resolve_public_address(url, client)


async def fetch_public_page(url: str, *, timeout=None, max_bytes=1024 * 1024) -> dict:
    request_timeout=max(0.1,float(config.news_verify_timeout if timeout is None else timeout))
    async def fetch():
        current = url
        async with httpx.AsyncClient(timeout=request_timeout, trust_env=False) as client:
            for _ in range(4):
                address = await _public_address(current, client)
                original = httpx.URL(current)
                # Host + TLS SNI retain the website identity while the connection
                # uses exactly the address we validated, with no second DNS lookup.
                headers = {"Host": original.netloc.decode("ascii"), "Connection": "close",
                           "User-Agent": "Mozilla/5.0 (compatible; NewsDateCheck/1.0)"}
                async with client.stream("GET", original.copy_with(host=address), headers=headers,
                                         extensions={"sni_hostname": original.host}, follow_redirects=False) as response:
                    if response.is_redirect:
                        current = urljoin(current, response.headers.get("location", ""))
                        continue
                    response.raise_for_status()
                    content_type=response.headers.get("content-type", "").lower()
                    if not any(t in content_type for t in ("html", "text/plain")):
                        return {"status": "not_html", "published_at": ""}
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            return {"status": "too_large", "published_at": ""}
                    return {"status":"ok","url":current,"content_type":content_type,
                            "body":data.decode(response.encoding or "utf-8",errors="replace")}
        return {"status": "too_many_redirects", "published_at": ""}
    try:
        return await asyncio.wait_for(fetch(), timeout=request_timeout)
    except (TimeoutError, httpx.TimeoutException):
        return {"status": "timeout", "published_at": ""}
    except httpx.HTTPStatusError as error:
        return {"status": f"http_{error.response.status_code}", "published_at": ""}
    except (ValueError, OSError, httpx.HTTPError, RecursionError) as error:
        return {"status": type(error).__name__, "published_at": ""}


async def inspect_publication_date(url: str) -> dict:
    page=await fetch_public_page(url)
    if page["status"]!="ok":
        return {"status":page["status"],"published_at":""}
    if "html" not in page["content_type"]:
        return {"status":"not_html","published_at":""}
    value=extract_publication_date(page["body"])
    return {"status":"verified" if value else "missing_metadata","published_at":value}


async def read_publication_date(url: str) -> str:
    return (await inspect_publication_date(url))["published_at"]
