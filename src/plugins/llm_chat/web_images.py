"""Find existing web images with native search and retrieve the selected picture."""
import asyncio
import hashlib
import io
import json
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx
from PIL import Image

from .config import config

HEADERS = {"User-Agent": "LysBot/1.0 (https://github.com/BiteTheDDDDt/LysBot)"}
API = "https://commons.wikimedia.org/w/api.php"


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _plain(value):
    parser = _PlainText()
    parser.feed(value)
    return " ".join("".join(parser.parts).split())


@dataclass(frozen=True)
class WebImage:
    id: str
    title: str
    url: str
    source: str
    credit: str


async def search_images(query: str) -> list[WebImage]:
    from .roles import role_available
    from .web_search import search

    if role_available("search"):
        result = await search(query, find_images=True, limit=15)
        candidates = _native_candidates(result)
        if candidates:
            return candidates
    return await _search_commons(query)


def _native_candidates(result: dict) -> list[WebImage]:
    text = (result.get("search_summary") or {}).get("content", "")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    except ValueError:
        payload = {}
    rows = payload.get("images") if isinstance(payload, dict) else None
    # Also accept image files explicitly cited by the provider when its text
    # uses ordinary prose. A page citation alone is never an image candidate.
    rows = list(rows[:3]) if isinstance(rows, list) else []
    for item in result.get("items", []):
        url = item.get("url", "")
        if url.split("?", 1)[0].lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            rows.append({"image_url": url, "title": item.get("title")})
    images = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("image_url"), str):
            continue
        url = row["image_url"].strip()
        try:
            parsed = httpx.URL(url)
        except httpx.InvalidURL:
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.host or parsed.userinfo or len(url) > 4096:
            continue
        key = "photo:" + hashlib.sha256(url.encode()).hexdigest()[:24]
        title = row.get("title")
        images[key] = WebImage(key, title[:200] if isinstance(title, str) and title else parsed.host, url, url, "")
        if len(images) == 3:
            break
    return list(images.values())


async def _search_commons(query: str) -> list[WebImage]:
    params = {
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "search", "gsrnamespace": 6, "gsrlimit": 8,
        "gsrsearch": query + " filetype:bitmap", "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata", "iiurlwidth": 960,
        "iiextmetadatafilter": "Artist|Attribution|LicenseShortName",
    }
    async with httpx.AsyncClient(headers=HEADERS, timeout=12) as client:
        response = await client.get(API, params=params)
        response.raise_for_status()
        payload = response.json()
    if "error" in payload:
        raise ValueError("Wikimedia search failed: " + str(payload["error"].get("code")))
    results = []
    for page in sorted(payload.get("query", {}).get("pages", []), key=lambda p: p["index"]):
        info = next(iter(page.get("imageinfo", [])), {})
        url = httpx.URL(info.get("thumburl") or info.get("url") or "")
        if (url.scheme != "https" or url.host not in {"thumb.wikimedia.org", "upload.wikimedia.org"}
                or url.port or url.userinfo):
            continue
        if info.get("thumbmime", info.get("mime")) not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            continue
        meta = info.get("extmetadata", {})
        def field(name):
            return _plain(meta.get(name, {}).get("value", ""))
        credit = " · ".join(filter(None, [field("Attribution") or field("Artist"), field("LicenseShortName")]))
        page_id = page["pageid"]
        results.append(WebImage(f"commons:{page_id}", page["title"].removeprefix("File:"), str(url),
                                f"https://commons.wikimedia.org/?curid={page_id}", credit))
    return results


async def download_preview(image: WebImage) -> bytes:
    limit = max(1, int(config.image_download_max_bytes))
    if image.id.startswith("photo:"):
        from .public_net import fetch_pinned_bytes

        async with asyncio.timeout(18), httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(10, connect=4)) as client:
            data, _ = await fetch_pinned_bytes(image.url, client=client, max_bytes=limit, headers=HEADERS,
                                               content_type_prefixes=("image/", "application/octet-stream"))
        await asyncio.to_thread(_verify_image, data)
        return data
    async with httpx.AsyncClient(headers=HEADERS, timeout=12) as client:
        async with client.stream("GET", image.url, follow_redirects=False) as response:
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";")[0] not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                raise ValueError("unexpected image format")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > limit:
                    raise ValueError("image exceeds size limit")
                data.extend(chunk)
    if not data:
        raise ValueError("empty image")
    return bytes(data)


def _verify_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG", "PNG", "WEBP", "GIF")) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("image exceeds pixel limit")
            image.load()
    except Image.DecompressionBombError as error:
        raise ValueError("image exceeds pixel limit") from error
