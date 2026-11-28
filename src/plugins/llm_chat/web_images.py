"""Search Wikimedia Commons for existing pictures and retrieve the selected preview."""
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx

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
