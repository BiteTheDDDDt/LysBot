"""多模态消息处理"""
import asyncio
import base64
from collections import OrderedDict
from typing import List, Optional, Tuple
import httpx
from nonebot import logger

from .config import config


_download_cache: OrderedDict[str, bytes] = OrderedDict()
_download_pending: dict[str, asyncio.Task[Optional[bytes]]] = {}
_download_semaphore = asyncio.Semaphore(max(1, int(config.image_download_concurrency)))


def _download_cache_limit() -> int:
    try:
        return max(0, int(config.image_download_cache_size))
    except Exception:
        return 256


def _get_cached_download(url: str) -> Optional[bytes]:
    if not url or _download_cache_limit() <= 0:
        return None
    cached = _download_cache.get(url)
    if cached is not None:
        _download_cache.move_to_end(url)
    return cached


def _remember_download(url: str, image_data: bytes) -> None:
    limit = _download_cache_limit()
    if not url or not image_data or limit <= 0:
        return
    _download_cache[url] = image_data
    _download_cache.move_to_end(url)
    while len(_download_cache) > limit or sum(map(len, _download_cache.values())) > max(1, int(config.image_download_cache_max_bytes)):
        _download_cache.popitem(last=False)


async def _download_image_uncached(url: str) -> Optional[bytes]:
    try:
        # Incoming image URLs are user-controlled: resolve and pin a public
        # address so loopback/link-local/private targets are never fetched.
        from .public_net import fetch_pinned_bytes

        limit = max(1, int(config.image_download_max_bytes))
        async with _download_semaphore, httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            data, _ = await fetch_pinned_bytes(
                url,
                client=client,
                max_bytes=limit,
                content_type_prefixes=("image/", "application/octet-stream"),
                headers={"User-Agent": "Mozilla/5.0 (compatible; QQBot/1.0)"},
            )
            return data
    except Exception as error:
        logger.warning(f"[multimodal] 图片下载失败: {error}")
        return None

async def download_image(url: str) -> Optional[bytes]:
    """下载图片，按 URL 做内存缓存并合并并发请求。"""
    cached = _get_cached_download(url)
    if cached is not None:
        logger.debug(f"[multimodal] 命中图片下载缓存 url={url}")
        return cached

    task = _download_pending.get(url)
    if task is None:
        async def fetch():
            try:
                data = await asyncio.wait_for(_download_image_uncached(url), timeout=30)
                if data:
                    _remember_download(url, data)
                return data
            except asyncio.TimeoutError:
                return None
            finally:
                if _download_pending.get(url) is asyncio.current_task():
                    _download_pending.pop(url, None)
        task = asyncio.create_task(fetch(), name="image-download")
        _download_pending[url] = task
    # One cancelled message must not cancel a download shared by other messages.
    return await asyncio.shield(task)


async def close_downloads() -> None:
    tasks = list(_download_pending.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _download_pending.clear()
    _download_cache.clear()


async def image_to_base64(image_data: bytes) -> str:
    """图片转base64"""
    return base64.b64encode(image_data).decode("utf-8")


async def get_image_base64(url: str) -> Optional[str]:
    """从URL获取图片的base64编码"""
    image_data = await download_image(url)
    if image_data:
        return await image_to_base64(image_data)
    return None


async def process_image_urls(urls: List[str]) -> List[Tuple[str, str]]:
    """批量处理图片URL，返回 (url, base64) 列表。"""
    selected = list(dict.fromkeys(urls))[:max(1, int(config.incoming_images_limit))]
    encoded = await asyncio.gather(*(get_image_base64(url) for url in selected))
    return [(url, b64) for url, b64 in zip(selected, encoded) if b64]
