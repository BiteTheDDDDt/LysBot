"""展示当前实际模型的辅助工具。"""

from __future__ import annotations

import re
import time
import asyncio
from typing import Any
from contextlib import AsyncExitStack
from weakref import WeakValueDictionary

from .config import PERSONA
from .logging_setup import general_logger


log = general_logger.bind(category="general")

_group_last_cards: dict[tuple, tuple[str, float]] = {}
_delivery_locks = WeakValueDictionary()
_group_base_cards: dict[tuple, str] = {}
_TRAILING_PAREN_RE = re.compile(r"^(.*?)[（(]([^（）()]{2,96})[）)]$")
_proxy_owners: dict[str, str] = {}
_proxy_refresh_at = 0.0
_proxy_refresh_lock = asyncio.Lock()


async def refresh_proxy_model_identity() -> None:
    """Read CLIProxyAPI's model owners, without making an inference request."""
    global _proxy_refresh_at
    from .config import config
    from .providers import get_provider_proxy
    import httpx

    if not config.cliproxyapi_api_key or time.monotonic() < _proxy_refresh_at:
        return
    async with _proxy_refresh_lock:
        if time.monotonic() < _proxy_refresh_at:
            return
        try:
            async with httpx.AsyncClient(proxy=get_provider_proxy("cliproxyapi"), timeout=2.0) as client:
                response = await client.get(
                    config.cliproxyapi_base_url.rstrip("/") + "/models",
                    headers={"Authorization": f"Bearer {config.cliproxyapi_api_key}"},
                )
                response.raise_for_status()
                owners = {
                    str(item["id"]): str(item["owned_by"])
                    for item in response.json().get("data", [])
                    if isinstance(item, dict) and item.get("id") and item.get("owned_by")
                }
            _proxy_owners.update(owners)
            _proxy_refresh_at = time.monotonic() + 3600
        except Exception:
            # Missing metadata must not prevent replies or expose credential errors.
            _proxy_refresh_at = time.monotonic() + 60


def format_provider_label(provider: str, model_id: str = "") -> str:
    if provider in {"cliproxyapi", "codexproxy"}:
        owner = _proxy_owners.get(model_id, "").lower()
        upstream = {"antigravity": "Antigravity", "openai": "OpenAI", "codex": "Codex", "xai": "xAI"}.get(owner, owner)
        return "CLIProxyAPI" + (f"/{upstream}" if upstream else "")
    return {"openrouter": "OpenRouter", "google": "Google", "anthropic": "Anthropic"}.get(provider, provider)


def format_model_label(spec_or_model_id: str | None) -> str:
    """将 provider:model_id 压成适合展示的短标签。"""
    raw = (spec_or_model_id or "").strip().strip("\"'")
    if not raw:
        return ""
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    raw = raw.strip()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return raw[:40]


def format_model_spec_label(spec_or_model_id: str | None) -> str:
    raw = (spec_or_model_id or "").strip().strip("\"'")
    if not raw:
        return ""
    if ":" not in raw:
        return format_model_label(raw)
    provider, model_id = raw.split(":", 1)
    provider = provider.strip()
    model_id = model_id.strip()
    provider = format_provider_label(provider, model_id)
    if "/" in model_id:
        model_id = model_id.rsplit("/", 1)[-1]
    label = f"{provider}:{model_id}" if provider and model_id else raw
    return label[:72]


async def label_usage_models(stats: dict) -> None:
    """Freeze catalog labels in the briefing snapshot, independent of renderer caches."""
    await refresh_proxy_model_identity()
    for row in stats.get("by_spec") or []:
        spec = row.get("spec") or ""
        row["display_label"] = format_model_spec_label(spec)
        if spec.split(":", 1)[0] in {"cliproxyapi", "codexproxy"}:
            row["platform_source"] = "model_catalog" if _proxy_owners.get(spec.partition(":")[2]) else "unknown"
            if not _proxy_owners.get(spec.partition(":")[2]):
                row["display_label"] += "（上游未确认）"


def _strip_existing_model_label(name: str) -> str:
    text = (name or "").strip() or PERSONA.name
    while True:
        m = _TRAILING_PAREN_RE.search(text)
        if not m:
            return text or PERSONA.name
        label = m.group(2)
        if not re.match(r"^(?:CLIProxyAPI|codexproxy|cliproxyapi|Google|Anthropic|OpenRouter|openai|deepseek|nvidia|dashscope|siliconflow|tencentmaas|volc|xiaomimimo)(?:/[^:]+)?:|^(?:gpt-|gemini-|claude-|grok-|deepseek-)", label, re.I):
            return text
        text = (m.group(1) or "").strip()
        if not text:
            return PERSONA.name
    return text


def delivery_lock(bot, group_id):
    if not bot or not group_id or not getattr(bot, "self_id", None):
        return AsyncExitStack()
    key = (str(bot.self_id), int(group_id))
    lock = _delivery_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _delivery_locks[key] = lock
    return lock


async def update_group_card_for_response(bot: Any, group_id: int | None, response: Any) -> None:
    """Best-effort identity after confirmed text delivery; caller holds delivery_lock."""
    from .config import config
    if not config.enable_model_card:
        return
    spec = getattr(response, "model_spec", None)
    if not bot or not group_id or not spec:
        return
    key = (str(bot.self_id), int(group_id))
    try:
        async with asyncio.timeout(5):
            await refresh_proxy_model_identity()
            label = format_model_spec_label(spec)
            if not label:
                return
            cached = _group_last_cards.get(key)
            if cached and cached[0] == label and time.monotonic() - cached[1] < 60:
                return
            self_id = int(bot.self_id)
            info = await bot.get_group_member_info(group_id=int(group_id), user_id=self_id, no_cache=True)
            current = str(info.get("card") or info.get("nickname") or PERSONA.name)
            base = _strip_existing_model_label(current)
            target = f"{base}（{label}）"
            if current != target:
                await bot.set_group_card(group_id=int(group_id), user_id=self_id, card=target)
                log.info(f"[model_identity] 群 {group_id} 名片已更新为 {target}")
            _group_base_cards[key] = base
            _group_last_cards[key] = (label, time.monotonic())
    except Exception as error:
        log.warning(f"[model_identity] 更新群 {group_id} 名片失败: {type(error).__name__}: {error}")
