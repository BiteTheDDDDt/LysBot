"""按用途选择模型：特殊工具、高用量任务、通用任务、全局备用。

业务 role 保留用于日志、超时和提示词策略，模型配置由这里集中映射。
具体模型 ID 只来自环境配置；高用量任务未单独配置时继承 MODEL_GENERAL。
特殊工具只使用自己的显式链，普通文本任务在链尾追加 MODEL_FALLBACK。
"""
from __future__ import annotations

import os

from .config import config
from .providers import (
    parse_chain,
    parse_spec,
    provider_configured,
    supports_image_generation,
    supports_tts,
    supports_vision,
    supports_web_search,
)


MODEL_GROUPS = (
    ("特殊工具", ("search", "vision", "tts", "image_gen")),
    ("高用量任务", ("chat", "planner", "summary", "important_info")),
    ("通用任务", ("general",)),
    ("全局备用", ("fallback",)),
)
HIGH_VOLUME_ROLES = frozenset({"chat", "planner", "summary", "important_info"})
SPECIAL_TOOL_ROLES = frozenset({"search", "vision", "tts", "image_gen"})
GENERAL_ROLES = ("general", "briefing", "news_interest", "jrrp", "keyword_extract", "image_driver")
ROLE_MODEL_KEYS = {
    "search": "MODEL_SEARCH",
    "vision": "MODEL_VISION",
    "tts": "MODEL_TTS",
    "image_gen": "MODEL_IMAGE_GEN",
    "chat": "MODEL_CHAT",
    "planner": "MODEL_PLANNER",
    "summary": "MODEL_SUMMARY",
    "important_info": "MODEL_IMPORTANT_INFO",
    **dict.fromkeys(GENERAL_ROLES, "MODEL_GENERAL"),
    "fallback": "MODEL_FALLBACK",
}


def env_key_for(role: str) -> str:
    try:
        return ROLE_MODEL_KEYS[role]
    except KeyError:
        raise ValueError(f"未知模型用途 role={role!r}") from None


def role_config_source(role: str) -> str:
    """返回实际读取的配置键，供状态展示与排错使用。"""
    key = env_key_for(role)
    if role in HIGH_VOLUME_ROLES and not os.environ.get(key, "").strip():
        return "MODEL_GENERAL"
    return key


def _raw_for(role: str) -> str:
    key = role_config_source(role)
    raw = os.environ.get(key, "").strip()
    if not raw:
        raise ValueError(f"role={role}: 缺少模型配置 {key}")
    return raw


def resolve_role(role: str) -> tuple[str, list[str]]:
    """按配置顺序返回主模型和备用链，重复项代表再次尝试。"""
    if role == "fallback" and not os.environ.get("MODEL_FALLBACK", "").strip():
        return "", []
    chain = parse_chain(_raw_for(role))
    if not chain:
        raise ValueError(f"role={role}: {role_config_source(role)} 的模型链不能为空")
    if role != "fallback" and role not in SPECIAL_TOOL_ROLES:
        raw = os.environ.get("MODEL_FALLBACK", "").strip()
        if raw:
            chain.extend(parse_chain(raw))
    return chain[0], chain[1:]


def all_roles() -> list[str]:
    return list(ROLE_MODEL_KEYS)


def model_allowed_for_role(role: str, spec: str) -> bool:
    """启动校验和运行时共用的工具能力边界。"""
    if role == "vision":
        return supports_vision(spec)
    if role == "search":
        return supports_web_search(spec)
    if role == "tts":
        return supports_tts(spec)
    if role == "image_gen":
        return supports_image_generation(spec)
    if role == "image_driver":
        try:
            provider, _ = parse_spec(spec)
        except ValueError:
            return False
        # 当前仅实现 CLIProxyAPI Responses 图片工具协议。模型是否支持该
        # 上游工具由部署者选择，不能通过模型名称猜测，也不能使用绘图模型本身。
        return provider == "cliproxyapi" and not supports_image_generation(spec)
    return True


def role_is_required(role: str) -> bool:
    return role not in SPECIAL_TOOL_ROLES | {"fallback", "image_driver"}


def role_available(role: str) -> bool:
    """Configuration is the sole source of optional model availability."""
    if role in SPECIAL_TOOL_ROLES:
        return bool(os.environ.get(env_key_for(role), "").strip())
    return bool(os.environ.get(role_config_source(role), "").strip())


def validate_all() -> list[str]:
    """校验配置来源、候选协议能力和至少一个可用 provider，不发出模型请求。"""
    legacy = sorted(key for key, value in os.environ.items() if key.startswith("ROLE_") and value.strip())
    if legacy:
        return [
            f"旧配置 {', '.join(legacy)} 已停用，请迁移到 MODEL_* 并移除旧键；"
            "参见 docs/configuration.md"
        ]
    if os.environ.get("MODEL_IMAGE_DRIVER", "").strip():
        return ["MODEL_IMAGE_DRIVER 已合并到 MODEL_GENERAL，请移除旧配置项；外层模型沿用通用模型和全局备用链"]

    errors: list[str] = []
    if config.enable_daily_news_broadcast:
        if not config.enable_daily_briefing:
            errors.append("ENABLE_DAILY_NEWS_BROADCAST requires ENABLE_DAILY_BRIEFING=true")
        if not role_available("search"):
            errors.append("ENABLE_DAILY_NEWS_BROADCAST requires MODEL_SEARCH")
    for _, group_roles in MODEL_GROUPS:
        for role in group_roles:
            key = env_key_for(role)
            if not role_is_required(role) and not os.environ.get(key, "").strip():
                continue
            try:
                primary, chain = resolve_role(role)
            except ValueError as exc:
                errors.append(str(exc))
                continue

            usable = []
            for spec in [primary, *chain]:
                try:
                    provider, _ = parse_spec(spec)
                except ValueError as exc:
                    errors.append(f"{key}: {exc}")
                    continue
                if not model_allowed_for_role(role, spec):
                    errors.append(f"{key}: {spec!r} 不支持当前 {role} 适配器所需的能力")
                    continue
                if provider_configured(provider):
                    usable.append(spec)
            if not usable:
                errors.append(f"{key}: 没有能力匹配且已配置 API key 的模型候选")

    return errors


def model_config_status() -> str:
    """按配置层展示继承后的链，避免把所有业务 role 重复展开。"""
    lines = []
    for label, group_roles in MODEL_GROUPS:
        lines.append(f"{label}:")
        for role in group_roles:
            key = env_key_for(role)
            source = role_config_source(role)
            inherited = "（继承 MODEL_GENERAL）" if source != key else ""
            try:
                primary, chain = resolve_role(role)
                value = " → ".join([primary, *chain]) if primary else "未配置"
            except ValueError:
                value = "未配置" if role_is_required(role) else "未启用 / 无需配置"
            lines.append(f"  · {key}{inherited}: {value}")
    return "\n".join(lines)
