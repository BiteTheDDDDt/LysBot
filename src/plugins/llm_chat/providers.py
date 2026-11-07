"""Provider 注册表 + spec 解析。

每个 provider 是一组「API 协议 + key/url 来源」。具体使用哪个模型由调用方传入
spec 字符串决定，格式为 `provider:model_id`，例如
`nvidia:moonshotai/kimi-k2.5`、`gemini:gemini-3-flash-preview`。

调用链：
    .env -> config 字段（API key/URL）
    providers.py -> PROVIDERS（字段映射 + 协议 + 能力位）
    roles.py -> 把 role 名解析成 spec 字符串列表
    models.py -> 按 spec 懒建 adapter
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple



@dataclass(frozen=True)
class ProviderSpec:
    name: str
    adapter_kind: str            # "openai" | "claude" | "volc"
    api_key_attr: str            # config 字段名
    base_url_attr: str
    default_extra: Optional[Dict[str, Any]] = None  # 每次请求附带的 extra_body
    supports_web_search: bool = False                # provider-wide native search; GPT is checked per model below
    auth_header: Optional[str] = None                # 非空则用此 header 名注入 api_key（如小米的 "api-key"），否则走标准 Authorization: Bearer


PROVIDERS: Dict[str, ProviderSpec] = {
    "deepseek":   ProviderSpec("deepseek",   "openai", "deepseek_api_key",   "deepseek_base_url"),
    "openai":     ProviderSpec("openai",     "openai", "openai_api_key",     "openai_base_url"),
    "cliproxyapi": ProviderSpec("cliproxyapi", "openai", "cliproxyapi_api_key", "cliproxyapi_base_url"),
    "nvidia":     ProviderSpec("nvidia",     "openai", "nvidia_api_key",     "nvidia_base_url"),
    "dashscope":  ProviderSpec("dashscope",  "openai", "dashscope_api_key",  "dashscope_base_url",
                               supports_web_search=True),
    "google":     ProviderSpec("google",     "openai", "google_api_key",     "google_base_url"),
    "openrouter": ProviderSpec("openrouter", "openai", "openrouter_api_key", "openrouter_base_url",
                               default_extra={"reasoning": {"enabled": True}}),
    "siliconflow": ProviderSpec("siliconflow", "openai", "siliconflow_api_key", "siliconflow_base_url"),
    "tencentmaas": ProviderSpec("tencentmaas", "openai", "tencentmaas_api_key", "tencentmaas_base_url"),
    "anthropic":  ProviderSpec("anthropic",  "claude", "anthropic_api_key",  "anthropic_base_url"),
    "volc":       ProviderSpec("volc",       "volc",   "volc_api_key",       "volc_base_url"),
    "xiaomimimo": ProviderSpec("xiaomimimo", "openai", "xiaomimimo_api_key", "xiaomimimo_base_url",
                               auth_header="api-key"),
}


# 已实测不支持 OpenAI tools 协议的 (provider, model_id 子串) 黑名单。
# 这里只保留具体模型例外；不要用空子串禁用整个 provider。
TOOL_UNSUPPORTED_RULES: List[Tuple[str, str]] = []

# 支持图片输入的 (provider, model_id 子串) 白名单
# 子串匹配大小写不敏感；如果以后想用别的视觉模型，需在这里加白。
VISION_RULES: List[Tuple[str, str]] = [
    ("cliproxyapi", "gpt-5."),
    ("cliproxyapi", "gpt-6-luna"),
    ("cliproxyapi", "gemini"),
    ("cliproxyapi", "claude"),
    ("nvidia", "gemma"),
    ("nvidia", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
    ("google", "gemma"),
    ("siliconflow", "qwen"),
    ("siliconflow", "VL"),
    ("siliconflow", "GLM-4.6V"),
    ("siliconflow", "GLM-4.5V"),
]

# Image-generation endpoints are not usable as chat vision models.
VISION_EXCLUDES: List[Tuple[str, str]] = [
    ("cliproxyapi", "image"),
]

# 支持文本转语音输出的 (provider, model_id 子串) 白名单。
# TTS 使用 provider 原生音频协议，不经过普通 OpenAI chat adapter。
TTS_RULES: List[Tuple[str, str]] = [
    ("google", "tts"),
]

# 支持专用图片生成端点的 (provider, model_id 子串) 白名单。
# 图片生成不经过普通 chat/vision adapter。
IMAGE_GENERATION_RULES: List[Tuple[str, str]] = [
    ("cliproxyapi", "gpt-image"),
    ("openai", "gpt-image"),
]


def parse_spec(spec: str) -> Tuple[str, str]:
    """`'nvidia:moonshotai/kimi-k2.5'` -> `('nvidia', 'moonshotai/kimi-k2.5')`"""
    if not isinstance(spec, str):
        raise ValueError(f"spec 必须是字符串: {spec!r}")
    s = spec.strip()
    if ":" not in s:
        raise ValueError(f"无效的 spec {spec!r}（应为 provider:model_id 格式）")
    provider, _, model_id = s.partition(":")
    provider = provider.strip()
    if provider == "codexproxy":
        provider = "cliproxyapi"
    model_id = model_id.strip()
    if not provider or not model_id:
        raise ValueError(f"无效的 spec {spec!r}（provider 或 model_id 为空）")
    if provider not in PROVIDERS:
        raise ValueError(f"未知 provider {provider!r}（已注册: {', '.join(PROVIDERS)}）")
    return provider, model_id


def parse_chain(raw: str) -> List[str]:
    """`'a:x | b:y | c:z'` -> `['a:x', 'b:y', 'c:z']`，过滤空项。"""
    if not raw:
        return []
    return [re.sub(r"^codexproxy\s*:", "cliproxyapi:", p.strip()) for p in raw.split("|") if p.strip()]


def provider_configured(provider: str) -> bool:
    """该 provider 是否在 .env 里配了 API key。"""
    from .config import config
    provider = "cliproxyapi" if provider == "codexproxy" else provider
    pspec = PROVIDERS.get(provider)
    if not pspec:
        return False
    return bool(getattr(config, pspec.api_key_attr, ""))


def get_provider_proxy(provider: str) -> Optional[str]:
    """读取 provider 级代理配置。

    示例：
      PROVIDER_PROXY_GOOGLE=http://127.0.0.1:7890
      PROVIDER_PROXY_OPENROUTER=http://127.0.0.1:7890

    未配置则直连；值为 none/direct/false/0 也视为直连。
    """
    provider = "cliproxyapi" if provider == "codexproxy" else provider
    key = f"PROVIDER_PROXY_{provider.upper().replace('-', '_')}"
    raw = os.getenv(key, "").strip()
    if not raw and provider == "cliproxyapi":
        raw = os.getenv("PROVIDER_PROXY_CODEXPROXY", "").strip()
    raw = raw or os.getenv("PROVIDER_PROXY_ALL", "").strip()
    if raw.lower() in {"", "none", "null", "direct", "false", "0"}:
        return None
    return raw


def configured_providers() -> List[str]:
    return [p for p in PROVIDERS if provider_configured(p)]


def model_supports_tools(spec: str) -> bool:
    """该 spec 是否能用 OpenAI tools / function calling 协议。"""
    try:
        provider, model_id = parse_spec(spec)
    except ValueError:
        return False
    for p, sub in TOOL_UNSUPPORTED_RULES:
        if provider == p and (not sub or sub in model_id):
            return False
    return True


def supports_vision(spec: str) -> bool:
    try:
        provider, model_id = parse_spec(spec)
    except ValueError:
        return False
    model_id_lower = model_id.lower()
    for p, sub in VISION_EXCLUDES:
        if provider == p and (not sub or sub.lower() in model_id_lower):
            return False
    for p, sub in VISION_RULES:
        if provider == p and (not sub or sub.lower() in model_id_lower):
            return True
    return False


def supports_tts(spec: str) -> bool:
    try:
        provider, model_id = parse_spec(spec)
    except ValueError:
        return False
    model_id_lower = model_id.lower()
    for provider_name, model_substring in TTS_RULES:
        if provider == provider_name and (
            not model_substring or model_substring.lower() in model_id_lower
        ):
            return True
    return False


def supports_image_generation(spec: str) -> bool:
    try:
        provider, model_id = parse_spec(spec)
    except ValueError:
        return False
    model_id_lower = model_id.lower()
    for provider_name, model_substring in IMAGE_GENERATION_RULES:
        if provider == provider_name and (
            not model_substring or model_substring.lower() in model_id_lower
        ):
            return True
    return False


def web_search_backend(spec: str) -> Optional[str]:
    """Only allow models whose native search protocol is implemented here.

    CLIProxyAPI also serves Gemini/Grok/image models: an OpenAI-compatible chat
    endpoint alone does not imply Responses web_search support.
    """
    try:
        provider, model_id = parse_spec(spec)
    except ValueError:
        return None
    if provider == "dashscope":
        return "dashscope"
    if provider in {"openai", "cliproxyapi"} and re.fullmatch(
        r"gpt-(?:6-(?:astra|luna)|5\.6-(?:sol|terra|luna)|5\.5)(?:-\d{4}-\d{2}-\d{2})?",
        model_id.lower(),
    ):
        return "openai_responses"
    return None


def supports_web_search(spec: str) -> bool:
    return web_search_backend(spec) is not None
