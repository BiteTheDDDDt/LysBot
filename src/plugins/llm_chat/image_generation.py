"""Role-routed image generation adapters.

Image output uses provider-native endpoints instead of the chat adapter.  The
initial implementation targets the OpenAI-compatible Images API exposed by
the local CLIProxyAPI service.
"""
from __future__ import annotations

import asyncio
import base64
from datetime import date
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from nonebot import logger

from .config import config
from .llm_logger import NativeCall, native_usage
from .providers import (
    PROVIDERS,
    get_provider_proxy,
    parse_spec,
    provider_configured,
)
from .roles import env_key_for, model_allowed_for_role, resolve_role


IMAGE_ORIENTATION_SIZES = {
    "square": "1024x1024",
    "landscape": "1536x1024",
    "portrait": "1024x1536",
}
ALLOWED_IMAGE_SIZES = set(IMAGE_ORIENTATION_SIZES.values())
from .execution import ExecutionPolicy, Resources, execution_scope, operation
from .network_retry import retry_budget

IMAGE_REQUEST = ExecutionPolicy("image.request", permission="send",
    timeout=lambda: retry_budget(max(1.0, float(config.image_gen_request_timeout))))

_generation_semaphore = asyncio.Semaphore(max(1, int(config.image_gen_max_concurrency)))
_daily_usage = {"date": "", "count": 0}


def _reserve_generation():
    today = date.today().isoformat()
    if _daily_usage["date"] != today:
        _daily_usage.update(date=today, count=0)
    limit = int(config.image_gen_daily_limit)
    if limit > 0 and _daily_usage["count"] >= limit:
        raise ImageGenerationError(f"今日进程内生图额度已用完，上限 {limit} 张")
    # No await between checking and reserving; failed requests may still be billed.
    _daily_usage["count"] += 1


class ImageGenerationError(RuntimeError):
    """Raised when an image provider does not return a usable image."""


@dataclass(frozen=True)
class GeneratedImage:
    image_bytes: bytes
    mime_type: str
    model_spec: str
    size: str
    revised_prompt: str = ""
    driver_model_spec: str = ""  # Legacy result field; direct image requests leave this empty.


def resolve_image_size(orientation: str = "auto") -> str:
    normalized = str(orientation or "auto").strip().lower()
    if normalized == "auto":
        size = str(config.image_gen_default_size or "1024x1024").strip().lower()
    else:
        size = IMAGE_ORIENTATION_SIZES.get(normalized, "")
    if size not in ALLOWED_IMAGE_SIZES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_SIZES))
        raise ImageGenerationError(f"不支持的图片尺寸 {size!r}，可选：{allowed}")
    return size


def _detect_image_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ImageGenerationError("图片接口返回了无法识别的图片格式")


def _validate_image(image_bytes: bytes) -> str:
    if not image_bytes:
        raise ImageGenerationError("图片接口返回了空文件")
    max_bytes = max(1, int(config.image_gen_max_bytes))
    if len(image_bytes) > max_bytes:
        raise ImageGenerationError(
            f"生成图片大小 {len(image_bytes)} bytes 超过限制 {max_bytes} bytes"
        )
    return _detect_image_mime(image_bytes)


def _decode_image_data(encoded: str) -> bytes:
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ImageGenerationError("图片接口返回的 base64 数据无效") from exc


async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    files: Optional[list] = None,
    call: Optional[NativeCall] = None,
) -> httpx.Response:
    from .network_retry import post_with_retry
    try:
        return await post_with_retry(client, url, headers=headers, body=body,
                                     operation='image.generate', files=files, call=call)
    except Exception as error:
        raise ImageGenerationError(f"请求失败：{error}") from error


async def _download_remote_image(url: str, client: httpx.AsyncClient) -> bytes:
    """Download a provider-returned image without letting it reach private hosts.

    The provider client is reused so an accounted proxy still applies; each hop
    is resolved and checked before it is requested, and the body is streamed
    with the configured size cap.
    """
    from .public_net import fetch_pinned_bytes

    try:
        data, _ = await fetch_pinned_bytes(
            url, client=client, max_bytes=max(1, int(config.image_gen_max_bytes)),
        )
        return data
    except (ValueError, OSError) as exc:
        raise ImageGenerationError(f"生成图片来源不被允许：{exc}") from exc
    except httpx.HTTPError as exc:
        raise ImageGenerationError(f"下载生成图片失败：{exc}") from exc


async def _extract_generated_image(
    payload: dict[str, Any],
    client: httpx.AsyncClient,
) -> tuple[bytes, str]:
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        encoded = item.get("b64_json") or item.get("base64") or item.get("b64")
        if encoded:
            return _decode_image_data(str(encoded)), str(item.get("revised_prompt") or "")
        remote_url = str(item.get("url") or "").strip()
        if remote_url:
            return await _download_remote_image(remote_url, client), str(item.get("revised_prompt") or "")
    raise ImageGenerationError("图片接口响应中没有 b64_json 或图片 URL")




def _provider_connection(provider: str) -> tuple[str, dict[str, str]]:
    provider_spec = PROVIDERS[provider]
    api_key = str(getattr(config, provider_spec.api_key_attr, "") or "").strip()
    base_url = str(getattr(config, provider_spec.base_url_attr, "") or "").rstrip("/")
    if not api_key or not base_url:
        raise ImageGenerationError(f"provider {provider!r} 的 API key 或 base URL 未配置")

    headers = {"Content-Type": "application/json"}
    if provider_spec.auth_header:
        headers[provider_spec.auth_header] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return f"{base_url}/images/generations", headers


async def _generate_openai_compatible_image(
    prompt: str,
    size: str,
    provider: str,
    model_id: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    reference_images: tuple[bytes, ...] = (),
    fallback_attempt: int = 0,
) -> GeneratedImage:
    endpoint, headers = _provider_connection(provider)
    body = {
        "model": model_id,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "output_format": "png",
    }
    files = None
    if reference_images:
        endpoint = endpoint.removesuffix("/generations") + "/edits"
        headers.pop("Content-Type", None)  # httpx supplies the multipart boundary.
        extensions = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
        files = [
            ("image[]", (f"reference-{index}.{extensions[_detect_image_mime(raw)]}", raw, _detect_image_mime(raw)))
            for index, raw in enumerate(reference_images, 1)
        ]

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            proxy=get_provider_proxy(provider),
            timeout=max(1.0, float(config.image_gen_request_timeout)),
        )

    async with Resources() as resources:
        if owns_client:
            resources.own_client(client)
        call = resources.enter_context(NativeCall(role="image_gen", model=f"{provider}:{model_id}",
                                                  caller="image:generate", fallback_attempt=fallback_attempt))
        try:
            async with execution_scope(IMAGE_REQUEST, slots=_generation_semaphore):
                response = await _post_with_retries(
                    client,
                    endpoint,
                    headers=headers,
                    body=body,
                    files=files,
                    call=call,
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise ImageGenerationError("图片接口返回的 JSON 不是对象")
            call.usage = native_usage(payload.get("usage"))
            image_bytes, revised_prompt = await _extract_generated_image(payload, client)
            mime_type = _validate_image(image_bytes)
            logger.info(
                f"[image_gen] 生成成功 spec={provider}:{model_id} size={size} "
                f"prompt_chars={len(prompt)} bytes={len(image_bytes)} mime={mime_type}"
            )
            return GeneratedImage(
                image_bytes=image_bytes,
                mime_type=mime_type,
                model_spec=f"{provider}:{model_id}",
                size=size,
                revised_prompt=revised_prompt,
            )
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError(f"图片接口返回格式异常：{exc}") from exc




class ImageGenerationManager:
    """Resolve image models separately; Responses drivers share the general chain."""

    @staticmethod
    def _resolve_candidates(role: str = "image_gen") -> tuple[str, list[str]]:
        try:
            primary, fallback_chain = resolve_role(role)
        except Exception as exc:
            raise ImageGenerationError(f"{env_key_for(role)} 路由解析失败：{exc}") from exc

        candidates: list[str] = []
        for spec in [primary] + fallback_chain:
            try:
                provider, _ = parse_spec(spec)
            except ValueError:
                continue
            if not provider_configured(provider) or not model_allowed_for_role(role, spec):
                continue
            candidates.append(spec)
        return primary, candidates


    async def generate(
        self,
        prompt: str,
        orientation: str = "auto",
        *,
        client: Optional[httpx.AsyncClient] = None,
        reference_images: Optional[list[str]] = None,
    ) -> GeneratedImage:
        from .roles import role_available
        if not role_available("image_gen"):
            raise ImageGenerationError("当前未启用 AI 生图功能")
        clean_prompt = (prompt or "").replace("\x00", "").strip()
        if not clean_prompt:
            raise ImageGenerationError("图片描述不能为空")
        max_chars = max(1, int(config.image_gen_max_prompt_chars))
        if len(clean_prompt) > max_chars:
            raise ImageGenerationError(f"图片描述超过 {max_chars} 字")
        size = resolve_image_size(orientation)
        if reference_images is not None and not isinstance(reference_images, list):
            raise ImageGenerationError("参考图片必须是 base64 列表")
        if len(reference_images or []) > max(1, int(config.incoming_images_limit)):
            raise ImageGenerationError("参考图片过多，请明确选择需要使用的图片")
        originals = tuple(_decode_image_data(encoded) for encoded in (reference_images or []))
        for raw in originals:
            if len(raw) > max(int(config.image_download_max_bytes), int(config.image_gen_max_bytes)):
                raise ImageGenerationError("参考图片超过大小限制")
            _detect_image_mime(raw)

        primary, candidates = self._resolve_candidates()
        if not candidates:
            raise ImageGenerationError(
                f"MODEL_IMAGE_GEN 主 spec={primary} 没有可用候选，"
                "请检查 provider API key 和图片生成能力白名单"
            )

        _reserve_generation()
        last_error: Optional[Exception] = None
        for attempt, spec in enumerate(candidates):
            provider, model_id = parse_spec(spec)
            try:
                if provider in {"cliproxyapi", "openai"}:
                    result = await _generate_openai_compatible_image(
                        clean_prompt,
                        size,
                        provider,
                        model_id,
                        client=client,
                        fallback_attempt=attempt,
                        reference_images=originals,
                    )
                else:
                    raise ImageGenerationError(
                        f"图片生成 provider {provider!r} 尚未实现原生适配器"
                    )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[image_gen:fallback] primary={primary} spec={spec} 失败: {exc}; "
                    f"剩余={len(candidates) - attempt - 1}"
                )
                continue

            if attempt > 0:
                logger.warning(
                    f"[image_gen:fallback] 主 {primary} 失败，成功落在 {spec} "
                    f"(尝试#{attempt})"
                )
            return result

        raise ImageGenerationError(
            f"MODEL_IMAGE_GEN 全部 {len(candidates)} 个候选均失败，最后错误：{last_error}"
        ) from last_error


image_generation_manager = ImageGenerationManager()


@operation(ExecutionPolicy("image.generate", permission="send", timeout=lambda: float(config.chat_media_turn_timeout)))
async def generate_image(
    prompt: str,
    orientation: str = "auto",
    *,
    client: Optional[httpx.AsyncClient] = None,
    reference_images: Optional[list[str]] = None,
) -> GeneratedImage:
    return await image_generation_manager.generate(
        prompt,
        orientation=orientation,
        client=client,
        reference_images=reference_images,
    )
