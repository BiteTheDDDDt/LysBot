"""Gemini text-to-speech generation helpers."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
import wave
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from nonebot import logger

from .config import config
from .llm_logger import NativeCall, native_usage
from .providers import get_provider_proxy, parse_spec, provider_configured, supports_tts
from .roles import resolve_role


from .config import PERSONA
DEFAULT_PHOS_VOICE_PERSONA = PERSONA.voice


EMOTION_PROMPTS = {
    "neutral": "Use the configured character conversational baseline; relaxed but never flat.",
    "cheerful": (
        "Let a smile enter the voice, becoming brighter and slightly quicker without sounding sugary or childish."
    ),
    "sad": (
        "Become quieter and more fragile, with restrained disappointment; "
        "avoid melodrama and preserve the configured vocal identity."
    ),
    "angry": "Sound stubborn, sharp, and a little impatient; keep control and do not turn it into sustained shouting.",
    "whisper": "Use a close, light whisper while preserving intelligibility and the configured vocal identity.",
    "excited": (
        "Speak faster with sparkling curiosity and barely contained enthusiasm; remain intelligible and natural."
    ),
    "sarcastic": (
        "Use playful, mischievous teasing with a small knowing smile; "
        "lightly cutting, never hostile or world-weary."
    ),
}

from .execution import ExecutionPolicy, Resources, execution_scope, operation
from .network_retry import retry_budget

VOICE_REQUEST = ExecutionPolicy("voice.request", permission="send",
    timeout=lambda: retry_budget(max(1.0, float(config.google_tts_request_timeout))))

_generation_semaphore = asyncio.Semaphore(max(1, int(config.google_tts_max_concurrency)))


class TTSGenerationError(RuntimeError):
    """Raised when Gemini does not return usable speech audio."""


@dataclass(frozen=True)
class GeneratedSpeech:
    """Audio ready to be sent as a QQ record message."""

    wav_bytes: bytes
    source_mime_type: str
    model_spec: str = ""


def _build_tts_prompt(text: str, emotion: str, voice_persona: str) -> str:
    style = EMOTION_PROMPTS.get(emotion, EMOTION_PROMPTS["neutral"])
    persona = (voice_persona or DEFAULT_PHOS_VOICE_PERSONA).strip()
    return (
        "Perform a single-speaker character voice from the transcript below. Speak only the transcript; "
        "never read the section labels, audio profile, or director notes aloud. Preserve the transcript's language, "
        "and use contemporary, natural Mandarin pronunciation for Chinese text.\n\n"
        "<AUDIO_PROFILE>\n"
        f"{persona}\n"
        "</AUDIO_PROFILE>\n\n"
        "<DIRECTOR_NOTES>\n"
        f"Current emotion: {style}\n"
        "The current emotion may color the performance, but it must not replace the core vocal identity.\n"
        "</DIRECTOR_NOTES>\n\n"
        "<TRANSCRIPT>\n"
        f"{text}\n"
        "</TRANSCRIPT>"
    )


def _extract_audio(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Extract and concatenate inline audio parts from generateContent output."""
    chunks: list[bytes] = []
    mime_type = ""

    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            encoded = inline.get("data")
            if not encoded:
                continue
            try:
                chunks.append(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError, TypeError) as exc:
                raise TTSGenerationError("Gemini 返回了无法解码的音频数据") from exc
            if not mime_type:
                mime_type = str(inline.get("mimeType") or inline.get("mime_type") or "")

    if not chunks:
        feedback = payload.get("promptFeedback") or payload.get("prompt_feedback") or {}
        block_reason = feedback.get("blockReason") or feedback.get("block_reason")
        suffix = f"，原因：{block_reason}" if block_reason else ""
        raise TTSGenerationError(f"Gemini 没有返回音频{suffix}")

    return b"".join(chunks), mime_type


def _pcm_to_wav(pcm: bytes, *, sample_rate: int = 24000) -> bytes:
    if not pcm:
        raise TTSGenerationError("Gemini 返回了空音频")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _audio_to_wav(audio: bytes, mime_type: str) -> bytes:
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return audio

    normalized_mime = (mime_type or "").lower()
    if normalized_mime and not any(marker in normalized_mime for marker in ("l16", "pcm", "wav")):
        raise TTSGenerationError(f"Gemini 返回了暂不支持的音频格式：{mime_type}")

    rate_match = re.search(r"rate=(\d+)", normalized_mime)
    sample_rate = int(rate_match.group(1)) if rate_match else 24000
    return _pcm_to_wav(audio, sample_rate=sample_rate)


async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    call: Optional[NativeCall] = None,
) -> httpx.Response:
    from .network_retry import post_with_retry
    try:
        return await post_with_retry(client, url, headers=headers, body=body,
                                     operation='voice.generate', call=call)
    except Exception as error:
        raise TTSGenerationError(f"请求失败：{error}") from error


async def _generate_google_speech(
    text: str,
    emotion: str,
    model_id: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    fallback_attempt: int = 0,
) -> GeneratedSpeech:
    """Generate speech with Google's native Gemini generateContent endpoint."""
    transcript = (text or "").replace("\x00", "").strip()
    if not transcript:
        raise TTSGenerationError("朗读内容不能为空")
    if emotion not in EMOTION_PROMPTS:
        raise TTSGenerationError(f"不支持的语气：{emotion}")

    api_key = str(config.google_api_key or "").strip()
    if not api_key:
        raise TTSGenerationError("未配置 GOOGLE_API_KEY")

    voice = str(config.google_tts_voice or "").strip()
    base_url = str(config.google_tts_base_url or "").rstrip("/")
    if not model_id or not voice or not base_url:
        raise TTSGenerationError("Gemini TTS 模型、音色或接口地址未配置完整")

    url = f"{base_url}/models/{model_id}:generateContent"
    body = {
        "contents": [{
            "parts": [{
                "text": _build_tts_prompt(
                    transcript,
                    emotion,
                    str(config.google_tts_persona_prompt or ""),
                )
            }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice},
                }
            },
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            proxy=get_provider_proxy("google"),
            timeout=max(1.0, float(config.google_tts_request_timeout)),
        )

    async with Resources() as resources:
        if owns_client:
            resources.own_client(client)
        call = resources.enter_context(NativeCall(role="tts", model=f"google:{model_id}",
                                                  caller="voice:generate", fallback_attempt=fallback_attempt))
        try:
            async with execution_scope(VOICE_REQUEST, slots=_generation_semaphore):
                response = await _post_with_retries(
                    client,
                    url,
                    headers=headers,
                    body=body,
                    call=call,
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise TTSGenerationError("Gemini TTS 返回的 JSON 不是对象")
            call.usage = native_usage(payload.get("usageMetadata"))
            audio, mime_type = _extract_audio(payload)
            wav_bytes = _audio_to_wav(audio, mime_type)
            usage = payload.get("usageMetadata") or {}
            logger.info(
                f"[tts] 生成成功 model={model_id} voice={voice} emotion={emotion} "
                f"chars={len(transcript)} wav_bytes={len(wav_bytes)} "
                f"output_tokens={usage.get('candidatesTokenCount', '?')}"
            )
            return GeneratedSpeech(
                wav_bytes=wav_bytes,
                source_mime_type=mime_type,
                model_spec=f"google:{model_id}",
            )
        except (ValueError, TypeError) as exc:
            raise TTSGenerationError(f"Gemini TTS 返回格式异常：{exc}") from exc


class TTSManager:
    """Resolve MODEL_TTS and execute provider-native speech adapters with fallback."""

    @staticmethod
    def _resolve_candidates() -> tuple[str, list[str]]:
        try:
            primary, fallback_chain = resolve_role("tts")
        except Exception as exc:
            raise TTSGenerationError(f"MODEL_TTS 路由解析失败：{exc}") from exc

        candidates: list[str] = []
        for spec in [primary] + fallback_chain:
            try:
                provider, _ = parse_spec(spec)
            except ValueError:
                continue
            if not provider_configured(provider) or not supports_tts(spec):
                continue
            candidates.append(spec)
        return primary, candidates

    async def generate(
        self,
        text: str,
        emotion: str = "neutral",
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> GeneratedSpeech:
        from .roles import role_available
        if not role_available("tts"):
            raise TTSGenerationError("MODEL_TTS 未配置")
        primary, candidates = self._resolve_candidates()
        if not candidates:
            raise TTSGenerationError(
                f"MODEL_TTS 主 spec={primary} 没有可用候选，请检查 provider API key 和 TTS 能力白名单"
            )

        last_error: Optional[Exception] = None
        for attempt, spec in enumerate(candidates):
            provider, model_id = parse_spec(spec)
            try:
                if provider == "google":
                    result = await _generate_google_speech(
                        text,
                        emotion,
                        model_id,
                        client=client,
                        fallback_attempt=attempt,
                    )
                else:
                    raise TTSGenerationError(f"TTS provider {provider!r} 尚未实现原生音频适配器")
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"[tts:fallback] primary={primary} spec={spec} 失败: {exc}; "
                    f"剩余={len(candidates) - attempt - 1}"
                )
                continue

            if attempt > 0:
                logger.warning(f"[tts:fallback] 主 {primary} 失败，成功落在 {spec} (尝试#{attempt})")
            return result

        raise TTSGenerationError(
            f"MODEL_TTS 全部 {len(candidates)} 个候选均失败，最后错误：{last_error}"
        ) from last_error


tts_manager = TTSManager()


@operation(ExecutionPolicy("voice.generate", permission="send", timeout=lambda: float(config.chat_media_turn_timeout)))
async def generate_speech(
    text: str,
    emotion: str = "neutral",
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> GeneratedSpeech:
    return await tts_manager.generate(text, emotion=emotion, client=client)
