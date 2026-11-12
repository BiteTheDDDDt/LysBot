"""Shared model messages, responses, usage and per-task session context.

This module creates no provider clients or model manager.
"""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .providers import parse_spec
from .text_safety import unicode_safe


REQUEST_TIMEOUT_EXTRA_KEY = "_request_timeout"
REQUEST_MAX_RETRIES_EXTRA_KEY = "_request_max_retries"


class ModelTurnTimeout(asyncio.TimeoutError):
    """The complete model/tool turn exceeded its configured wall-clock budget."""

    def __init__(self, *, role: str, required_tool: Optional[str], timeout: float):
        self.required_tool = required_tool
        self.timeout = timeout
        super().__init__(
            f"模型回合超时：role={role} required_tool={required_tool or '-'} 上限={timeout:g}秒"
        )

    @property
    def user_message(self) -> str:
        if self.required_tool == "generate_image":
            return "这次图片生成等太久了，还没拿到结果。"
        if self.required_tool == "send_voice":
            return "这次语音生成等太久了，还没拿到结果。"
        return "这次处理超时了，稍后再试一下。"


# 跨 await 传递当前会话 id；嵌套 LLM 调用（如工具内 model_manager.chat）自动继承
current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)


@dataclass
class Message:
    """消息结构"""
    role: str  # system, user, assistant
    content: str
    name: str = None
    images: Optional[List[str]] = None
    is_summary: bool = False
    is_thinking: bool = False
    is_reference: bool = False
    timestamp: Optional[str] = None
    reasoning_details: Optional[Any] = None
    sender_id: Optional[int] = None
    source_id: Optional[int] = None
    memory_refs: Optional[List[dict]] = None
    model_spec: Optional[str] = None  # Actual text producer, not the configured role default.

    def __post_init__(self):
        for field in ("role", "content", "name", "images", "timestamp", "reasoning_details"):
            setattr(self, field, unicode_safe(getattr(self, field)))


class ChatResponse(str):
    def __new__(cls, text: str, reasoning_details: Optional[Any] = None,
                usage: Optional[Dict[str, int]] = None,
                model_spec: Optional[str] = None,
                primary_spec: Optional[str] = None,
                fallback_attempt: int = 0,
                role: Optional[str] = None,
                caller: Optional[str] = None,
                session_id: Optional[str] = None):
        if text is None:
            text = ""
        obj = str.__new__(cls, unicode_safe(text))
        obj.reasoning_details = unicode_safe(reasoning_details)
        obj.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        obj.model_spec = model_spec
        obj.primary_spec = primary_spec
        obj.fallback_attempt = fallback_attempt
        obj.role = role
        obj.caller = caller
        obj.session_id = session_id
        obj.provider = None
        obj.model_id = None
        if model_spec:
            try:
                provider, model_id = parse_spec(model_spec)
                obj.provider = provider
                obj.model_id = model_id
            except ValueError:
                pass
        return obj


def _attach_response_context(
    resp: str,
    *,
    model_spec: str,
    primary_spec: str,
    fallback_attempt: int,
    role: Optional[str],
    caller: str,
    session_id: str,
) -> ChatResponse:
    if not isinstance(resp, ChatResponse):
        resp = ChatResponse(
            str(resp or ""),
            reasoning_details=getattr(resp, "reasoning_details", None),
            usage=getattr(resp, "usage", None),
        )
    provider, model_id = parse_spec(model_spec)
    resp.model_spec = model_spec
    resp.primary_spec = primary_spec
    resp.fallback_attempt = fallback_attempt
    resp.role = role or "ad-hoc"
    resp.caller = caller
    resp.session_id = session_id
    resp.provider = provider
    resp.model_id = model_id
    return resp


def _extract_openai_usage(response: Any) -> Dict[str, int]:
    """从 OpenAI/OpenAI-兼容 SDK 的 response 中提取 usage。返回 0 而不是 None。"""
    u = getattr(response, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    tt = getattr(u, "total_tokens", 0) or (pt + ct)
    return {"prompt_tokens": int(pt), "completion_tokens": int(ct), "total_tokens": int(tt)}


def _add_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
    }
