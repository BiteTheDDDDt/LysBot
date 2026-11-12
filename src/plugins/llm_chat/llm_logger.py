"""LLM 调用专用日志器

用法（在 ModelManager.chat 里）::

    call = start_call(role=role, model=model_name, messages=messages, caller=caller)
    try:
        response = await adapter.chat(messages)
        finish_call(call, response=response)
        return response
    except Exception as e:
        finish_call(call, error=e)
        raise

文件输出落到 data/logs/llm/，纯文本，多行块（不是 jsonl）。
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from .logging_setup import llm_logger


_usage_tasks: set[asyncio.Task] = set()


async def drain_usage_tasks() -> None:
    tasks = list(_usage_tasks)
    if not tasks:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)
    except asyncio.TimeoutError:
        llm_logger.warning("[shutdown] 用量记录未能全部落盘")


PREVIEW_HEAD = 200
PREVIEW_TAIL = 80
SYSTEM_PROMPT_PREVIEW = 200
DEFAULT_RESPONSE_LOG_MAX_CHARS = 4000
PLANNER_CONTEXT_MARKER = "最近对话（"


def _short(text: str, head: int = PREVIEW_HEAD, tail: int = PREVIEW_TAIL) -> str:
    if text is None:
        return "<None>"
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= head + tail + 16:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}...<省略{omitted}字>...{text[-tail:]}"


def _planner_prompt_preview(text: str) -> str:
    """Keep the recent-context section visible in planner CALL logs."""
    marker_index = text.find(PLANNER_CONTEXT_MARKER)
    if marker_index < 0:
        return _short(text)

    rules_head = text[:PREVIEW_HEAD]
    context_block = text[marker_index:]
    context_preview = _short(context_block, head=280, tail=900)
    return (
        f"[planner_prompt len={len(text)}] {rules_head}"
        "...<规划规则省略，保留最近对话>...\n"
        f"{context_preview}"
    )


def _bounded(text: str, max_chars: int) -> tuple[str, bool]:
    if text is None:
        text = "<None>"
    if not isinstance(text, str):
        text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n...<响应日志截断，原始长度 {len(text)} 字，已保留前 {max_chars} 字>", True


def _indent_block(text: str) -> str:
    if not text:
        return "  <empty>"
    return "\n".join(f"  {line}" for line in text.splitlines())


def _hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]


def _format_message(msg: Any, idx: int, call_role: str = "") -> str:
    if hasattr(msg, "role"):
        role = getattr(msg, "role", "?")
        content = getattr(msg, "content", "") or ""
        name = getattr(msg, "name", None)
        has_images = bool(getattr(msg, "images", None))
    elif isinstance(msg, dict):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        parts.append("<image>")
            content = "".join(parts)
        name = msg.get("name")
        has_images = "image" in str(msg.get("content", "")).lower()
    else:
        role = "?"
        content = str(msg)
        name = None
        has_images = False

    content = str(content) if content is not None else ""
    if role == "system":
        body = f"[hash={_hash(content)} len={len(content)}] {_short(content, SYSTEM_PROMPT_PREVIEW, 40)}"
    elif call_role == "planner" and role == "user":
        body = _planner_prompt_preview(content)
    else:
        body = _short(content)

    name_part = f" name={name}" if name else ""
    img_part = " +images" if has_images else ""
    return f"  #{idx} [{role}{name_part}{img_part}] {body}"


@dataclass
class LLMCall:
    call_id: str
    role: str
    model: str
    caller: str
    session_id: str = "_unknown_"
    messages: list = field(default_factory=list)
    start_ts: float = field(default_factory=time.perf_counter)
    extra: dict = field(default_factory=dict)


def _new_call_id() -> str:
    return uuid.uuid4().hex[:8]


def start_call(
    *,
    role: str,
    model: str,
    messages: Iterable,
    caller: str = "?",
    session_id: str = "_unknown_",
    extra: dict | None = None,
) -> LLMCall:
    call = LLMCall(
        call_id=_new_call_id(),
        role=role,
        model=model,
        caller=caller,
        session_id=session_id,
        messages=list(messages),
        extra=dict(extra or {}),
    )
    formatted = "\n".join(
        _format_message(m, i, call.role)
        for i, m in enumerate(call.messages)
    )
    llm_logger.info(
        "[CALL>] id={cid} role={role} model={model} caller={caller} session={sid} msgs={n}\n{body}",
        cid=call.call_id, role=role, model=model, caller=caller, sid=session_id,
        n=len(call.messages), body=formatted or "  <no messages>",
    )
    return call


def finish_call(
    call: LLMCall,
    *,
    response: Any = None,
    tool_calls: list | None = None,
    fallback_used: str | None = None,
    fallback_attempt: int = 0,
    usage: dict | None = None,
    error: BaseException | None = None,
) -> None:
    import asyncio as _asyncio
    from .usage_db import record_usage

    latency_ms = int((time.perf_counter() - call.start_ts) * 1000)

    # 持久化（fire-and-forget），失败/成功都记
    try:
        loop = _asyncio.get_running_loop()
        task = loop.create_task(record_usage(
            session_id=call.session_id,
            role=call.role,
            spec=call.model,
            caller=call.caller,
            usage=usage,
            latency_ms=latency_ms,
            success=(error is None),
            fallback_attempt=fallback_attempt,
        ))
        _usage_tasks.add(task)
        task.add_done_callback(_usage_tasks.discard)
    except RuntimeError:
        # 没有运行中的事件循环（极少见，比如同步上下文调用），跳过持久化
        pass

    if error is not None:
        from .diagnostics import error_summary
        cancelled = isinstance(error, _asyncio.CancelledError)
        llm_logger.log(
            "WARNING" if cancelled else "ERROR",
            "[CALL<] id={cid} role={role} model={model} latency={ms}ms status={status} {detail}",
            cid=call.call_id, role=call.role, model=call.model, ms=latency_ms,
            status="cancelled" if cancelled else "failed", detail=error_summary(error),
        )
        return

    text = response if isinstance(response, str) else (str(response) if response is not None else "")
    if usage:
        usage_inline = (
            f" tokens=p{usage.get('prompt_tokens', 0)}+"
            f"c{usage.get('completion_tokens', 0)}=t{usage.get('total_tokens', 0)}"
        )
    else:
        usage_inline = ""
    tail_lines = []
    if tool_calls:
        tail_lines.append(f"  tool_calls={tool_calls}")
    if fallback_used:
        tail_lines.append(f"  fallback_used={fallback_used}")
    tail = ("\n" + "\n".join(tail_lines)) if tail_lines else ""

    llm_logger.info(
        "[CALL<] id={cid} role={role} model={model} latency={ms}ms{usage} len={l}{tail}",
        cid=call.call_id, role=call.role, model=call.model, ms=latency_ms,
        usage=usage_inline, l=len(text), tail=tail,
    )

    max_chars = DEFAULT_RESPONSE_LOG_MAX_CHARS
    response_body, truncated = _bounded(text, max_chars)
    llm_logger.info(
        "[CALL_RESPONSE] id={cid} role={role} model={model} len={l} truncated={truncated} max_chars={max_chars}\n{body}",
        cid=call.call_id,
        role=call.role,
        model=call.model,
        l=len(text),
        truncated=truncated,
        max_chars=max_chars,
        body=_indent_block(response_body),
    )


class NativeCall:
    """Log one native media request, including retries and cancellation.

    Only usage and a short status are recorded here, never binary media data.
    """
    def __init__(self, *, role, model, caller, fallback_attempt=0):
        from .conversation import current_scope
        from .model_types import current_session_id
        scope = current_scope()
        self.arguments = dict(role=role, model=model, caller=caller, messages=[],
                              session_id=scope.session_id if scope else current_session_id.get())
        self.fallback_attempt = fallback_attempt
        self.usage = None
        self.error = None
        self.call = None

    def __enter__(self):
        return self

    def start(self):
        if self.call is None:
            self.call = start_call(**self.arguments)

    def retry(self, error):
        finish_call(self.call, error=error, usage=self.usage, fallback_attempt=self.fallback_attempt)
        self.usage = None
        self.call = None

    def __exit__(self, error_type, error, traceback):
        if self.call is None:
            return
        finish_call(self.call, response="媒体请求完成" if error is None and self.error is None else None,
                    error=error or self.error, usage=self.usage, fallback_attempt=self.fallback_attempt)


def native_usage(payload):
    """Normalize supplied counters; missing values remain unknown."""
    if not isinstance(payload, dict):
        return None
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens", "promptTokenCount"),
        "completion_tokens": ("completion_tokens", "output_tokens", "candidatesTokenCount"),
        "total_tokens": ("total_tokens", "totalTokenCount"),
    }
    result = {}
    for target, names in aliases.items():
        for name in names:
            value = payload.get(name)
            if type(value) is int and value >= 0:
                result[target] = value
                break
    if "total_tokens" not in result and all(k in result for k in ("prompt_tokens", "completion_tokens")):
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result or None
