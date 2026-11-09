"""多模型统一适配器 + ModelManager。

所有模型通过 spec 字符串引用，格式 `provider:model_id`，例如
`nvidia:moonshotai/kimi-k2.5`。Adapter 按 spec 懒建并缓存。

业务调用 `model_manager.chat(messages, role="judge", caller="...", session_id="...")`，
路由由 `roles.resolve_role(role)` 决定主+回退链。
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from contextvars import ContextVar
from collections import OrderedDict
import asyncio
import math
import time
import re
import hashlib
import httpx
from openai import AsyncOpenAI

from .diagnostics import error_summary
from .config import config, get_tool_usage_prompt, NO_TOOL_PROMPT
from .llm_logger import start_call, finish_call
from .logging_setup import general_logger
from .roles import model_allowed_for_role, resolve_role
from .streaming import collect_completion
from .response_safety import INVALID_REPLY, is_tool_call_text
from .text_safety import unicode_safe
from .providers import (
    PROVIDERS,
    parse_spec,
    provider_configured,
    configured_providers,
    model_supports_tools,
    supports_vision,
    web_search_backend,
    get_provider_proxy,
)


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


class ModelAdapter(ABC):
    """模型适配器基类。`model_name` 为 spec 字符串。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        extra_body: Optional[Dict[str, Any]] = None,
        tools_enabled: bool = False,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
    ) -> str:
        pass

    @abstractmethod
    async def chat_with_images(
        self,
        messages: List[Message],
        images: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
        tools_enabled: bool = False,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
    ) -> str:
        pass


_FAKE_TOOL_CALL_PATTERNS = [
    re.compile(r"<\|?tool_call\|?>.*?<\|?/tool_call\|?>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|?tool_call\|?>.*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<function\s*=.*?</function>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<function\s*=.*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<parameter\s*=.*?</parameter>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<parameter\s*=.*", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tools?>.*?</tools?>", re.DOTALL | re.IGNORECASE),
    re.compile(r"```tool_call.*?```", re.DOTALL | re.IGNORECASE),
    re.compile(r"```tool_code.*?```", re.DOTALL | re.IGNORECASE),
]


def _strip_fake_tool_calls(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for pat in _FAKE_TOOL_CALL_PATTERNS:
        cleaned = pat.sub("", cleaned)
    return cleaned.strip()


class OpenAICompatibleAdapter(ModelAdapter):
    """OpenAI 兼容接口适配器。"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        model_id: str,
        extra_body: Optional[Dict[str, Any]] = None,
        auth_header: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.model_id = model_id
        try:
            self._provider, _ = parse_spec(model_name)
        except ValueError:
            self._provider = ""
        request_timeout = max(1.0, float(config.llm_request_timeout))
        max_retries = 0  # network_retry owns retries; never multiply SDK attempts.
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": request_timeout,
            "max_retries": max_retries,
        }
        self._http_client = None
        if proxy_url:
            self._http_client = httpx.AsyncClient(proxy=proxy_url, timeout=request_timeout)
            client_kwargs["http_client"] = self._http_client
        elif self._uses_buffered_stream():
            # SDK 3.7's default httpx2 transport can leave an async generator
            # open at SSE completion. Use the existing stable httpx dependency.
            self._http_client = httpx.AsyncClient(timeout=request_timeout, trust_env=False)
            client_kwargs["http_client"] = self._http_client
        if auth_header:
            client_kwargs["default_headers"] = {auth_header: api_key}
        self.client = AsyncOpenAI(**client_kwargs)
        self.default_extra_body = dict(extra_body) if extra_body else {}

    def _uses_buffered_stream(self) -> bool:
        return self._provider == "cliproxyapi" and self.model_id.lower() == "gpt-oss-120b-medium"

    async def _create_completion(self, client, kwargs: Dict[str, Any]):
        # Antigravity's non-streaming gpt-oss endpoint rejects an upstream-added
        # stream_options field (reproduced with CLIProxyAPI v7.2.149). Only this
        # confirmed route uses SSE; callers still receive a complete response.
        if not self._uses_buffered_stream():
            return await client.chat.completions.create(**kwargs)

        stream_kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        async with await client.chat.completions.create(**stream_kwargs) as raw_stream:
            return await collect_completion(raw_stream)

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            msg_dict = {"role": msg.role}
            if hasattr(msg, "reasoning_details") and msg.reasoning_details:
                msg_dict["reasoning_details"] = msg.reasoning_details
            if msg.name:
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", msg.name)
                if not safe_name or safe_name != msg.name:
                    name_hash = hashlib.md5(msg.name.encode("utf-8")).hexdigest()[:8]
                    msg_dict["name"] = f"user_{name_hash}"
                else:
                    msg_dict["name"] = safe_name[:64]
            if msg.images:
                content = [{"type": "text", "text": msg.content}]
                for img in msg.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    })
                msg_dict["content"] = content
            else:
                msg_dict["content"] = msg.content
            formatted.append(msg_dict)
        return formatted

    def prepare_request(
        self,
        messages: List[Message],
        extra_body: Optional[Dict[str, Any]] = None,
        tools_enabled: bool = False,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
    ):
        from .tools import get_chat_tools

        formatted = self._format_messages(messages)

        supports_tools = tools_enabled and model_supports_tools(self.model_name)
        if required_tool and not supports_tools:
            raise RuntimeError(
                f"模型 {self.model_name} 不支持 required_tool={required_tool}"
            )
        selected_tools = get_chat_tools(required_tool) if supports_tools else []
        tool_addendum = (
            get_tool_usage_prompt(
                required_tool or "",
                media_followup=media_followup,
            )
            if supports_tools
            else NO_TOOL_PROMPT
        )
        for i in range(len(formatted) - 1, -1, -1):
            if formatted[i].get("role") == "system" and isinstance(formatted[i].get("content"), str):
                formatted[i] = {**formatted[i], "content": formatted[i]["content"] + tool_addendum}
                break

        kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": formatted,
            "stream": False,
            "temperature": config.model_temperature,
            "top_p": config.model_top_p,
            "frequency_penalty": config.model_frequency_penalty,
            "presence_penalty": config.model_presence_penalty,
            "max_tokens": config.model_max_tokens,
        }

        if self._provider == "tencentmaas" and self.model_id.lower() == "kimi-k3":
            # Kimi K3 的采样参数是固定值，官方建议不要显式传入；输出上限
            # 也使用 max_completion_tokens，而不是兼容模型常用的 max_tokens。
            for fixed_param in (
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ):
                kwargs.pop(fixed_param, None)
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        if self._provider == "google":
            kwargs.pop("frequency_penalty", None)
            kwargs.pop("presence_penalty", None)

        if supports_tools:
            kwargs["tools"] = selected_tools
            # NVIDIA NIM 要求 tools 与 tool_choice 配套传入。
            if self._provider == "nvidia":
                kwargs["tool_choice"] = "auto"

        merged_extra: Dict[str, Any] = {}
        if self.default_extra_body:
            merged_extra.update(self.default_extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        request_timeout = merged_extra.pop(REQUEST_TIMEOUT_EXTRA_KEY, None)
        request_max_retries = merged_extra.pop(REQUEST_MAX_RETRIES_EXTRA_KEY, None)
        if merged_extra:
            kwargs["extra_body"] = merged_extra

        client = self.client
        request_options: Dict[str, Any] = {}
        if request_timeout is not None:
            request_options["timeout"] = max(1.0, float(request_timeout))
            kwargs["timeout"] = request_options["timeout"]
        if request_max_retries is not None:
            request_options["max_retries"] = 0
        if request_options:
            client = self.client.with_options(**request_options)
        return client, kwargs, formatted, supports_tools, selected_tools

    async def chat(self, messages, extra_body=None, tools_enabled=False,
                   required_tool=None, media_followup=True) -> str:
        from .agent_runner import AgentRunner, current_run
        active = current_run.get() if tools_enabled else None
        runner = active or AgentRunner(current_session_id.get() or "_unknown_")
        token = current_run.set(runner) if tools_enabled and active is None else None
        completed=False
        try:
            if token is not None:
                await runner.start(messages)
            response=await runner.run(self,messages,extra_body,tools_enabled,required_tool,media_followup)
            completed=True
            return response
        finally:
            if token is not None:
                try:
                    await runner.finish("complete" if completed else "interrupted")
                finally:
                    current_run.reset(token)

    async def chat_with_images(
        self,
        messages: List[Message],
        images: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
        tools_enabled: bool = False,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
    ) -> str:
        if messages and messages[-1].role == "user":
            messages[-1].images = images
        return await self.chat(
            messages,
            extra_body=extra_body,
            tools_enabled=tools_enabled,
            required_tool=required_tool,
            media_followup=media_followup,
        )


class VolcAdapter(OpenAICompatibleAdapter):
    """火山引擎豆包适配器，重写 _format_messages 以适配多模态消息顺序。"""

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        formatted = []
        for msg in messages:
            if msg.images:
                content = []
                for img in msg.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    })
                content.append({"type": "text", "text": msg.content})
                formatted.append({"role": msg.role, "content": content})
            else:
                formatted.append({"role": msg.role, "content": msg.content})
        return formatted


class ClaudeAdapter(ModelAdapter):
    """Claude (Anthropic 原生协议) 适配器"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        model_id: str,
        proxy_url: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model_id = model_id
        self.proxy_url = proxy_url

    async def chat(
        self,
        messages: List[Message],
        extra_body: Optional[Dict[str, Any]] = None,
        tools_enabled: bool = False,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
    ) -> str:
        if required_tool:
            raise RuntimeError(
                f"Claude 原生适配器暂不支持 required_tool={required_tool}"
            )
        system_parts = []
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})
        system_prompt = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": chat_messages,
        }
        if extra_body:
            payload.update(extra_body)

        async with httpx.AsyncClient(proxy=self.proxy_url) as client:
            from .network_retry import request_with_retry
            async def request():
                response = await client.post(f"{self.base_url}/v1/messages", headers=headers, json=payload, timeout=60.0)
                response.raise_for_status()
                return response
            response = await request_with_retry(request, operation=self.model_name)
            data = response.json()
            text = data.get("content", [{}])[0].get("text", "")
            u = data.get("usage") or {}
            pt = int(u.get("input_tokens", 0) or 0)
            ct = int(u.get("output_tokens", 0) or 0)
            usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
            return ChatResponse(text, usage=usage)

    async def chat_with_images(
        self,
        messages: List[Message],
        images: List[str],
        extra_body: Optional[Dict[str, Any]] = None,
        tools_enabled: bool = False,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
    ) -> str:
        if messages and messages[-1].role == "user":
            messages[-1].images = images
        return await self.chat(
            messages,
            extra_body=extra_body,
            tools_enabled=tools_enabled,
            required_tool=required_tool,
            media_followup=media_followup,
        )


def _quota_reset_delay(error: Exception) -> float:
    """Use structured provider cooldowns, never guess from error text."""
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        body = getattr(current, "body", None)
        detail = body.get("error", body) if isinstance(body, dict) else None
        if getattr(current, "status_code", None) == 429 and isinstance(detail, dict) and detail.get("code") == "model_cooldown":
            seconds = detail.get("reset_seconds")
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds > 0:
                return float(seconds)
        current = current.__cause__
    return 0.0


class ModelManager:
    """模型管理器：按 spec 懒建 adapter；提供 role 路由 + 多级回退。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters: Dict[str, ModelAdapter] = {}
            cls._instance._model_cooldowns = {}
            cls._instance._current_model_override: Optional[str] = None
            cls._instance._vision_analysis_cache: OrderedDict[str, dict] = OrderedDict()
            cls._instance._vision_pending: Dict[str, asyncio.Task[dict]] = {}
        return cls._instance

    # -------- spec / adapter 管理 --------
    def get_or_build_adapter(self, spec: str) -> ModelAdapter:
        provider, model_id = parse_spec(spec)
        spec = f"{provider}:{model_id}"
        if spec in self._adapters:
            return self._adapters[spec]
        pspec = PROVIDERS[provider]
        api_key = getattr(config, pspec.api_key_attr, "")
        base_url = getattr(config, pspec.base_url_attr, "")
        proxy_url = get_provider_proxy(provider)
        if not api_key:
            raise ValueError(
                f"provider {provider!r} 未配置 API key (需 .env 设置 {pspec.api_key_attr.upper()})"
            )

        if pspec.adapter_kind == "openai":
            adapter = OpenAICompatibleAdapter(
                model_name=spec,
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                extra_body=dict(pspec.default_extra) if pspec.default_extra else None,
                auth_header=pspec.auth_header,
                proxy_url=proxy_url,
            )
        elif pspec.adapter_kind == "claude":
            adapter = ClaudeAdapter(
                model_name=spec,
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                proxy_url=proxy_url,
            )
        elif pspec.adapter_kind == "volc":
            adapter = VolcAdapter(
                model_name=spec,
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                proxy_url=proxy_url,
            )
        else:
            raise ValueError(f"未知 adapter_kind: {pspec.adapter_kind}")

        self._adapters[spec] = adapter
        return adapter

    @property
    def current_model(self) -> str:
        if self._current_model_override:
            return self._current_model_override
        primary, _ = resolve_role("chat")
        return primary

    @current_model.setter
    def current_model(self, spec: str):
        provider, model_id = parse_spec(spec)
        spec = f"{provider}:{model_id}"
        if not provider_configured(provider):
            raise ValueError(f"provider {provider!r} 未配置 API key，无法切换到 {spec}")
        self.get_or_build_adapter(spec)
        self._current_model_override = spec

    @property
    def available_providers(self) -> List[str]:
        return configured_providers()

    def get_model_display_name(self, spec: str) -> str:
        try:
            provider, model_id = parse_spec(spec)
            from .model_identity import format_provider_label
            return f"{format_provider_label(provider, model_id)}:{model_id}"
        except ValueError:
            return spec

    def _resolve_candidates(
        self,
        *,
        role: Optional[str],
        model: Optional[str] = None,
        require_vision: bool = False,
        allow_fallback: bool = True,
    ) -> tuple[str, List[str]]:
        if role:
            primary, fallback_chain = resolve_role(role)
        else:
            primary = model or self.current_model
            global_primary, global_chain = resolve_role("fallback")
            fallback_chain = [global_primary] + global_chain

        if model and model != primary:
            fallback_chain = [primary] + fallback_chain
            primary = model

        if not allow_fallback:
            fallback_chain = []

        candidates: List[str] = []
        for spec in [primary] + fallback_chain:
            try:
                provider, model_id = parse_spec(spec)
                spec = f"{provider}:{model_id}"
            except ValueError:
                continue
            if not provider_configured(provider):
                continue
            if role and not model_allowed_for_role(role, spec):
                continue
            if require_vision and not supports_vision(spec):
                continue
            candidates.append(spec)
        return primary, candidates

    # -------- 主入口：chat --------
    async def run_turn(self, messages: List[Message], **kwargs) -> str:
        """Execute a conversational turn: planner owns tools; chat owns wording."""
        if kwargs.get("role") not in (None, "chat"):
            raise ValueError("run_turn only accepts conversational turns")
        return await self.chat(messages, **kwargs, _orchestrate=True)

    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        role: Optional[str] = None,
        caller: str = "?",
        session_id: Optional[str] = None,
        allow_fallback: bool = True,
        required_tool: Optional[str] = None,
        media_followup: bool = True,
        _orchestrate: bool = False,
        progress_callback=None,
        reply_policy=None,
    ) -> str:
        # 解析最终 session_id：显式 > contextvar > _unknown_
        sid = session_id or current_session_id.get() or "_unknown_"
        token = current_session_id.set(sid)
        run_token = None
        runner = None
        try:
            effective_role = role or "chat"
            budget = {
                "planner": config.planner_total_timeout,
                "summary": config.summary_total_timeout,
                "chat": config.chat_turn_timeout,
            }.get(effective_role, config.background_llm_total_timeout)
            if effective_role == "chat" and required_tool in {"generate_image", "send_voice"}:
                budget = max(float(budget), float(config.chat_media_turn_timeout))
            budget = max(0.1, float(budget))
            if _orchestrate:
                from .agent_runner import AgentRunner, current_run
                runner = AgentRunner(sid,budget)
                runner.progress_callback = progress_callback
                runner.reply_policy = reply_policy
                run_token = current_run.set(runner)
                await runner.start(messages)
            # This deadline includes model planning/tool rounds, media generation,
            # the QQ send and output description. HTTP timeouts remain per request.
            async def execute_turn():
                from .conversation import activity_scope
                from .network_retry import retry_scope
                with activity_scope(caller or effective_role, kind="turn", session_id=sid,
                                    run_id=runner.id if runner else None), retry_scope(budget):
                    return await (self._run_stages if _orchestrate else self._chat_impl)(
                        messages, model=model, role=role, caller=caller, session_id=sid,
                        allow_fallback=allow_fallback, required_tool=required_tool,
                        media_followup=media_followup,
                    )
            task = asyncio.create_task(execute_turn())
            try:
                async with asyncio.timeout(max(0.01, runner.remaining()) if runner else budget) as turn_timeout:
                    if runner:
                        runner.timeout_scope = turn_timeout
                    response = await task
                if runner:
                    if not isinstance(response,ChatResponse):
                        response=ChatResponse(str(response or ""))
                    response.agent_run=runner
                    response.reply_policy=runner.reply_policy
                    await runner.finish("complete",response)
                return response
            except asyncio.TimeoutError as error:
                if not turn_timeout.expired():
                    raise  # Preserve a nested timeout; it did not exhaust this turn.
                if runner and runner.render_fallback is not None:
                    await runner.validate_scope()
                    response = runner.render_fallback
                    response.agent_run = runner
                    response.reply_policy = runner.reply_policy
                    general_logger.warning(f"[render_fallback] session={sid} caller={caller} turn deadline; deliver prepared answer")
                    await runner.finish("complete", response)
                    return response
                timeout_error = ModelTurnTimeout(
                    role=effective_role, required_tool=required_tool,
                    timeout=(runner.deadline - runner.started_at if runner else budget),
                )
                general_logger.error(f"[turn_timeout] session={sid} caller={caller} {timeout_error}")
                raise timeout_error from error
        finally:
            try:
                if run_token is not None:
                    try:
                        if runner.status=="running":
                            await runner.finish("interrupted")
                    finally:
                        current_run.reset(run_token)
            finally:
                current_session_id.reset(token)

    async def _run_stages(self, messages, *, model, role, caller, session_id,
                          allow_fallback, required_tool, media_followup):
        from .agent_runner import current_run
        from .conversation_style import PLANNING_STYLE, RENDERING_STYLE, build_rendering_handoff

        runner = current_run.get()
        planning_messages = [*messages, Message(role="system", content=(
            PLANNING_STYLE + "\n较长任务可用 report_progress 告知已有发现或实际阻碍，不为进展拖延任务。"
        ))]
        plan = await self._chat_impl(
            planning_messages, model=None, role="planner", caller=caller + ":plan",
            session_id=session_id, allow_fallback=allow_fallback,
            required_tool=required_tool, media_followup=media_followup, tools_enabled=True,
        )
        if not str(plan).strip():
            return plan  # A delivered media-only turn needs no text renderer.
        runner.planning_result = str(plan)[:4000]
        await runner.checkpoint()
        await runner.validate_scope()
        if runner.reply_policy and runner.reply_policy.scene == "proactive" and not runner.should_reply:
            general_logger.info("[主动发言] 执行规划决定放弃本轮参与，未发送消息")
            return ChatResponse("", usage=getattr(plan, "usage", None))
        from dataclasses import replace
        # The execution planner has the latest context and can shorten or expand
        # ordinary replies. An earlier participation decision is only provisional.
        policy = runner.reply_policy
        if policy and policy.scene in {"chat", "multiuser", "close", "proactive", "poke"} and not required_tool and runner.reply_mode:
            runner.reply_policy = replace(policy, detailed=runner.reply_mode == "detailed" and policy.scene in {"chat", "multiuser"}, tiny=runner.reply_mode == "tiny")
        draft = str(plan) if runner.identity_verdict else runner.answer_draft
        if runner.response_mode == "factual" and draft and not is_tool_call_text(draft):
            runner.render_fallback = _attach_response_context(
                ChatResponse(draft, usage=getattr(plan, "usage", None)),
                model_spec=getattr(plan, "model_spec", None), primary_spec=getattr(plan, "primary_spec", None),
                fallback_attempt=getattr(plan, "fallback_attempt", 0),
                role="chat", caller=caller + ":render_fallback", session_id=session_id,
            )
        rendering_messages = [m for m in messages if m.role == "system"]
        if runner.reply_policy:
            old_hint, final_hint = policy.system_hint(), runner.reply_policy.system_hint()
            if any(old_hint in m.content for m in rendering_messages):
                rendering_messages = [replace(m, content=m.content.replace(old_hint, final_hint)) for m in rendering_messages]
            else:
                rendering_messages.append(Message(role="system", content=final_hint))
        # Factual context has already been interpreted/checked by the planner.
        # Replaying guesses or earlier wrong answers invites the renderer to
        # reinterpret them as evidence. Banter still needs conversational cues.
        if runner.response_mode == "social":
            recent = [m for m in messages if m.role in {"user", "assistant"} and not m.is_summary and not m.is_reference and not m.is_thinking][-6:]
            rendering_messages.append(Message(role="user", content="最近对话，仅供接话和理解指代：\n" +
                "\n".join(f"{m.role}: {m.content[:400]}" for m in recent)))
        rendering_messages.extend([
            Message(role="system", content=RENDERING_STYLE),
            Message(role="user", content="用户本轮问题：" + runner.user_question),
            Message(role="system", content=build_rendering_handoff(
                str(plan), response_mode=runner.response_mode,
                requirements=runner.reply_requirements, answer_draft=draft,
                identity=bool(runner.identity_verdict),
            )),
        ])
        try:
            result = await self._chat_impl(
                rendering_messages, model=model, role="chat", caller=caller + ":render",
                session_id=session_id, allow_fallback=allow_fallback,
                required_tool=None, media_followup=False,
            )
        except Exception:
            if runner.render_fallback is None:
                raise
            await runner.validate_scope()
            general_logger.warning(f"[render_fallback] session={session_id} caller={caller} rendering failed; deliver prepared answer")
            return runner.render_fallback
        # Identity is a verified account mapping, not an invitation to reinterpret it.
        if runner.identity_verdict and str(result) != str(plan):
            result = _attach_response_context(
                ChatResponse(str(plan), usage=result.usage), model_spec=result.model_spec,
                primary_spec=result.primary_spec, fallback_attempt=result.fallback_attempt,
                role="chat", caller=caller + ":render", session_id=session_id,
            )
        result.usage = _add_usage(getattr(plan, "usage", None) or {}, getattr(result, "usage", None) or {})
        return result

    async def _chat_impl(
        self,
        messages: List[Message],
        model: Optional[str],
        role: Optional[str],
        caller: str,
        session_id: str,
        allow_fallback: bool,
        required_tool: Optional[str],
        media_followup: bool,
        tools_enabled: bool = False,
    ) -> str:
        primary, candidates = self._resolve_candidates(
            role=role,
            model=model,
            allow_fallback=allow_fallback,
        )

        if not candidates:
            raise ValueError(
                f"role={role} 主 spec={primary} 没有任何可用候选（检查 provider API key 配置）"
            )

        per_call_extra_factory = self._extra_for_role(role)

        last_err: Optional[Exception] = None
        log = general_logger.bind(category="llm")
        for idx, spec in enumerate(candidates):
            if tools_enabled:
                from .agent_runner import current_run
                run = current_run.get()
                if run and run.planning_remaining() <= 0:
                    last_err = last_err or TimeoutError("planner budget exhausted; final reply time is reserved")
                    break
            cooldown = self._model_cooldowns.get(spec)
            if cooldown and cooldown[0] > time.monotonic():
                last_err = cooldown[1]
                log.info(f"[fallback] spec={spec} 仍在接口声明的冷却期，跳过")
                continue
            self._model_cooldowns.pop(spec, None)
            if role == "search" and web_search_backend(spec) == "openai_responses":
                from .native_search import native_search
                try:
                    native = await native_search(
                        spec, "\n".join(message.content for message in messages),
                        session_id=session_id, fallback_attempt=idx, caller=caller,
                    )
                    # The legacy string tool also exposes real, clickable API sources.
                    citations = "\n".join(
                        f"[{source['index']}] {source['title']}: {source['url']}"
                        for source in native["sources"]
                    )
                    return _attach_response_context(
                        ChatResponse(native["content"] + "\n\n来源：\n" + citations, usage=native.get("usage")),
                        model_spec=spec, primary_spec=primary, fallback_attempt=idx,
                        role=role, caller=caller, session_id=session_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_err = error
                    log.warning(f"[fallback] 原生搜索 spec={spec} 失败: {error}，尝试下一个")
                    continue
            try:
                adapter = self.get_or_build_adapter(spec)
            except Exception as e:
                last_err = e
                log.warning(f"[fallback] 构造 adapter 失败 spec={spec}: {e}")
                continue

            extra = per_call_extra_factory(spec) if per_call_extra_factory else None

            if role == "planner" and not tools_enabled:
                extra = {**(extra or {}), REQUEST_TIMEOUT_EXTRA_KEY: min(20.0, config.planner_total_timeout)}

            call = start_call(
                role=role or "ad-hoc",
                model=spec,
                messages=messages,
                caller=caller,
                session_id=session_id,
                extra={"attempt": idx, "primary": primary},
            )
            try:
                resp = await adapter.chat(
                    messages,
                    extra_body=extra,
                    tools_enabled=tools_enabled,
                    required_tool=required_tool,
                    media_followup=media_followup,
                )
                finish_call(
                    call,
                    response=resp,
                    fallback_used=(spec if idx > 0 else None),
                    fallback_attempt=idx,
                    usage=getattr(resp, "usage", None),
                )
                resp = _attach_response_context(
                    resp,
                    model_spec=spec,
                    primary_spec=primary,
                    fallback_attempt=idx,
                    role=role,
                    caller=caller,
                    session_id=session_id,
                )
                if idx > 0:
                    log.warning(
                        f"[fallback] role={role or '-'} 主 {primary} 失败, "
                        f"成功落在 {spec} (尝试#{idx})"
                    )
                return resp
            except asyncio.CancelledError as error:
                finish_call(call, error=error, usage=getattr(error,"usage",None), fallback_attempt=idx)
                raise
            except Exception as e:
                last_err = e
                reset_delay = _quota_reset_delay(e)
                if reset_delay:
                    self._model_cooldowns[spec] = (time.monotonic() + reset_delay, e)
                finish_call(call, error=e, usage=getattr(e,"usage",None), fallback_attempt=idx)
                log.warning(
                    f"[fallback] role={role or '-'} spec={spec} 失败: {error_summary(e)}, "
                    f"尝试下一个 (剩 {len(candidates) - idx - 1})"
                )

        log.error(
            f"[fallback] role={role or '-'} 全部 {len(candidates)} 个候选都失败, "
            f"最后错误: {error_summary(last_err)}"
        )
        raise last_err if last_err else RuntimeError("所有候选模型均不可用")

    def _extra_for_role(self, role: Optional[str]):
        """按 role 决定每次调用要附加的 extra_body。"""
        if role == "important_info":
            def factory(spec: str):
                return {
                    REQUEST_TIMEOUT_EXTRA_KEY: config.important_info_request_timeout,
                    REQUEST_MAX_RETRIES_EXTRA_KEY: 0,
                }
            return factory
        if role == "search":
            def factory(spec: str):
                if web_search_backend(spec) == "dashscope":
                    return {
                        "enable_search": True,
                        "search_options": {
                            "forced_search": True,
                            "search_strategy": "max",
                        },
                    }
                return None
            return factory
        if role == "vision":
            def factory(spec: str):
                return {
                    REQUEST_TIMEOUT_EXTRA_KEY: config.vision_request_timeout,
                }
            return factory
        return None

    # -------- 视觉 --------
    async def describe_image(self, image_base64: str, session_id: Optional[str] = None) -> str:
        analysis = await self._get_image_analysis(
            image_base64,
            session_id=session_id,
            caller="vision:describe_image",
        )
        return str(analysis.get("description") or "")

    async def analyze_image_for_meme(self, image_base64: str, session_id: Optional[str] = None) -> dict:
        """一次 vision 调用同时拿到：自然语言描述 + 表情包元数据。

        返回 {description: str, is_meme: bool, topic: str|None}
        - description: 给现有图片识别用途用（写入聊天上下文）
        - is_meme + topic: 给表情包采集用
        失败时降级为 {description: <原文或空>, is_meme: False, topic: None}
        """
        from .roles import role_available
        if not role_available("vision"):
            return {"description": "[图片：未配置识图模型]", "is_meme": False, "topic": None}

        return await self._get_image_analysis(
            image_base64,
            session_id=session_id,
            caller="vision:analyze_image_for_meme",
        )

    def _vision_cache_limit(self) -> int:
        try:
            return max(0, int(config.vision_cache_size))
        except Exception:
            return 256

    def _vision_cache_key(self, image_base64: str) -> str:
        return hashlib.sha256((image_base64 or "").encode("utf-8")).hexdigest()

    def _get_cached_vision_analysis(self, key: str) -> Optional[dict]:
        if self._vision_cache_limit() <= 0:
            return None
        cached = self._vision_analysis_cache.get(key)
        if cached is not None:
            self._vision_analysis_cache.move_to_end(key)
            return dict(cached)
        return None

    def _remember_vision_analysis(self, key: str, value: dict) -> None:
        limit = self._vision_cache_limit()
        if limit <= 0:
            return
        self._vision_analysis_cache[key] = dict(value)
        self._vision_analysis_cache.move_to_end(key)
        while len(self._vision_analysis_cache) > limit:
            self._vision_analysis_cache.popitem(last=False)

    async def _get_image_analysis(
        self,
        image_base64: str,
        *,
        session_id: Optional[str],
        caller: str,
    ) -> dict:
        if self._vision_cache_limit() <= 0:
            return await self._analyze_image_for_meme_uncached(
                image_base64,
                session_id=session_id,
                caller=caller,
            )

        key = self._vision_cache_key(image_base64)
        cached = self._get_cached_vision_analysis(key)
        if cached is not None:
            general_logger.bind(category="llm").debug(
                f"[vision_cache] 命中图片识别缓存 key={key[:8]} caller={caller}"
            )
            return cached

        pending = self._vision_pending.get(key)
        if pending is not None:
            return dict(await asyncio.shield(pending))

        async def analyze():
            try:
                result = await self._analyze_image_for_meme_uncached(
                    image_base64, session_id=session_id, caller=caller,
                )
                self._remember_vision_analysis(key, result)
                return result
            finally:
                self._vision_pending.pop(key, None)

        task = asyncio.create_task(analyze())
        self._vision_pending[key] = task
        # Shared recognition survives one caller timing out; always consume errors
        # even when all callers have already left.
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return dict(await asyncio.shield(task))

    async def close_pending(self) -> None:
        tasks = list(self._vision_pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _analyze_image_for_meme_uncached(
        self,
        image_base64: str,
        *,
        session_id: Optional[str],
        caller: str,
    ) -> dict:
        import json as _json
        import re as _re
        from .memes import ALLOWED_TOPICS, MEME_CLASSIFICATION_RULES

        prompt_text = (
            "请分析这张图片。先客观描述可见内容（场景、人物、物体、文字），不要在描述中替图片编造聊天用途，"
            '然后另起一行输出严格 JSON：{"is_meme": bool, "topic": str|null, "random_eligible": bool, "text_sentence_count": int, "meme_kind": "reaction"|"joke"|null, "reason": "简短分类依据"}。\n'
            + MEME_CLASSIFICATION_RULES + "\n"
            "正例格式：\n"
            "一只柴犬咧嘴大笑的特写，下方有'我可以'白色字幕。\n"
            "{\"is_meme\": true, \"topic\": \"funny\", \"random_eligible\": true, \"text_sentence_count\": 1, \"meme_kind\": \"reaction\", \"reason\": \"大笑反应\"}\n"
            "反例格式：\n"
            "一张商店活动公告，列出日期、地点和商品价格。\n"
            "{\"is_meme\": false, \"topic\": null, \"random_eligible\": false, \"text_sentence_count\": 4, \"meme_kind\": null, \"reason\": \"普通商品公告\"}"
        )
        msgs = [Message(role="user", content=prompt_text)]
        resp = await self._chat_with_images_impl(
            msgs,
            [image_base64],
            role="vision",
            session_id=session_id,
            caller=caller,
        )

        text = (resp or "").strip()
        # 提取最后一个 JSON 对象作为 metadata，前面其余文本作为 description
        m = _re.search(r"\{[^{}]*\"is_meme\"[^{}]*\}\s*$", text, _re.DOTALL)
        if not m:
            m = _re.search(r"\{[^{}]*\"is_meme\"[^{}]*\}", text, _re.DOTALL)
        meta = {"is_meme": False, "topic": None, "random_eligible": None}
        classification = {}
        if m:
            try:
                parsed = _json.loads(m.group(0))
                if isinstance(parsed, dict):
                    # 只接受 JSON 原生布尔值 true。字符串 "true"/"false"、
                    # 数字 1/0 等非规范输出一律按非表情包处理，避免 bool("false") == True。
                    raw_is_meme = parsed.get("is_meme")
                    sentence_count = parsed.get("text_sentence_count")
                    if type(sentence_count) is int and sentence_count >= 3:
                        parsed.update(is_meme=False, random_eligible=False, topic=None, meme_kind=None)
                        raw_is_meme = False
                    eligible = parsed.get("random_eligible")
                    t = parsed.get("topic")
                    if raw_is_meme is True and isinstance(t, str) and t in ALLOWED_TOPICS:
                        meta["is_meme"] = True
                        meta["topic"] = t
                    else:
                        meta["is_meme"] = False
                        meta["topic"] = None
                    if type(raw_is_meme) is bool and type(eligible) is bool:
                        meta["random_eligible"] = meta["is_meme"] and eligible
                    classification = {"meme_kind": parsed.get("meme_kind"),
                                      "classification_reason": str(parsed.get("reason") or "")[:400],
                                      "text_sentence_count": sentence_count if type(sentence_count) is int and sentence_count >= 0 else None}
                description = text[:m.start()].strip() or text.strip()
            except Exception:
                description = text
        else:
            description = text

        return {
            "description": description,
            **classification,
            "is_meme": meta["is_meme"],
            "topic": meta["topic"],
            "random_eligible": meta["random_eligible"],
        }

    async def _chat_with_images_impl(
        self,
        messages: List[Message],
        images: List[str],
        *,
        model: Optional[str] = None,
        role: str = "vision",
        session_id: Optional[str] = None,
        caller: str = "vision:chat_with_images",
    ) -> ChatResponse:
        loop = asyncio.get_running_loop()
        slots = getattr(self, "_vision_slots", None)
        if slots is None or slots[0] is not loop:
            slots = (loop, asyncio.Semaphore(max(1, int(config.vision_max_concurrency))))
            self._vision_slots = slots

        async def limited():
            async with slots[1]:
                return await self._chat_with_images_candidates(
                    messages, images, model=model, role=role, session_id=session_id, caller=caller,
                )

        from .network_retry import retry_scope
        budget = max(0.1, float(config.background_llm_total_timeout))
        with retry_scope(budget):
            scope = asyncio.timeout(budget)
            try:
                async with scope:
                    return await limited()
            except TimeoutError as error:
                if not scope.expired():
                    raise
                raise TimeoutError(f"识图总预算耗尽（{budget:g} 秒，包含排队、重试和备用模型）") from error

    async def _chat_with_images_candidates(
        self,
        messages: List[Message],
        images: List[str],
        *,
        model: Optional[str] = None,
        role: str = "vision",
        session_id: Optional[str] = None,
        caller: str = "vision:chat_with_images",
    ) -> ChatResponse:
        from .network_retry import retry_budget, retry_scope, remaining_retry_budget
        deadline = time.monotonic() + max(0.1, float(config.background_llm_total_timeout))
        sid = session_id or current_session_id.get() or "_unknown_"
        token = current_session_id.set(sid)
        try:
            primary, candidates = self._resolve_candidates(
                role=role,
                model=model,
                require_vision=True,
            )
            if not candidates:
                raise ValueError(
                    f"role={role} 主 spec={primary} 没有任何可用视觉候选"
                    "（检查 MODEL_VISION 配置、视觉白名单和 provider API key）"
                )

            per_call_extra_factory = self._extra_for_role(role)
            last_err: Optional[Exception] = None
            log = general_logger.bind(category="llm")
            for idx, spec in enumerate(candidates):
                try:
                    adapter = self.get_or_build_adapter(spec)
                except Exception as e:
                    last_err = e
                    log.warning(f"[fallback] 构造 vision adapter 失败 spec={spec}: {e}")
                    continue

                extra = per_call_extra_factory(spec) if per_call_extra_factory else None
                call = start_call(
                    role=role,
                    model=spec,
                    messages=messages,
                    caller=caller,
                    session_id=sid,
                    extra={"attempt": idx, "primary": primary},
                )
                remaining = min(deadline - time.monotonic(), remaining_retry_budget())
                # Keep one full request for the next configured candidate.
                reserve = min(float(config.vision_request_timeout), remaining / 2) if idx + 1 < len(candidates) else 0
                candidate_budget = max(0.001, min(retry_budget(config.vision_request_timeout), remaining - reserve))
                log.debug(f"[vision_budget] spec={spec} remaining={remaining:.2f}s "
                          f"candidate={candidate_budget:.2f}s reserved={reserve:.2f}s")
                try:
                    with retry_scope(candidate_budget):
                        resp = await asyncio.wait_for(
                            adapter.chat_with_images(
                                messages, images, extra_body=extra, tools_enabled=False,
                            ),
                            timeout=candidate_budget,
                        )
                    finish_call(
                        call,
                        response=resp,
                        fallback_used=(spec if idx > 0 else None),
                        fallback_attempt=idx,
                        usage=getattr(resp, "usage", None),
                    )
                    resp = _attach_response_context(
                        resp,
                        model_spec=spec,
                        primary_spec=primary,
                        fallback_attempt=idx,
                        role=role,
                        caller=caller,
                        session_id=sid,
                    )
                    if idx > 0:
                        log.warning(
                            f"[fallback] role={role} 主 {primary} 失败, "
                            f"成功落在 {spec} (尝试#{idx})"
                        )
                    return resp
                except Exception as e:
                    last_err = e
                    finish_call(call, error=e, fallback_attempt=idx)
                    log.warning(
                        f"[fallback] role={role} vision spec={spec} 失败: {error_summary(e)}, "
                        f"尝试下一个 (剩 {len(candidates) - idx - 1})"
                    )
                except asyncio.CancelledError as error:
                    finish_call(call, error=error, fallback_attempt=idx)
                    raise

            log.error(
                f"[fallback] role={role} 全部 {len(candidates)} 个视觉候选都失败, "
                f"最后错误: {error_summary(last_err)}"
            )
            raise last_err if last_err else RuntimeError("所有视觉候选模型均不可用")
        finally:
            current_session_id.reset(token)

    async def chat_with_images(
        self,
        messages: List[Message],
        images: List[str],
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        caller: str = "vision:chat_with_images",
    ) -> str:
        from .roles import role_available
        if not role_available("vision"):
            return "[图片：未配置识图模型]"
        target_model = model if model and supports_vision(model) else None
        return await self._chat_with_images_impl(
            messages,
            images,
            model=target_model,
            role="vision",
            session_id=session_id,
            caller=caller,
        )


# 全局模型管理器
model_manager = ModelManager()
