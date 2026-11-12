"""Provider wire protocols and request formatting, independent of model routing."""
from abc import ABC, abstractmethod
import hashlib
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from .config import config, get_tool_usage_prompt, NO_TOOL_PROMPT
from .providers import parse_spec, model_supports_tools
from .streaming import collect_completion
from .model_types import (
    Message, ChatResponse, current_session_id,
    REQUEST_TIMEOUT_EXTRA_KEY, REQUEST_MAX_RETRIES_EXTRA_KEY,
)


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
