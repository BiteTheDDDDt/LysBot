"""多模型统一适配器 + ModelManager。

所有模型通过 spec 字符串引用，格式 `provider:model_id`，例如
`nvidia:moonshotai/kimi-k2.5`。Adapter 按 spec 懒建并缓存。

业务调用 `model_manager.chat(messages, role="judge", caller="...", session_id="...")`，
路由由 `roles.resolve_role(role)` 决定主+回退链。
"""
from typing import List, Optional, Dict
from collections import OrderedDict
import asyncio
import math
import time
import hashlib

from .model_turn import execute_model_call, run_stages
from .diagnostics import error_summary
from .config import config
from .llm_logger import start_call, finish_call
from .logging_setup import general_logger
from .roles import model_allowed_for_role, resolve_role
from .providers import (
    PROVIDERS,
    parse_spec,
    provider_configured,
    configured_providers,
    supports_vision,
    web_search_backend,
    get_provider_proxy,
)


from .model_types import (
    Message as Message,
    ChatResponse as ChatResponse,
    ModelTurnTimeout as ModelTurnTimeout,
    current_session_id as current_session_id,
    REQUEST_TIMEOUT_EXTRA_KEY as REQUEST_TIMEOUT_EXTRA_KEY,
    REQUEST_MAX_RETRIES_EXTRA_KEY as REQUEST_MAX_RETRIES_EXTRA_KEY,
    _attach_response_context as _attach_response_context,
    _extract_openai_usage as _extract_openai_usage,
    _add_usage as _add_usage,
)
from .model_adapters import (
    ModelAdapter as ModelAdapter,
    OpenAICompatibleAdapter as OpenAICompatibleAdapter,
    VolcAdapter as VolcAdapter,
    ClaudeAdapter as ClaudeAdapter,
    _strip_fake_tool_calls as _strip_fake_tool_calls,
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
        return await execute_model_call(
            self,
            messages=messages,
            model=model,
            role=role,
            caller=caller,
            session_id=session_id,
            allow_fallback=allow_fallback,
            required_tool=required_tool,
            media_followup=media_followup,
            _orchestrate=_orchestrate,
            progress_callback=progress_callback,
            reply_policy=reply_policy,
        )

    async def _run_stages(self, messages, *, model, role, caller, session_id,
                          allow_fallback, required_tool, media_followup):
        return await run_stages(
            self,
            messages=messages,
            model=model,
            role=role,
            caller=caller,
            session_id=session_id,
            allow_fallback=allow_fallback,
            required_tool=required_tool,
            media_followup=media_followup,
        )

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
