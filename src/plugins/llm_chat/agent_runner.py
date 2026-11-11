"""Provider-independent conversation/tool execution. Providers perform one request."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextvars import ContextVar
from copy import deepcopy

from .config import config
from .logging_setup import general_logger
from .response_safety import INVALID_REPLY, is_tool_call_text

current_run: ContextVar[AgentRunner | None] = ContextVar("agent_run", default=None)
_current_action: ContextVar[str | None] = ContextVar("agent_action", default=None)
from .capabilities import CAPABILITIES, READ_TOOLS, MEDIA_TOOLS, MEMORY_READ_TOOLS
from .execution import ExecutionPolicy, execution_scope

# Internal prompt/log budgets; deliberately not operator-facing settings.
CONTEXT_MAX_CHARS = 40000
TOOL_RESULT_LOG_MAX_CHARS = 2000

_runtime_path = None
active_runs: dict[str, AgentRunner] = {}


def configure_store(path):
    global _runtime_path
    _runtime_path = path


def recover_store(path):
    from .memory_index import connect

    with connect(path) as db:
        db.execute("UPDATE agent_runs SET status='interrupted' WHERE status='running'")
        db.execute("DELETE FROM agent_runs WHERE updated_at<?", (time.time() - 7 * 86400,))


def clear_store(path):
    if _runtime_path == path:
        configure_store(None)


def steer(session_id, requester_id, text):
    run = active_runs.get(session_id)
    if not run or run.requester_id != requester_id or not text or len(text) > 3000:
        return False
    run.steering = (run.steering + [text])[-5:]
    from .web_search import register_user_links
    register_user_links(run, text)
    run.cache.clear()
    return True


class DeliveryUncertain(RuntimeError):
    """A side effect may have happened; automatic delivery replay is forbidden."""


async def record_tool_delivery(status, receipt):
    run, key = current_run.get(), _current_action.get()
    if run is not None and key in run.actions:
        run.actions[key].update(status="sent" if status == "sent" else "unknown", receipt=receipt)
        await run.checkpoint()


class AgentRunner:
    def __init__(self, session_id="_unknown_", budget=None):
        self.id = uuid.uuid4().hex
        self.session_id = session_id
        self.started_at = time.monotonic()
        self.deadline = self.started_at + float(budget or config.chat_turn_timeout)
        self.timeout_scope = None
        self.planning_active = False
        self.authorized_media_tool = None
        self.transcript: list[dict] = []
        self.tool_round = 0
        self.tool_calls = 0
        self.rejected_tool_calls = 0
        self.no_new_evidence_calls = 0
        self.cache: dict[str, str] = {}
        self.evidence: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}
        self.actions: dict[str, dict] = {}
        self.steering: list[str] = []
        self.status = "running"
        self._read_slots = asyncio.Semaphore(max(1, config.agent_read_concurrency))
        self.memory_path = _runtime_path
        self.memory_generation = None
        self.requester_id = None
        self.final_answer = None
        self.outcomes = {}
        self.step_metrics = []
        self.user_question = ""
        self.identity_question = False
        self.identity_verdict = None
        self.response_mode = "factual"
        self.planning_result = ""
        self.should_reply = True
        self.reply_requirements = []
        self.reply_mode = ""
        self.answer_draft = ""
        self.reply_policy = None
        self.render_fallback = None
        self.context_run_ids = []
        self.context_questions: list[str] = []
        self.injected_memories = []
        self.memory_correction_epoch = ""
        self.api_errors = []
        self.progress_callback = None
        self.progress_count = 0
        self.progress_last_at = time.monotonic()
        self.progress_messages = []

    async def start(self, messages):
        from .tools import _poke_context

        context = _poke_context.get() or {}
        self.injected_memories = [ref for m in messages for ref in (getattr(m, "memory_refs", None) or [])
                                  if ref.get("id", "") in m.content]
        self.request_source_ids = []
        self.requester_id = context.get("user_id") or next(
            (m.sender_id for m in reversed(messages) if m.role == "user" and m.sender_id), None
        )
        if self.memory_path and self.session_id != "_unknown_":
            from .memory_index import connect, generation

            def load():
                with connect(self.memory_path) as db:
                    from .memory_changes import correction_epoch
                    self.memory_correction_epoch = correction_epoch(db, self.session_id)
                    for text in context.get("history_request_texts", [])[:30]:
                        if not isinstance(text, str) or not text:
                            continue
                        row = db.execute("SELECT id FROM messages_detail WHERE session_id=? AND role='user' AND content=? ORDER BY id DESC LIMIT 1",
                                         (self.session_id, text)).fetchone()
                        if row:
                            self.request_source_ids.append(row[0])
                    return generation(db, self.session_id)

            try:
                self.memory_generation = await asyncio.to_thread(load)
            except Exception as error:  # noqa: BLE001 -- persistence failure must not lose the current user turn.
                general_logger.warning(f"[agent_start] memory unavailable: {type(error).__name__}")
        active_runs[self.session_id] = self
        latest = messages[-1].content if messages else ""
        self.user_question = latest or ""
        from .working_context import restore_context
        from sqlite3 import Error as SQLiteError

        try:
            await restore_context(self)
        except (SQLiteError, OSError, ValueError) as error:
            general_logger.warning(f"[working_context] restore unavailable: {type(error).__name__}")
        from .web_search import register_user_links

        supplied = [m for m in messages if m.role == "user" and not (
            m.is_reference or m.is_summary or m.is_thinking
        )]
        for message in supplied[-12:]:
            register_user_links(self, message.content)
        await self.checkpoint()

    async def validate_scope(self):
        if not self.memory_path or self.memory_generation is None:
            return
        from .conversation import TurnSuperseded
        from .memory_index import SOURCE_TABLES, connect, digest, generation

        def valid():
            with connect(self.memory_path) as db:
                from .memory_changes import correction_epoch
                if correction_epoch(db, self.session_id) != self.memory_correction_epoch:
                    return False
                if generation(db, self.session_id) != self.memory_generation:
                    return False
                from .memory_changes import active_sql, version
                for item in self.injected_memories:
                    if item.get("invalidated_by"):
                        continue
                    _, gen, kind, mid = item["id"].split(":")
                    if kind not in {"summaries", "important_info"} or int(gen) != self.memory_generation:
                        return False
                    row = db.execute(f"SELECT * FROM {kind} WHERE id=? AND session_id=? AND {active_sql(kind)}",
                                     (int(mid), self.session_id)).fetchone()
                    if not row or version(row) != item["version"]:
                        return False
                refs = {
                    ref for item in self.evidence.values() for ref in item.get("source_ids", []) if ref.startswith("h:")
                }
                for ref in refs:
                    parts = ref.split(":")
                    if len(parts) != 4 or parts[2] not in SOURCE_TABLES or not parts[3].isdigit():
                        return False
                    row = db.execute(
                        f"SELECT content FROM {SOURCE_TABLES[parts[2]]} WHERE id=? AND session_id=?",
                        (int(parts[3]), self.session_id),
                    ).fetchone()
                    item = self.evidence.get(ref, {})
                    if not row or (item.get("content_hash") and digest(row[0]) != item["content_hash"]):
                        return False
                return True

        if not await asyncio.to_thread(valid):
            raise TurnSuperseded()

    async def checkpoint(self, *, strict=False):
        if not self.memory_path or self.memory_generation is None:
            if strict and self.memory_path:
                raise DeliveryUncertain("无法确认持久化会话状态，本次动作未执行")
            return
        from .memory_index import connect, encode, generation

        state = {
            "requester_id": self.requester_id,
            "transcript": self.transcript,
            "evidence": self.evidence,
            "sources": {key: {k: v for k, v in source.items() if k != "body"} for key, source in self.sources.items()},
            "actions": self.actions,
            "tool_calls": self.tool_calls,
            "rejected_tool_calls": self.rejected_tool_calls,
            "tool_round": self.tool_round,
            "steering": self.steering,
            "remaining": self.remaining(),
            "step_metrics": self.step_metrics,
            "user_question": self.user_question,
            "identity_question": self.identity_question,
            "identity_verdict": self.identity_verdict,
            "response_mode": self.response_mode,
            "planning_result": self.planning_result,
            "should_reply": self.should_reply,
            "reply_requirements": self.reply_requirements,
            "reply_mode": self.reply_mode,
            "answer_draft": self.answer_draft,
            "context_run_ids": self.context_run_ids,
            "injected_memories": self.injected_memories,
            "memory_correction_epoch": self.memory_correction_epoch,
            "api_errors": self.api_errors,
            "progress_messages": self.progress_messages,
            "progress_count": self.progress_count,
            "final_answer": self.final_answer,
        }
        payload = encode(state)

        def save():
            with connect(self.memory_path, operation="agent.checkpoint") as db:
                db.execute("BEGIN IMMEDIATE")
                if generation(db, self.session_id) != self.memory_generation:
                    return False
                db.execute(
                    """INSERT INTO agent_runs VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,state_json=excluded.state_json,updated_at=excluded.updated_at""",
                    (self.id, self.session_id, self.memory_generation, self.status, payload, time.time()),
                )
                return True

        try:
            import sqlite3
            for attempt in range(3):
                try:
                    saved = await asyncio.to_thread(save)
                    break
                except sqlite3.OperationalError as error:
                    if "database is locked" not in str(error).lower() or attempt == 2:
                        raise
                    await asyncio.sleep(0.2 * (attempt + 1))
            if not saved:
                from .conversation import TurnSuperseded

                raise TurnSuperseded()
        except Exception as error:
            from .diagnostics import error_summary
            general_logger.warning(f"[agent_checkpoint] run={self.id} error={error_summary(error)}")
            if strict:
                raise DeliveryUncertain("无法持久化发送状态，本次动作未执行") from error

    async def finish(self, status, answer=None):
        self.status = status
        if answer is not None:
            self.final_answer = {
                "answer": str(answer),
                "evidence_ids": list(dict.fromkeys(
                    ref for key, value in self.evidence.items()
                    if value.get("read_state") in {"original", "native_search"}
                    for ref in (value.get("source_ids", [key]) if value.get("read_state") == "native_search" else [key])
                    if ref in str(answer) or (
                        (self.sources.get(ref, {}).get("url") or value.get("url"))
                        and (self.sources.get(ref, {}).get("url") or value.get("url")) in str(answer)
                    )
                )),
                "read_evidence_ids": [
                    key for key, value in self.evidence.items()
                    if value.get("read_state") in {"original", "native_search"}
                ],
                "tool_limitations": [
                    value for value in self.outcomes.values() if value.get("status") in {"partial", "unavailable"}
                ],
            }
        if active_runs.get(self.session_id) is self:
            active_runs.pop(self.session_id, None)
        await self.checkpoint()

    async def deliver(self, text, send):
        await self.validate_scope()
        key = "reply:" + hashlib.sha256(text.encode()).hexdigest()
        prior = self.actions.get(key)
        if prior:
            if prior["status"] == "sent":
                return prior.get("receipt")
            raise DeliveryUncertain("已有发送尝试，不能自动重复发送")
        self.actions[key] = {"status": "started", "text": text}
        await self.checkpoint(strict=True)
        # Persistence yields to mute/reset commands. Check the live turn again
        # immediately before the transport call, after all awaited preparation.
        from .conversation import ensure_send_allowed

        ensure_send_allowed(int(self.session_id[6:]) if self.session_id.startswith("group_") else None)
        try:
            receipt = await send()
        except BaseException as error:
            self.actions[key]["status"] = "unknown"
            await self.checkpoint()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise DeliveryUncertain("发送结果未知，已保留记录且不自动重发") from error
        self.actions[key].update(
            status="sent", receipt=receipt if isinstance(receipt, (dict, str, int, type(None))) else str(receipt)
        )
        await self.checkpoint()
        return receipt

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def render_reserve(self) -> float:
        return min(20.0, (self.deadline - self.started_at) / 8)

    def research_reserve(self) -> float:
        return min(50.0, (self.deadline - self.started_at) / 3)

    def planning_remaining(self) -> float:
        return max(0.0, self.remaining() - self.render_reserve())

    async def step(self, backend, client, kwargs):
        from .conversation import ensure_send_allowed

        group_id = int(self.session_id[6:]) if self.session_id.startswith("group_") else None
        ensure_send_allowed(group_id)
        await self.validate_scope()
        if self.steering:
            correction = {"role": "user", "content": "用户对本次问题的补充：\n" + "\n".join(self.steering)}
            kwargs["messages"].append(correction)
            self.transcript.append(correction)
            self.steering.clear()
        closing_research = self.planning_active and self.remaining() <= self.research_reserve()
        if kwargs.get("tools"):
            # A link may arrive after the initial request was prepared.
            from .tools import get_chat_tools
            reader = next((t for t in get_chat_tools() if t["function"]["name"] == "read_web"), None)
            kwargs["tools"] = [t for t in kwargs["tools"] if t["function"]["name"] != "read_web"]
            if reader:
                kwargs["tools"].append(reader)
        if closing_research and kwargs.get("tools"):
            kwargs["tools"] = [t for t in kwargs["tools"]
                               if t["function"]["name"] not in READ_TOOLS | {"report_progress"}]
            if not kwargs["tools"]:
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
        if self.remaining() < 15 or self.tool_calls >= config.agent_max_tool_calls:
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
        search_active = any(m.get("role") == "tool" and m.get("name") in READ_TOOLS for m in self.transcript)
        model_remaining = self.planning_remaining() if self.planning_active else self.remaining()
        if model_remaining <= 0:
            raise asyncio.TimeoutError("planner budget exhausted; reserved time belongs to the final reply")
        timeout = min(config.llm_request_timeout, float(kwargs.get("timeout", config.llm_request_timeout)), model_remaining)
        if search_active:
            timeout = min(timeout, max(0.1, config.agent_search_step_timeout))
        kwargs = {**kwargs, "timeout": max(0.1, timeout)}
        messages = list(kwargs["messages"])
        if closing_research:
            messages.append({"role": "user", "content": (
                "本轮检索预算已结束，不再开启检索或发进展。现在用已获得的结果完成规划并交回答复；"
                "保留来源、必要条件和未确认项，有足够事实时提供可直接发送的 answer_draft。"
            )})
        if self.api_errors:
            messages.append({"role": "user", "content": "本轮 API 故障记录，仅作执行状态，不是事实证据。"
                             "区分请求失败与没有搜索结果；替代路径成功后不必向用户罗列故障。\n"
                             + json.dumps(self.api_errors[-4:], ensure_ascii=False)})
        if search_active:
            messages.append({
                "role": "user",
                "content": "检索核验提示：提取记忆、摘要、机器人回复及其转述都只是线索。"
                "具体人物对应须由用户原文的称呼/引用/提及关系支持；没有直接依据时说明尚未确认，不能把猜测说成事实。"
                f"本轮已执行 {self.tool_calls} 次工具，剩余约 {int(self.remaining())} 秒；下一跳应解决明确缺口，可批量查独立候选。",
            })
        if self.no_new_evidence_calls >= 3:
            messages.append(
                {
                    "role": "user",
                    "content": f"检索预算提示：连续 {self.no_new_evidence_calls} 次读取没有新增证据，已调用 {self.tool_calls}/{config.agent_max_tool_calls} 次。"
                    "请判断是否已有明确的新线索值得继续；否则用现有证据收尾，指出尚不能确认的部分。",
                }
            )
        # Remove complete old tool groups, never partial JSON or unpaired calls.
        while len(json.dumps(messages, ensure_ascii=False)) > CONTEXT_MAX_CHARS:
            starts = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")]
            if len(starts) < 2:
                break
            start = starts[0]
            end = start + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            del messages[start:end]
        if len(messages) < len(kwargs["messages"]):
            notes = [
                {"id": key, "excerpt": v.get("excerpt", "")[:350], "source_ids": v.get("source_ids", [])}
                for key, v in list(self.evidence.items())[-16:]
            ]
            messages.insert(
                1,
                {
                    "role": "user",
                    "content": "已读来源摘记，仅作证据线索，可按 ID 回读：" + json.dumps(notes, ensure_ascii=False),
                },
            )
        kwargs["messages"] = messages
        await self.checkpoint()
        # Snapshot protects observations/tests from subsequent list mutation.
        started = time.monotonic()
        status = "ok"
        try:
            from .network_retry import StepTimeout, request_with_retry
            remaining = self.planning_remaining if self.planning_active else self.remaining

            async def call_once():
                budget = max(0.01, min(timeout, remaining()))
                try:
                    return await asyncio.wait_for(
                        backend._create_completion(client, deepcopy(kwargs)), timeout=budget
                    )
                except StepTimeout:
                    raise
                except asyncio.TimeoutError as error:
                    # A locally enforced step deadline: the request was already
                    # submitted, so replaying it only burns the turn budget.
                    raise StepTimeout(f"step budget exhausted after {budget:g}s") from error

            return await request_with_retry(call_once, operation=backend.model_name, remaining=remaining)
        except BaseException as error:
            status = type(error).__name__
            if isinstance(error, Exception):
                from .diagnostics import api_failure
                self.api_errors = (self.api_errors + [api_failure(error, operation=backend.model_name)])[-8:]
            raise
        finally:
            metric = {"model": backend.model_name, "round": self.tool_round,
                      "elapsed_ms": round((time.monotonic() - started) * 1000), "status": status}
            self.step_metrics = (self.step_metrics + [metric])[-32:]
            general_logger.info(f"[agent_step] run={self.id} " + json.dumps(metric, ensure_ascii=False))

    async def execute(self, name, raw_args, execute_tool):
        from .tools import MediaValidationFailure

        await self.validate_scope()
        if name in MEDIA_TOOLS and name != self.authorized_media_tool:
            self.rejected_tool_calls += 1
            return json.dumps({"status": "unavailable", "error": {
                "code": "media_not_authorized", "retryable": False,
                "message": "本轮回复计划未请求该媒体操作，请继续完成原问题。",
            }}, ensure_ascii=False)
        if self.tool_calls >= config.agent_max_tool_calls:
            self.rejected_tool_calls += 1
            return json.dumps({"status": "unavailable", "error": {"code": "tool_budget_exhausted", "retryable": False}})
        if name in READ_TOOLS and self.planning_active and self.remaining() <= self.research_reserve():
            self.rejected_tool_calls += 1
            return json.dumps({"status": "unavailable", "error": {
                "code": "research_budget_exhausted", "retryable": False,
                "message": "请用已有结果完成答复，保留尚未确认的部分，不再开启检索。",
            }}, ensure_ascii=False)
        self.tool_calls += 1
        try:
            if len(raw_args) > 20000:
                raise ValueError("arguments_too_large")
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                raise TypeError("arguments_must_be_object")
        except (ValueError, TypeError) as error:
            result = json.dumps(
                {"status": "invalid_arguments", "error": {"code": str(error), "retryable": False}}, ensure_ascii=False
            )
            return MediaValidationFailure(result) if name in MEDIA_TOOLS else result
        key = hashlib.sha256(json.dumps([name, args], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if name == "report_progress":
            from .progress import report_progress
            from .conversation import activity_scope
            if set(args) - {"summary", "source_ids"}:
                return json.dumps({"status": "invalid_arguments"})
            with activity_scope(name, kind="capability", run_id=self.id, session_id=self.session_id):
                return await report_progress(self, **args)
        if name in MEMORY_READ_TOOLS and self.memory_path:
            from .memory_index import connect

            def version():
                with connect(self.memory_path) as db:
                    return [
                        tuple(row)
                        for row in db.execute(
                            "SELECT key,value FROM kv_meta WHERE key IN (?, 'memory_active_embedding','memory_active_chunk') ORDER BY key",
                            ("memory_revision:" + self.session_id,),
                        )
                    ]

            key += hashlib.sha256(json.dumps(await asyncio.to_thread(version)).encode()).hexdigest()
        if name in READ_TOOLS and key in self.cache:
            self.no_new_evidence_calls += 1
            return self.cache[key]
        if name in MEDIA_TOOLS and self.remaining() > 0:
            # Only planned media may extend the turn. Repeated calls do not
            # reset its start or grant more time to ordinary research.
            self.deadline = max(self.deadline, self.started_at + float(config.chat_media_turn_timeout))
            if self.timeout_scope is not None:
                self.timeout_scope.reschedule(asyncio.get_running_loop().time() + self.remaining())
        if name not in READ_TOOLS:
            prior = self.actions.get(key)
            if prior:
                return prior.get("result") or "已有执行尝试，状态未知；不得自动重复执行。"
            self.actions[key] = {"name": name, "status": "started"}
            await self.checkpoint(strict=True)
        try:
            policy = CAPABILITIES[name].execution if name in CAPABILITIES else ExecutionPolicy(name)
            reserve = self.research_reserve() if self.planning_active and name in READ_TOOLS else (1 if name in READ_TOOLS else 0)
            limit = max(0.01, min(policy.timeout() if policy.timeout else self.remaining(),
                                  self.remaining() - reserve))
            async with execution_scope(policy, session_id=self.session_id, run_id=self.id,
                                       timeout=limit, slots=self._read_slots if name in READ_TOOLS else None):
                if name in READ_TOOLS:
                    result = await execute_tool(name, args)
                else:
                    token = _current_action.set(key)
                    try:
                        result = await execute_tool(name, args)
                    finally:
                        _current_action.reset(token)
        except asyncio.TimeoutError:
            if name not in READ_TOOLS:
                if self.actions[key]["status"] != "sent":
                    self.actions[key]["status"] = "unknown"
                await self.checkpoint()
            return json.dumps({"status": "unavailable", "error": {"code": "tool_timeout", "retryable": True}})
        except BaseException:
            if name not in READ_TOOLS:
                if self.actions[key]["status"] != "sent":
                    self.actions[key]["status"] = "unknown"
                await self.checkpoint()
            raise
        if name in READ_TOOLS:
            try:
                data = json.loads(str(result))
            except (TypeError, ValueError):
                data = {}
            self.outcomes[key] = {
                "tool": name,
                "status": data.get("status"),
                "warnings": data.get("warnings", []),
                "error": data.get("error"),
            }
            if isinstance(data.get("error"), dict):
                # Preserve the sanitized records, never arbitrary tool payloads.
                detail = data["error"]
                if detail.get("observed_at") and detail.get("operation"):
                    self.api_errors = (self.api_errors + [detail])[-8:]
            if isinstance(data.get("items"), list):
                # Compatibility aliases stay at the public tool boundary; the
                # model only needs one copy of each excerpt.
                data.pop("hits", None)
                for item in data["items"]:
                    if isinstance(item, dict) and item.get("content") == item.get("excerpt"):
                        item.pop("content", None)
                result = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            if data.get("status") not in {"unavailable", "invalid_arguments"}:
                self.cache[key] = str(result)
            added = False
            for item in data.get("items", []):
                if isinstance(item, dict) and item.get("id"):
                    prior = self.evidence.get(item["id"])
                    added |= not prior or (
                        item.get("read_state") == "original" and prior.get("read_state") != "original"
                    )
                    if not (item.get("read_state") == "search_snippet" and prior
                            and prior.get("read_state") in {"original", "native_search"}):
                        self.evidence[item["id"]] = item
            summary = data.get("search_summary")
            if name == "search_web" and isinstance(summary, dict) and isinstance(summary.get("content"), str):
                refs = [ref for ref in summary.get("related_source_ids", []) if ref in self.sources]
                if refs and summary["content"].strip():
                    # Anchor a composite answer once; source_ids retains all of
                    # its references, not an attribution to the first webpage.
                    self.evidence[refs[0]] = {
                        **self.evidence.get(refs[0], {}), "id": refs[0], "kind": "web",
                        "excerpt": summary["content"], "read_state": "native_search",
                        "provenance": "native_search_answer", "source_ids": refs,
                        "query": summary.get("query", ""), "searched_at": summary.get("searched_at"),
                        "truncated": bool(summary.get("truncated")),
                    }
                    added = True
            self.no_new_evidence_calls = 0 if added or data.get("aggregation") or data.get("rows") else self.no_new_evidence_calls + 1
        else:
            if isinstance(result, MediaValidationFailure):
                self.actions.pop(key, None)
            else:
                value = str(result)
                try:
                    payload = json.loads(value)
                except ValueError:
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                delivery = getattr(result, "delivery_status", None) or payload.get("delivery_status")
                recorded = self.actions[key]["status"]
                if delivery == "unconfirmed" or recorded == "unknown":
                    status = "unknown"
                elif delivery in {"confirmed", "sent"} or recorded == "sent":
                    status = "sent"
                elif name == "revise_memory" and payload.get("status") in {"preview", "applied", "undone"}:
                    status = "completed"
                else:
                    status = "failed"
                self.actions[key].update(status=status, result=value)
            await self.checkpoint()
        return result

    async def run(
        self, backend, messages, extra_body=None, tools_enabled=False, required_tool=None, media_followup=True
    ):
        previous = self.planning_active
        previous_media_tool = self.authorized_media_tool
        self.planning_active = tools_enabled
        self.authorized_media_tool = required_tool if tools_enabled and required_tool in MEDIA_TOOLS else None
        try:
            return await self._run(backend, messages, extra_body, tools_enabled, required_tool, media_followup)
        finally:
            self.planning_active = previous
            self.authorized_media_tool = previous_media_tool

    async def _run(
        self, backend, messages, extra_body=None, tools_enabled=False, required_tool=None, media_followup=True
    ):
        from .models import ChatResponse, _add_usage, _extract_openai_usage, _strip_fake_tool_calls
        from .tools import MediaValidationFailure, execute_tool, image_was_sent_this_round, voice_was_sent_this_round

        client, kwargs, formatted, supports_tools, selected_tools = backend.prepare_request(
            messages, extra_body, tools_enabled, required_tool, media_followup
        )
        # Retain completed tool observations when ModelManager changes provider.
        formatted.extend(self.transcript)
        kwargs["messages"] = formatted
        log = general_logger.bind(category="llm")
        accumulated_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            max_tool_rounds = max(1, int(config.llm_max_tool_rounds))
        except (TypeError, ValueError):
            max_tool_rounds = 3
        media_sent_during_call = False
        required_tool_attempted = False
        required_tool_forced = False
        media_attempted_during_call = False
        media_retry_pending = False
        media_validation_retries = 0
        media_failure_reply = (
            "语音生成或发送失败，请稍后再试。" if required_tool == "send_voice" else "图片生成或发送失败，请稍后再试。"
        )
        try:
            response = await self.step(backend, client, kwargs)
            accumulated_usage = _add_usage(accumulated_usage, _extract_openai_usage(response))
            tool_round = self.tool_round
            while True:
                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise RuntimeError("上游返回空 choices（可能被安全过滤或服务异常）")
                message = choices[0].message
                if message is None:
                    finish_reason = getattr(choices[0], "finish_reason", None)
                    raise RuntimeError(f"上游返回空 message（finish_reason={finish_reason!r}，常见于安全过滤命中）")
                finish_reason = getattr(choices[0], "finish_reason", None)
                if finish_reason in {"length", "content_filter"}:
                    raise RuntimeError(f"模型输出未完整结束（finish_reason={finish_reason}），不能发送残缺正文或执行残缺工具")
                tool_calls = list(getattr(message, "tool_calls", None) or [])
                has_real_tool_calls = bool(tool_calls)
                log.debug(
                    f"[adapter] spec={backend.model_name} model_id={backend.model_id} "
                    f"supports_tools={supports_tools} returned_tool_calls={has_real_tool_calls} "
                    f"tool_round={tool_round}/{max_tool_rounds} "
                    f"content_len={len(getattr(message, 'content', None) or '')}"
                )

                if not has_real_tool_calls:
                    if media_retry_pending:
                        # A textual imitation is not a corrected function call.
                        log.warning("[adapter] 参数修正轮未返回真实工具调用，结束媒体请求")
                        return ChatResponse(media_failure_reply, usage=accumulated_usage)
                    if required_tool and not required_tool_attempted:
                        if required_tool_forced or tool_round >= max_tool_rounds:
                            raise RuntimeError(f"模型未按要求调用 required_tool={required_tool}")
                        # 允许模型先自主 recall/search；只有它准备直接结束时，
                        # 才用 named tool_choice 保证 Planner 的媒体意图被执行。
                        kwargs["messages"] = formatted
                        kwargs["tools"] = selected_tools
                        kwargs["tool_choice"] = {
                            "type": "function",
                            "function": {"name": required_tool},
                        }
                        required_tool_forced = True
                        log.debug(f"[adapter] 模型准备结束但尚未执行必需工具，强制调用 {required_tool}")
                        response = await self.step(backend, client, kwargs)
                        accumulated_usage = _add_usage(
                            accumulated_usage,
                            _extract_openai_usage(response),
                        )
                        continue
                    break
                if not supports_tools:
                    log.warning(f"[adapter] 忽略 {backend.model_name} 在未授权情况下返回的 tool_calls")
                    break
                if not kwargs.get("tools"):
                    # A provider may ignore the tool-free final request. Never
                    # execute another side effect after media has completed.
                    log.warning("[adapter] 忽略无工具收尾请求中返回的 tool_calls")
                    if media_sent_during_call:
                        return ChatResponse("", usage=accumulated_usage)
                    break
                if tool_round >= max_tool_rounds:
                    log.warning(
                        f"[adapter] spec={backend.model_name} 达到工具轮次上限 {max_tool_rounds}，忽略额外 tool_calls"
                    )
                    break

                tool_round += 1
                self.tool_round = tool_round
                tool_names = [str(call.function.name) for call in tool_calls]
                current_tool_names = {item["function"]["name"] for item in kwargs["tools"]}
                unexpected_tool_names = [name for name in tool_names if name not in current_tool_names
                                         or (name in MEDIA_TOOLS and name != self.authorized_media_tool)]
                if unexpected_tool_names:
                    raise RuntimeError(
                        f"模型返回了本轮未授权的工具 {unexpected_tool_names}；允许的工具为 {sorted(current_tool_names)}"
                    )
                if required_tool_forced and required_tool not in tool_names:
                    raise RuntimeError(f"模型忽略指定 required_tool，返回 {tool_names}，期望 {required_tool}")
                if backend._provider == "nvidia":
                    kwargs["tool_choice"] = "auto"
                else:
                    kwargs.pop("tool_choice", None)
                assistant_call = (
                    message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
                )
                formatted.append(assistant_call)
                # Provider reasoning is transient request state. Persist the
                # observable calls/content, not the provider's reasoning text.
                self.transcript.append(
                    {key: value for key, value in assistant_call.items() if key in {"role", "content", "tool_calls"}}
                )
                recall_before_media = "recall_history" in tool_names and any(
                    name in MEDIA_TOOLS for name in tool_names
                )
                media_attempted = False
                media_sent_this_round = False
                retryable_media_tool = None
                media_retry_pending = False

                # Only independent read calls share a batch; side effects retain
                # the ordered media validation/defer path below.
                parallel_results = None
                if len(tool_calls) > 1 and all(name in READ_TOOLS for name in tool_names):
                    parallel_results = await asyncio.gather(
                        *(
                            self.execute(call.function.name, call.function.arguments, execute_tool)
                            for call in tool_calls
                        )
                    )

                for call_index, tool_call in enumerate(tool_calls):
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    log.info(f"[tool_call] round={tool_round}/{max_tool_rounds} {func_name} args={str(raw_args)[:200]}")
                    is_media = func_name in MEDIA_TOOLS
                    if is_media and media_attempted:
                        tool_result = "本轮已有媒体调用，其余媒体调用已跳过；不得重复生成或发送。"
                    elif recall_before_media and is_media:
                        tool_result = (
                            "本轮媒体调用已暂缓：请先读取 recall_history 的结果，下一轮再根据召回内容重新调用媒体工具。"
                        )
                    else:
                        if is_media:
                            media_attempted_during_call = True
                            media_failure_reply = (
                                "语音生成或发送失败，请稍后再试。" if func_name == "send_voice"
                                else "图片生成或发送失败，请稍后再试。"
                            )
                        tool_result = (
                            parallel_results[call_index]
                            if parallel_results is not None
                            else await self.execute(func_name, raw_args, execute_tool)
                        )
                        if is_media:
                            media_attempted = True
                            if isinstance(tool_result, MediaValidationFailure):
                                retryable_media_tool = func_name
                                media_failure_reply = "媒体请求失败，尚未生成或发送：" + str(tool_result)
                            if func_name == required_tool:
                                required_tool_attempted = True

                    max_tool_chars = TOOL_RESULT_LOG_MAX_CHARS
                    tool_result_text = str(tool_result)
                    tool_result_truncated = False
                    if max_tool_chars > 0 and len(tool_result_text) > max_tool_chars:
                        tool_result_text = (
                            tool_result_text[:max_tool_chars]
                            + f"\n...<工具结果日志截断，原始长度 {len(str(tool_result))} 字，已保留前 {max_tool_chars} 字>"
                        )
                        tool_result_truncated = True
                    tool_result_block = "\n".join(f"  {line}" for line in tool_result_text.splitlines()) or "  <empty>"
                    log.info(
                        f"[tool_result] round={tool_round}/{max_tool_rounds} "
                        f"{func_name} len={len(str(tool_result))} "
                        f"truncated={tool_result_truncated}\n{tool_result_block}"
                    )

                    observation = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(tool_result),
                    }
                    formatted.append(observation)
                    self.transcript.append(observation)

                    media_sent_this_round = image_was_sent_this_round() or voice_was_sent_this_round()
                    media_sent_during_call = media_sent_during_call or media_sent_this_round

                kwargs["messages"] = formatted
                if (
                    retryable_media_tool == self.authorized_media_tool
                    and retryable_media_tool is not None
                    and not media_sent_during_call
                    and media_validation_retries < 1
                    and tool_round < max_tool_rounds
                ):
                    # Only a preflight rejection proves no quota/delivery side
                    # effect occurred. Allow one correction within the round cap.
                    media_validation_retries += 1
                    media_retry_pending = True
                    kwargs["tools"] = [
                        item for item in selected_tools if item["function"]["name"] == retryable_media_tool
                    ]
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": retryable_media_tool},
                    }
                    log.info(
                        f"[adapter] 媒体参数校验失败，允许一次修正 "
                        f"tool={retryable_media_tool} round={tool_round}/{max_tool_rounds}"
                    )
                elif media_attempted:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    if not media_sent_during_call:
                        # Do not ask the model to invent a failure explanation,
                        # or fall back and risk a second billable generation.
                        return ChatResponse(media_failure_reply, usage=accumulated_usage)
                    if media_sent_this_round and not media_followup:
                        # No text part follows the media: the side effect itself
                        # completes the reply. Avoid another model request.
                        log.debug(f"[adapter] 媒体已发送，输出计划中无后续 text，结束 round={tool_round}")
                        return ChatResponse("", usage=accumulated_usage)

                    # Successful media with a planned text part gets exactly
                    # one tool-free turn; delivery remains non-repeatable.
                    log.debug(
                        f"[adapter] 媒体工具已执行，发起最终无工具回复 "
                        f"round={tool_round} sent={media_sent_this_round} "
                        f"followup={media_followup}"
                    )
                elif tool_round >= max_tool_rounds:
                    if required_tool and not required_tool_attempted:
                        raise RuntimeError(f"达到工具轮次上限前未执行 required_tool={required_tool}")
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    log.debug(f"[adapter] 工具轮次已达上限，发起最终无工具请求 round={tool_round}")
                else:
                    kwargs["tools"] = selected_tools
                    if required_tool and not required_tool_attempted and tool_round >= max_tool_rounds - 1:
                        kwargs["tool_choice"] = {
                            "type": "function",
                            "function": {"name": required_tool},
                        }
                        required_tool_forced = True
                    else:
                        if backend._provider == "nvidia":
                            kwargs["tool_choice"] = "auto"
                        else:
                            kwargs.pop("tool_choice", None)
                    log.debug(f"[adapter] 继续下一轮工具请求 round={tool_round + 1}/{max_tool_rounds}")

                response = await self.step(backend, client, kwargs)
                accumulated_usage = _add_usage(accumulated_usage, _extract_openai_usage(response))
        except asyncio.CancelledError as error:
            error.usage = accumulated_usage
            raise
        except Exception as e:
            if media_sent_during_call:
                # Never retry the whole model fallback chain after the QQ media
                # has already been sent; doing so could generate a duplicate.
                log.warning(f"[adapter] 媒体已发送但最终收尾失败，保留媒体并返回空文字: {e}")
                return ChatResponse("", usage=accumulated_usage)
            if media_attempted_during_call:
                log.warning(f"[adapter] 媒体尝试后请求失败，禁止整条模型链重试: {e}")
                return ChatResponse(media_failure_reply, usage=accumulated_usage)
            # 降为 warning：fallback 链会接管，最终全部失败时由 ModelManager 打 error
            from .diagnostics import error_summary
            log.warning(f"[adapter] {backend.model_name} API 调用异常: {error_summary(e)}")
            error = RuntimeError(f"API 请求出现错误: {error_summary(e)}")
            error.usage = accumulated_usage
            raise error from e

        content = message.content or ""
        if not content and not backend._uses_buffered_stream():
            for attr in ("output_text",):
                val = getattr(message, attr, None)
                if isinstance(val, str) and val.strip():
                    content = val
                    break
        if not content and not backend._uses_buffered_stream():
            try:
                if isinstance(message, dict):
                    content = message.get("content") or message.get("output_text") or ""
                elif hasattr(message, "model_dump"):
                    md = message.model_dump()
                    content = md.get("content") or md.get("output_text") or ""
            except (TypeError, ValueError, AttributeError) as error:
                log.debug(f"[adapter] final payload unavailable: {type(error).__name__}")
        if not content:
            log.warning(f"[adapter] {backend.model_name} 所有内容字段都为空")
            if not media_sent_during_call:
                if media_attempted_during_call:
                    return ChatResponse(media_failure_reply, usage=accumulated_usage)
                error = RuntimeError("模型没有返回最终答复，不能使用推理字段替代")
                error.usage = accumulated_usage
                raise error

        if tools_enabled:
            from .conversation_style import unpack_plan, looks_like_plan
            plan = unpack_plan(content)
            if plan is None and looks_like_plan(content):
                from .memory_index import digest
                log.warning(f"[planner_format] run={self.id} model={backend.model_name} "
                            f"length={len(content)} hash={digest(content)[:16]} raw={content[:1200]!r}")
                error = ValueError("planner 返回了无效的规划格式")
                error.usage = accumulated_usage
                raise error
            if plan:
                self.response_mode = plan["response_mode"]
                self.should_reply = plan["should_reply"]
                self.reply_requirements = plan["reply_requirements"]
                self.reply_mode = plan["reply_mode"]
                self.answer_draft = plan["answer_draft"]
                self.identity_question = plan["verify_group_identity"] and self.response_mode != "social"
                content = plan["content"]
        if is_tool_call_text(content):
            log.warning(f"[adapter] 拦截正文中的工具调用 JSON spec={backend.model_name}")
            return ChatResponse(
                "" if media_sent_during_call else INVALID_REPLY,
                usage=accumulated_usage,
            )
        cleaned = _strip_fake_tool_calls(content or "")
        if cleaned != (content or ""):
            log.info(
                f"[adapter] 已剥离伪工具调用文本 spec={backend.model_name} "
                f"原长度={len(content or '')} 清理后={len(cleaned)}"
            )
        from .identity_guard import UNCERTAIN, verify_identity

        if (self.identity_question and self.response_mode != "social"
                and self.session_id.startswith("group_") and not media_attempted_during_call):
            try:
                cleaned, verification_usage = await verify_identity(self, cleaned, backend, client, kwargs)
                accumulated_usage = _add_usage(accumulated_usage, verification_usage)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 -- a failed verifier must never publish the unverified identity.
                log.warning(f"[identity_guard] run={self.id} unavailable={type(error).__name__}")
                self.identity_verdict = {"status": "unavailable", "error": type(error).__name__}
                cleaned = UNCERTAIN
        return ChatResponse(
            cleaned,
            getattr(message, "reasoning_details", None),
            usage=accumulated_usage,
        )


async def deliver_reply(response,text,send, *, bot=None, group_id=None):
    from .model_identity import delivery_lock, update_group_card_for_response
    from .conversation import ensure_send_allowed
    async with delivery_lock(bot, group_id):
        sent = False
        async def transport():
            nonlocal sent
            ensure_send_allowed(group_id)
            receipt = await send()
            sent = True
            return receipt
        runner = getattr(response, "agent_run", None)
        receipt = await runner.deliver(text, transport) if runner else await transport()
        # Recovered receipts and ambiguous sends must not change the current card.
        if sent and text and isinstance(receipt, dict) and receipt.get("message_id") is not None:
            await update_group_card_for_response(bot, group_id, response)
        return receipt


def confirmed_reply(state):
    """Research completion does not imply successful delivery to the user."""
    return "\n".join(
        action["text"] for action in (state.get("actions") or {}).values()
        if action.get("status") == "sent" and isinstance(action.get("text"), str)
    )
