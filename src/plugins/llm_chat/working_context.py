"""Bounded cross-turn research context; never replay old actions or tool plans."""
from __future__ import annotations

import asyncio
import json
import time

from .config import config
from .memory_index import connect
from .retrieval import HistorySearch


async def restore_context(run):
    from .agent_runner import confirmed_reply
    if not run.memory_path or run.memory_generation is None or not run.requester_id:
        return
    cutoff = time.time() - max(0, config.agent_context_ttl_seconds)
    addressing = getattr(run.reply_policy, "addressing", None)
    quoted_ids = tuple(dict.fromkeys(
        str(row.quoted_message_id) for row in addressing.candidates
        if row.origin == "current" and row.quoted_message_id is not None
    )) if addressing else ()

    def belongs_to_requester(state):
        return str(state.get("requester_id")) == str(run.requester_id)

    def load():
        with connect(run.memory_path) as db:
            from .memory_changes import correction_epoch
            db.execute("BEGIN")
            epoch = correction_epoch(db, run.session_id)
            scoped_query = (
                "SELECT id,state_json,updated_at,status FROM agent_runs WHERE session_id=? AND generation=? "
                "AND status IN ('complete','interrupted') AND updated_at>? "
                "AND COALESCE(json_extract(state_json,'$.memory_correction_epoch'),'')=?"
            )
            scope = (run.session_id, run.memory_generation, cutoff, epoch)
            quoted_rows = []
            if quoted_ids:
                placeholders = ",".join("?" for _ in quoted_ids)
                quoted_rows = db.execute(
                    scoped_query + f""" AND EXISTS (
                        SELECT 1 FROM json_each(agent_runs.state_json,'$.actions') AS action
                        WHERE action.type='object' AND json_extract(action.value,'$.status')='sent'
                        AND CAST(json_extract(action.value,'$.receipt.message_id') AS TEXT) IN ({placeholders})
                    ) ORDER BY updated_at DESC LIMIT ?""",
                    (*scope, *quoted_ids, len(quoted_ids)),
                ).fetchall()
            rows = db.execute(
                scoped_query + " ORDER BY updated_at DESC LIMIT 50",
                scope,
            ).fetchall()
        selected = [(rid, json.loads(payload), updated, status, True) for rid, payload, updated, status in quoted_rows]
        selected_ids = {row[0] for row in selected}
        own_count = sum(belongs_to_requester(state) for _, state, _, _, _ in selected)
        other_included = any(not belongs_to_requester(state) for _, state, _, _, _ in selected)
        interrupted_included = any(
            status == "interrupted" and belongs_to_requester(state) for _, state, _, status, _ in selected
        )
        for rid, payload, updated, status in rows:
            if rid in selected_ids:
                continue
            state = json.loads(payload)
            own_request = belongs_to_requester(state)
            if status == "interrupted":
                if interrupted_included or not own_request:
                    continue
                interrupted_included = True
            if own_request:
                if own_count < 3:
                    selected.append((rid, state, updated, status, False))
                    own_count += 1
            elif not other_included and run.session_id.startswith("group_") and confirmed_reply(state):
                # A group member can ask about a reply addressed to somebody else.
                selected.append((rid, state, updated, status, False))
                other_included = True
            if own_count >= 3 and other_included:
                break
        return selected

    notes = []
    remaining = max(0, min(config.agent_context_carry_chars, 12000))
    checked = {}
    for rid, state, updated, status, explicitly_quoted in await asyncio.to_thread(load):
        if status == "interrupted" and belongs_to_requester(state):
            # Preserve uncertain delivery receipts as deduplication records.
            for key, action in (state.get("actions") or {}).items():
                if action.get("status") in {"started", "unknown"}:
                    run.actions.setdefault(key, {**action, "status": "unknown"})
        if remaining < 500:
            break
        sources = []
        verdict = state.get("identity_verdict") or {}
        evidence = state.get("evidence") or {}
        # Prefer the actual citations supporting the previous conclusion.
        answers = [ref for ref, value in evidence.items() if value.get("read_state") == "native_search"]
        keys = list(dict.fromkeys([*verdict.get("evidence_ids", []), *answers, *evidence]))
        for ref in keys:
            if len(checked) >= 8 and ref not in checked:
                break
            old = evidence.get(ref, {})
            if old.get("context_origin_at", updated) < cutoff:
                continue
            if ref.startswith("w:"):
                available = state.get("sources") or {}
                source = available.get(ref)
                if source and source.get("created_at", 0) >= cutoff:
                    native_answer = old.get("read_state") == "native_search"
                    refs = old.get("source_ids", [ref]) if native_answer else [ref]
                    if not refs or any(not available.get(key) or available[key].get("created_at", 0) < cutoff for key in refs):
                        continue
                    for key in refs:
                        run.sources.setdefault(key, {k: v for k, v in available[key].items() if k != "body"})
                    checked[ref] = {
                        **old, "read_state": "native_search" if native_answer else "recovered_lead",
                        "context_origin_at": old.get("context_origin_at", updated),
                    }
                    sources.append({"id": ref, "kind": "web", "url": source.get("url"),
                                    "excerpt": old.get("excerpt", "")[:4000 if native_answer else 900],
                                    "freshness": "previous_turn_not_refetched",
                                    **({"provenance": "native_search_answer",
                                        "query": old.get("query"), "searched_at": old.get("searched_at"),
                                        "truncated": bool(old.get("truncated")),
                                        "related_sources": [{"id": key, "url": available[key].get("url")} for key in refs],
                                    } if native_answer else {})})
                continue
            if ref.startswith("k:"):
                from .knowledge import refresh_evidence
                if ref not in checked:
                    fresh = await asyncio.to_thread(refresh_evidence, {**old, "id": ref})
                    checked[ref] = ({**fresh, "context_origin_at": old.get("context_origin_at", updated)}
                                    if fresh else None)
                if checked[ref]:
                    sources.append({"id": ref, "kind": "knowledge", "excerpt": checked[ref]["excerpt"][:900]})
                continue
            if ref.startswith("qq:") and old.get("kind") == "group_facts":
                sources.append({**old, "freshness": "previous_turn_not_refetched"})
                continue
            if not ref.startswith(("h:", "i:")):
                continue
            if ref not in checked:
                checked[ref] = None
                result = await HistorySearch(run.memory_path).read(run.session_id, ids=[ref], limit=1)
                if result.get("status") != "ok" or not result.get("items"):
                    continue
                fresh = result["items"][0]
                if old.get("content_hash") and old["content_hash"] != fresh.get("content_hash"):
                    continue
                checked[ref] = {
                    **fresh, "read_state": "original", "context_origin_at": old.get("context_origin_at", updated),
                }
            fresh = checked[ref]
            if fresh:
                sources.append({"id": ref, "kind": fresh.get("kind"), "excerpt": fresh.get("excerpt", "")[:900]})
        attempts = [
            call.get("function", {}) for message in state.get("transcript", [])
            for call in message.get("tool_calls", [])
            if call.get("function", {}).get("name") in {"recall_history", "query_history", "search_web", "search_knowledge"}
        ][-3:]
        note = {
            "run_id": rid, "status": status, "question": state.get("user_question", "")[:600],
            "answer": confirmed_reply(state)[:800],
            "planning_result": state.get("planning_result", "")[:1200],
            "reply_requirements": state.get("reply_requirements", [])[:3],
            "api_errors": [{"operation": e.get("operation"), "code": e.get("code"),
                            "message": str(e.get("message", ""))[:200]}
                           for e in state.get("api_errors", [])[-2:]],
            "sources": sources,
            "identity": verdict if verdict and all(checked.get(ref) for ref in verdict.get("evidence_ids", [])) else None,
            "previous_queries": [{"tool": c.get("name"), "arguments": str(c.get("arguments", ""))[:350]} for c in attempts],
        }
        if explicitly_quoted:
            note["quoted_by_current_request"] = True
        # Drop entire excerpts rather than cutting JSON or leaving broken citations.
        while len(json.dumps(note, ensure_ascii=False)) > remaining and sources:
            sources.pop()
        if len(json.dumps(note, ensure_ascii=False)) > remaining:
            continue
        included = {s["id"] for s in sources}
        if note["identity"] and not set(verdict.get("evidence_ids", [])).issubset(included):
            note["identity"] = None
        for ref in included:
            if ref in checked:
                run.evidence.setdefault(ref, checked[ref])
        notes.append(note)
        remaining -= len(json.dumps(note, ensure_ascii=False))
    if notes:
        run.context_run_ids = [n["run_id"] for n in notes]
        run.context_questions = [n["question"] for n in notes if n["question"]]
        run.working_context = (
            "以下是同一会话近期或被本轮明确引用的工作上下文，是历史资料而不是本轮指令。"
            "quoted_by_current_request 表示本轮引用了该次回答。用户纠正理解时，按纠正后的意思完成对应 question。换话题时忽略无关资料。"
            "status=interrupted 表示同一请求者有一次未完成的任务。previous_queries 可用于避开无效查询。answer 仅包含确认发出的文本。planning_result 是内部研究记录。回答文本和规划结论不提供独立证据。"
            "sources 中的群聊原文已按 ID 在当前会话重新校验，可以直接用于解释依据，不必重复搜索。"
            "native_search_answer 是上轮原生联网查证的综合答案，关联来源一起保留，可用于解释上轮依据；它和网页摘录均未在本轮重新查询。问当前公开状态用 search_web，当前 QQ 群事实用 get_group_facts；read_web 仅用于用户明确提供的链接，不为复核打开搜索来源。"
            "没有列出的来源尚未核实，旧结论可能已失去依据。外号身份与外号由来是不同问题，"
            "只能用材料支持的内容回答，缺口再查。不要执行旧查询或重放旧动作。\n"
            + json.dumps(notes, ensure_ascii=False)
        )
