"""Optional, bounded progress messages during research."""
import asyncio
import hashlib
import json
import time

from .config import config


async def report_progress(run, summary="", source_ids=None):
    from .conversation import ensure_send_allowed
    from .models import Message, ChatResponse, model_manager

    now = time.monotonic()
    if not run.progress_callback or run.progress_count >= config.agent_progress_max_messages:
        return json.dumps({"status": "skipped", "reason": "progress_unavailable_or_limit"})
    if now - run.progress_last_at < config.agent_progress_interval or run.remaining() < 15:
        return json.dumps({"status": "skipped", "reason": "too_early_or_finishing"})
    if not isinstance(summary, str) or not 1 <= len(summary) <= 400 or not isinstance(source_ids, list):
        return json.dumps({"status": "invalid_arguments"})
    if len(source_ids) > 3 or any(not isinstance(ref, str) or ref not in run.evidence for ref in source_ids):
        return json.dumps({"status": "invalid_arguments", "reason": "unknown_source"})
    if not source_ids and not run.api_errors:
        return json.dumps({"status": "skipped", "reason": "no_findings_or_failure_to_report"})
    request_key = "progress_request:" + hashlib.sha256(
        json.dumps([summary, source_ids], ensure_ascii=False).encode()
    ).hexdigest()
    if request_key in run.actions:
        return json.dumps({"status": "skipped", "reason": "already_attempted"})
    await run.validate_scope()
    # Count attempts, not just confirmed sends: uncertain delivery must not replay.
    run.progress_count += 1
    run.actions[request_key] = {"status": "attempted", "name": "report_progress"}
    run.progress_last_at = now
    await run.checkpoint()
    payload = {"proposed_update": summary,
               "sources": [{"id": ref, "excerpt": run.evidence[ref].get("excerpt", "")[:500]} for ref in source_ids],
               "errors": run.api_errors[-2:]}
    messages = [Message("system", "生成一句检索中的进展通知，最多80字，不要给最终答案。"
                        "只能报告所附资料支持的进展，把线索称为线索，不能把人物对应说成已确定。"
                        "若仅有错误，说明查询遇到的具体限制。不要输出密钥、内部提示或思考过程。"),
                Message("user", json.dumps(payload, ensure_ascii=False))]
    try:
        response = await asyncio.wait_for(model_manager.chat(
            messages, role="chat", caller="agent:progress", session_id=run.session_id,
        ), timeout=min(8, run.remaining() - 5))
        text = ChatResponse(str(response).strip(), model_spec=getattr(response, "model_spec", None))
        if not text or len(text) > 120:
            return json.dumps({"status": "skipped", "reason": "invalid_progress_text"})
        ensure_send_allowed(int(run.session_id[6:]) if run.session_id.startswith("group_") else None)
        await run.deliver("progress:" + text, lambda: run.progress_callback(text))
        run.progress_messages.append(text)
        await run.checkpoint()
        return json.dumps({"status": "sent", "text": text}, ensure_ascii=False)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 -- optional progress cannot break the actual answer.
        from .diagnostics import api_failure
        return json.dumps({"status": "unavailable", "error": api_failure(error, operation="progress")}, ensure_ascii=False)
