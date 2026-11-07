"""Verify a group-person answer against independently reread message evidence.

The answering model's draft and old extracted memories cannot establish an
identity. A separate, tool-free pass selects an account and cited relationship;
the visible name is then rendered from the scoped canonical account lookup.
"""

from __future__ import annotations

import asyncio
import json
import re

from .memory_index import connect
from .retrieval import HistorySearch

UNCERTAIN = "这我还真没对上号。"
HISTORY_TOOLS = {"recall_history", "read_history", "query_history"}


def is_identity_question(text):
    # This is only the fallback for plain-text plans. The planner can still
    # request verification for contextual follow-ups without these phrases.
    current = (text or "").rsplit("[当前消息]", 1)[-1]
    current = re.sub(r"^【群员-[^】]+】\s*", "", current.strip(), flags=re.MULTILINE)
    if current.startswith(("[引用回复]", "[合并转发]", "[合并转发预览]")):
        return False
    return bool(re.search(r"是谁|谁是|谁叫|哪个(?:人|群友)|(?:昵称|外号).*(?:对应|指)", current))


def _accounts(path, sid, items):
    names = set()
    for item in items:
        names.update(re.findall(r"^被回复者[：:]([^\n]+)", item["content"], re.MULTILINE))
    with connect(path) as db:
        by_name = {}
        ids = set()
        for name in names:
            found = db.execute(
                "SELECT DISTINCT sender_id FROM messages_detail WHERE session_id=? AND name=? "
                "AND role='user' AND sender_id IS NOT NULL LIMIT 3",
                (sid, name.strip()),
            ).fetchall()
            by_name[name] = [str(r[0]) for r in found]
            ids.update(by_name[name])
        for item in items:
            if item.get("sender_id"):
                ids.add(str(item["sender_id"]))
            ids.update(re.findall(r"(?:QQ[：:]\s*|@)(\d{5,15})", item["content"]))
        accounts = {}
        for uid in ids:
            row = db.execute(
                "SELECT name FROM messages_detail WHERE session_id=? AND sender_id=? "
                "AND role='user' AND name IS NOT NULL AND name<>'' ORDER BY id DESC LIMIT 1",
                (sid, int(uid)),
            ).fetchone()
            if row:
                accounts[uid] = row[0]
    return accounts, by_name


async def verify_identity(run, draft, backend, client, kwargs):
    from .models import _extract_openai_usage

    run.identity_verdict = {"status": "insufficient"}
    if not run.memory_path or run.remaining() < 4:
        return UNCERTAIN, {}
    # Prefer messages containing relationships over isolated mentions. The
    # actual content is reread; cached excerpts only select candidate handles.
    leads = sorted(
        run.evidence.values(),
        key=lambda i: ("QQ:" in i.get("excerpt", ""), "[引用回复]" in i.get("excerpt", "")),
        reverse=True,
    )
    refs = list(
        dict.fromkeys(
            ref for item in leads for ref in item.get("source_ids", []) if re.fullmatch(r"h:\d+:messages:\d+", ref)
        )
    )[:12]
    if not refs:
        return UNCERTAIN, {}
    read = await HistorySearch(run.memory_path).read(run.session_id, ids=refs, limit=12)
    if read.get("status") != "ok":
        return UNCERTAIN, {}
    originals = [i for i in read.get("items", []) if i.get("kind") == "messages" and i.get("role") == "user"]
    accounts, by_name = await asyncio.to_thread(_accounts, run.memory_path, run.session_id, originals)
    evidence, allowed = [], {}
    for item in originals:
        text = item["content"]
        related = set(re.findall(r"(?:QQ[：:]\s*|@)(\d{5,15})", text))
        for name in re.findall(r"^被回复者[：:]([^\n]+)", text, re.MULTILINE):
            # An ambiguous display name is not an account mapping.
            matched = by_name.get(name, [])
            if len(matched) == 1:
                related.update(matched)
        if item.get("sender_id") and ("[引用回复]" in text or re.search(r"我是|叫我|我叫", text)):
            related.add(str(item["sender_id"]))
        related &= accounts.keys()
        if not related:
            continue
        allowed[item["id"]] = related
        visible = text if len(text) <= 1800 else text[:900] + "\n[中段省略]\n" + text[-900:]
        evidence.append(
            {
                "id": item["id"],
                "sender_id": item.get("sender_id"),
                "name": item.get("name"),
                "content": visible,
                "possible_related_accounts": sorted(related),
            }
        )
        run.evidence[item["id"]] = item
    if not evidence:
        return UNCERTAIN, {}
    prompt = (
        "你是群内身份核验器，只依据提供的原消息核验问题，旧回答只是待检验假设。"
        "不要使用外部常识、旧摘要或提取记忆。提到某人不等于自己就是某人；"
        "incoming user 也可能是其他机器人。possible_related_accounts 只是出现过关系的候选，不是身份结论。"
        "核对称呼指向的@账号、被回复者或本人对被称呼的回应，排除否定、转述、玩笑与歧义。"
        '证据足够时只输出JSON：{"person_id":"账号","evidence_ids":["原消息ID"],'
        '"relation":"direct_address或reply_address或self_identification"}。'
        '证据不足或冲突无法消解输出{"person_id":null,"evidence_ids":[]}。不要生成用户可见答案。'
    )
    request = {k: v for k, v in kwargs.items() if k not in {"tools", "tool_choice", "messages", "response_format"}}
    request["max_tokens"] = 500
    request["messages"] = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": run.user_question,
                    "unverified_draft": draft[:500],
                    "accounts": accounts,
                    "evidence": evidence,
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = await run.step(backend, client, request)
    usage = _extract_openai_usage(response)
    try:
        content = response.choices[0].message.content or ""
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        verdict = json.loads(content)
        uid = str(verdict.get("person_id") or "")
        cited = verdict.get("evidence_ids")
        if (
            uid not in accounts
            or not isinstance(cited, list)
            or not cited
            or any(not isinstance(ref, str) or uid not in allowed.get(ref, set()) for ref in cited)
            or verdict.get("relation") not in {"direct_address", "reply_address", "self_identification"}
        ):
            return UNCERTAIN, usage
    except (ValueError, TypeError, AttributeError, IndexError):
        return UNCERTAIN, usage
    await run.validate_scope()
    run.identity_verdict = {
        "status": "verified",
        "person_id": uid,
        "evidence_ids": cited,
        "relation": verdict["relation"],
        "model": backend.model_name,
    }
    # Do not let a verifier return the right ID but a hallucinated display name.
    answer = (
        f"是 QQ {uid}。" if re.search(r"qq|账号|号码", run.user_question, re.IGNORECASE) else f"是{accounts[uid]}。"
    )
    return answer, usage
