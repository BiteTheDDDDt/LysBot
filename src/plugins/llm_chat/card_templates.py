"""Business data -> shared card documents. No model routing or delivery side effects."""

from .card_renderer import (
    CardBlock,
    CardDocument,
    render_document,
    ACCENT_CYAN,
    ACCENT_VIOLET,
    ACCENT_CORAL,
    ACCENT_BLUE,
    TEXT_TOPIC,
    TEXT_MEMBER,
    TEXT_SUMMARY,
)
from .model_identity import format_model_spec_label

MEMBER_LIMIT = 5
BRIEFING_TITLE = "今日博物志"


def briefing_member_section(data):
    members = data.get("active_members") or []
    legacy = bool(members) and all(isinstance(item, str) for item in members)
    entries = []
    for item in members[:MEMBER_LIMIT]:
        if isinstance(item, str):
            entries.append(item)
        elif isinstance(item, dict):
            tag = item.get("personality_tag", item.get("model_archetype", ""))
            entries.append(f"{item['name']} · {tag}\n{item['comment']}")
    return ("活跃选手" if legacy else "今日群友画像"), entries or ["今日席位暂空"]


def _usage_missing_note(row):
    count = int(row.get("unknown_usage_calls") or 0)
    return f"（{count:,} 次用量未完整返回）" if count else ""


def _usage_total_label(row):
    total = int(row.get("total_tokens") or 0)
    if not total and int(row.get("unknown_usage_calls") or 0):
        return "Token 未知" + _usage_missing_note(row)
    return f"{total:,} Token" + _usage_missing_note(row)


def usage_totals(stats):
    totals = stats.get("totals") or {}
    if not totals.get("calls"):
        return ["过去 24 小时没有产生任何 AI 调用。"]
    return [
        f"调用 {int(totals.get('calls') or 0):,} 次（失败 {int(totals.get('failures') or 0):,} / 回退 {int(totals.get('fallback_calls') or 0):,}）",
        f"Token：输入 {int(totals.get('prompt_tokens') or 0):,} / 输出 {int(totals.get('completion_tokens') or 0):,}{_usage_missing_note(totals)}",
        f"延迟：平均 {(totals.get('avg_latency_ms') or 0) / 1000:.1f} 秒 / 峰值 {(totals.get('max_latency_ms') or 0) / 1000:.1f} 秒",
    ]


def usage_models(stats):
    models = sorted(stats.get("by_spec") or [], key=lambda row: int(row.get("total_tokens") or 0), reverse=True)
    rows = [(row.get("display_label") or format_model_spec_label(row.get("spec")) or "未记录模型", row) for row in models]
    return [
        (
            name,
            f"{int(row.get('calls') or 0):,} 次（成功 {int(row.get('calls') or 0) - int(row.get('failures') or 0):,} / 失败 {int(row.get('failures') or 0):,}） · {_usage_total_label(row)}",
            f"输入 {int(row.get('prompt_tokens') or 0):,} / 输出 {int(row.get('completion_tokens') or 0):,} · 失败 {int(row.get('failures') or 0):,}",
        )
        for name, row in rows
    ]


def format_usage_lines(stats):
    return usage_totals(stats) + [f"{name}  ·  {total}" for name, total, _ in usage_models(stats)]


def build_card(template, data, *, group_name, group_id, date_label, activity_text="", portrait=None, member_portraits=None):
    label = f"{date_label}  ·  {str(group_name or group_id)[:48]}"
    blocks = []
    if template == "briefing":
        member_heading, member_entries = briefing_member_section(data)
        if portrait is not None:
            blocks.append(CardBlock("每日群像", image=portrait, accent=ACCENT_VIOLET))
        blocks.extend(
            [
                CardBlock(
                    "热门话题",
                    [f"• {s}" for s in (data.get("topics") or ["今天的话题有点散。"])[:5]],
                    body_color=TEXT_TOPIC,
                ),
                CardBlock(
                    member_heading,
                    [] if member_portraits is not None else member_entries,
                    image=member_portraits,
                    image_max_height=1600,
                    accent=ACCENT_VIOLET,
                    body_color=TEXT_MEMBER,
                ),
            ]
        )
        stats = data.get("llm_stats")
        if stats:
            blocks.append(CardBlock("AI 调用统计", format_usage_lines(stats), accent=ACCENT_BLUE, body_size=23))
        return CardDocument(
            BRIEFING_TITLE,
            label,
            blocks,
            data.get("subtitle") or "过去 24 小时群聊速览",
            activity_text,
            footer=f"简报模型 · {data['briefing_model']}" if data.get("briefing_model") else "",
        )
    if template == "news":
        for i, item in enumerate((data.get("items") or [])[:10]):
            meta = " · ".join(str(item[k]) for k in ("source", "published_at") if item.get(k))
            blocks.append(
                CardBlock(
                    str(item.get("editor_title") or item.get("headline") or "新闻"),
                    [str(item.get("summary") or "")],
                    metadata=[meta] if meta else [],
                    accent=(ACCENT_CYAN, ACCENT_VIOLET, ACCENT_CORAL, ACCENT_BLUE)[i % 4],
                    body_size=23,
                    entry_lines=5,
                )
            )
        if data.get("search_model"):
            blocks.append(CardBlock("检索模型", [str(data["search_model"])], accent=ACCENT_BLUE, body_size=21))
        return CardDocument("新闻", label, blocks or [CardBlock("暂无新闻")])
    if template == "memory_review":
        for item in (data.get("memories") or [])[:10]:
            meta = item.get("provenance") or {}
            state = meta.get("memory_state", "active")
            status = {"withdrawn": "已撤回", "stale": "待更新"}.get(
                state,
                {"reported": "转述", "planned": "计划", "uncertain": "待核实", "negated": "否定"}.get(
                    meta.get("assertion"), "历史记录"
                ),
            )
            sources = meta.get("source_ids") or []
            detail = []
            if item.get("created_at"):
                detail.append("记录于 " + str(item["created_at"])[:16])
            if sources:
                detail.append(f"依据 {len(sources)} 条原消息")
            if meta.get("correction"):
                detail.append("人工纠正：" + str(meta["correction"].get("reason") or ""))
            blocks.append(
                CardBlock(
                    status, [str(item.get("content") or "")], metadata=detail, accent=ACCENT_VIOLET, entry_lines=8
                )
            )
        return CardDocument(
            "记忆回顾",
            label,
            blocks or [CardBlock("还没有记下你的长期信息")],
            subtitle="最近 10 条记录 · 历史记录不代表当前事实" if blocks else "",
        )
    raise ValueError(f"unknown_card_template: {template}")


def render_card(template, data, **context):
    return render_document(build_card(template, data, **context))
