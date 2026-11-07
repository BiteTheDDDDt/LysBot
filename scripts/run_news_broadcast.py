"""对已发布简报执行一次真实新闻搜索 dry-run，不发群、不写消费记录。"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nonebot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--group",
        dest="group_ids",
        type=int,
        nargs="+",
        help="一个或多个群号",
    )
    source.add_argument(
        "--topic",
        dest="topics",
        nargs="+",
        help="用公开测试话题执行同链路，不读取群简报",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=72,
        help="允许读取多老的已发布简报，默认 72 小时",
    )
    return parser.parse_args()


async def _close_resources() -> None:
    from src.plugins.llm_chat.memory import memory_manager
    from src.plugins.llm_chat.models import model_manager
    from src.plugins.llm_chat.usage_db import close_db as close_usage_db

    await asyncio.sleep(0.2)
    await memory_manager.close()
    for adapter in list(model_manager._adapters.values()):
        client = getattr(adapter, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                await client.close()
            except Exception:
                pass
    await close_usage_db()


async def _main(args: argparse.Namespace) -> int:
    nonebot.init()
    from src.plugins.llm_chat.memory import memory_manager
    from src.plugins.llm_chat.news_broadcast import (
        build_news_text,
        generate_news_payload,
        render_news_image,
        save_news_image,
    )

    def render_preview(payload: dict, group_name: str, group_id: int) -> None:
        if not payload.get("items"):
            return
        image = render_news_image(
            payload,
            group_name,
            group_id,
            now.date().isoformat(),
        )
        preview_path = save_news_image(image, group_id)
        print(f"图片预览：{preview_path.resolve()}", flush=True)

    now = datetime.now()
    cutoff = now - timedelta(hours=max(1, args.max_age_hours))
    exit_code = 0
    try:
        if args.topics:
            print("\n===== 公开话题 / 新闻播报 dry-run =====", flush=True)
            payload = await generate_news_payload(
                {"topics": args.topics},
                now=now,
                session_id="news_dry_run_public",
            )
            if payload is None:
                print("搜索链路失败。", flush=True)
                return 2
            print(
                f"兴趣点：{'、'.join(payload.get('focus_topics') or []) or '无'}；"
                f"合格新闻：{len(payload.get('items') or [])} 条；"
                f"搜索模型：{payload.get('search_model') or '无'}",
                flush=True,
            )
            print(build_news_text(payload), flush=True)
            render_preview(payload, "公开话题 dry-run", 0)
            return 0

        for group_id in args.group_ids or []:
            print(f"\n===== 群 {group_id} / 新闻播报 dry-run =====", flush=True)
            briefing = await memory_manager.get_latest_published_briefing(
                group_id,
                published_after=cutoff,
            )
            if briefing is None:
                print(f"没有找到 {args.max_age_hours} 小时内已发布的简报。", flush=True)
                exit_code = 1
                continue

            print(
                f"简报 #{briefing['id']}，发布日期 {briefing['briefing_date']}，"
                f"发布时间 {briefing['published_at']}",
                flush=True,
            )
            payload = await generate_news_payload(
                briefing["card_data"],
                now=now,
                session_id=f"group_{group_id}",
            )
            if payload is None:
                print("搜索链路失败；这份简报会保留，不应标记为无新闻。", flush=True)
                exit_code = 2
                continue

            print(
                f"兴趣点：{'、'.join(payload.get('focus_topics') or []) or '无'}；"
                f"合格新闻：{len(payload.get('items') or [])} 条；"
                f"搜索模型：{payload.get('search_model') or '无'}",
                flush=True,
            )
            print(build_news_text(payload), flush=True)
            render_preview(payload, f"群 {group_id}", group_id)
    finally:
        await _close_resources()
    return exit_code


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(_main(_parse_args())))
