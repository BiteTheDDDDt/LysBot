"""Offline real-model diagnostic: paraphrase retrieval, provenance, latency, and scope."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Synthetic facts; no names/IDs in the queries and no metadata hints at retrieval time.
PAIRS = [
    ("我需要一台能随身带、离开插座也能用一整天的电脑。", "有人想要轻薄且续航持久的笔记本吗"),
    ("这副耳塞戴上以后，地铁轰隆隆的声音几乎听不见了。", "谁聊过主动降噪的耳机"),
    ("我在阳台种的番茄今天终于变红了，摘下来做了晚饭。", "谁收获了自己栽培的蔬菜"),
    ("手机掉进水里以后黑屏了，按开机键完全没有反应。", "谁的移动电话进液无法启动"),
    ("准备把帐篷和睡袋装进背包，周末去山里住一夜。", "有人计划野外露宿吗"),
    ("会议记录太长了，我只想留下决议、负责人和交付日期。", "谁需要把讨论整理成行动事项"),
    ("显卡显存装不下这个模型，改成四比特后终于能跑了。", "谁通过量化降低了推理的内存需求"),
    ("这个键盘按起来没有咔哒声，晚上敲字不会吵到室友。", "谁想在深夜安静打字"),
    ("我把照片同时存到两块硬盘，其中一块放在父母家。", "谁提到了异地保存数据副本"),
    ("公司搬到更远的地方，每天来回要花三个小时在路上。", "谁抱怨通勤耗时很长"),
    ("快递箱打开是碎的玻璃杯，商家已经同意退钱。", "谁收到破损商品后申请了退款"),
    ("我不吃肉，昨天用豆腐、蘑菇和青菜做了一桌晚餐。", "谁习惯吃素食"),
    ("网页每刷新一次都要重新登录，关掉浏览器也留不住会话。", "谁遇到了身份验证状态无法保持的问题"),
    ("路由器在客厅，卧室隔了两堵墙，视频总是一卡一卡的。", "谁的无线网络覆盖不好"),
    ("我给猫买了一个不用插电、里面有滚动小球的玩具。", "谁购买了宠物娱乐用品"),
    ("我先把每个月必交的钱留出来，其余的才拿去买喜欢的东西。", "谁讲过先安排固定开销再消费"),
    ("这次更新以后旧版本的配置文件读不了了，我写了转换脚本。", "谁处理过升级时的数据格式兼容"),
    ("打印出来的纸总是带着一道黑线，换了墨盒还是一样。", "谁反映过输出文档出现条纹"),
    ("我把重复的电话号码合并了，现在通讯录里每个人只留一条。", "谁清理了联系人中的冗余记录"),
    ("周末要骑车沿河走四十公里，已经把轮胎和刹车检查好了。", "谁安排了长距离自行车活动"),
    ("家里洗衣机的门关不紧，转动的时候水会从缝里漏出来。", "谁说家用电器密封损坏造成渗水"),
    ("这本小说的结局直到最后一页才揭晓凶手，我一晚上就读完了。", "谁看过悬疑推理故事"),
    ("我的植物叶片全垂下来了，才想起已经十天没给它浇水。", "谁忘记照料盆栽导致缺水"),
    ("我给服务加了重试间隔，失败以后不会一秒钟发几百次请求了。", "谁缓解了异常情况下的重试风暴"),
]


async def evaluate(args):
    import nonebot

    nonebot.init()
    from src.plugins.llm_chat import memory
    from src.plugins.llm_chat.local_embedding import MODEL, REVISION
    from src.plugins.llm_chat.memory_index import MemoryIndex, embed
    from src.plugins.llm_chat.retrieval import HistorySearch

    manager = memory.memory_manager
    manager._db_path = args.directory / "history.db"
    manager._closing = True
    db = await manager._get_db()
    index = MemoryIndex(manager._db_path)
    started = time.monotonic()
    report = {"model": MODEL, "revision": REVISION, "settings": index.settings, "cases": [], "network_allowed": False}
    try:
        existing = (await (await db.execute("SELECT count(*) FROM messages_detail")).fetchone())[0]
        if existing:
            raise ValueError("Use an empty diagnostic directory")
        start = datetime(2026, 9, 1, tzinfo=timezone(timedelta(hours=8)))
        rows = [
            (i + 1, "group_900000099", "user", 80000 + i, f"成员{i}", content, (start + timedelta(hours=i)).isoformat())
            for i, (content, _) in enumerate(PAIRS)
        ]
        # Exact matching content in another group must never enter this group's candidates.
        rows += [
            (1000 + i, "group_900000098", "user", 90000, "外群", query, start.isoformat())
            for i, (_, query) in enumerate(PAIRS)
        ]
        await db.executemany(
            "INSERT INTO messages_detail(id,session_id,role,sender_id,name,content,timestamp) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        await db.commit()
        await asyncio.to_thread(index._prepare_vectors)
        while await asyncio.to_thread(index._project_one):
            pass
        while rows := await asyncio.to_thread(index._pending_vectors):
            vectors = await embed([row["content"] for row in rows], index.settings)
            await asyncio.to_thread(index._commit_vectors, rows, vectors)
        await asyncio.to_thread(index._activate_if_complete)
        report["build_seconds"] = round(time.monotonic() - started, 3)
        search = HistorySearch(manager._db_path)
        for i, (content, query) in enumerate(PAIRS):
            result = {"query": query, "expected_id": i + 1}
            for mode in ["keyword", "semantic", "hybrid"]:
                before = time.monotonic()
                found = await search.search("group_900000099", query=query, mode=mode, limit=5)
                if found["status"] not in {"ok", "empty"}:
                    raise AssertionError(
                        f"Unexpected search result: {mode}: {found.get('warnings')} {found.get('error')}"
                    )
                rank = next((n + 1 for n, item in enumerate(found["items"]) if content in item["content"]), None)
                assert all("外群" not in json.dumps(item, ensure_ascii=False) for item in found["items"])
                result[mode] = {"rank": rank, "milliseconds": round((time.monotonic() - before) * 1000, 2)}
                if mode == "hybrid" and rank:
                    source = found["items"][rank - 1]["id"]
                    read = await search.read("group_900000099", ids=[source])
                    assert content in json.dumps(read, ensure_ascii=False)
                    assert all(item["read_state"] == "original" for item in read["items"])
            report["cases"].append(result)
        report["summary"] = {
            mode: {
                "recall_at_1": sum(case[mode]["rank"] == 1 for case in report["cases"]) / len(PAIRS),
                "recall_at_3": sum(0 < (case[mode]["rank"] or 99) <= 3 for case in report["cases"]) / len(PAIRS),
                "p50_ms": sorted(case[mode]["milliseconds"] for case in report["cases"])[len(PAIRS) // 2],
                "max_ms": max(case[mode]["milliseconds"] for case in report["cases"]),
            }
            for mode in ["keyword", "semantic", "hybrid"]
        }
        report["passed"] = (
            report["summary"]["semantic"]["recall_at_3"] >= 0.85
            and report["summary"]["hybrid"]["recall_at_3"] >= report["summary"]["keyword"]["recall_at_3"]
        )
        print(
            json.dumps(
                {"passed": report["passed"], "summary": report["summary"], "build_seconds": report["build_seconds"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if not report["passed"]:
            raise AssertionError("Local semantic diagnostic did not meet threshold")
    finally:
        (args.directory / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, default=ROOT / "data/models/bge-small-zh-v1.5")
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    if not args.directory.name.startswith("agentic-local-eval"):
        raise ValueError("Diagnostic directory name must start with agentic-local-eval")
    args.directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        LLM_CHAT_DATA_DIR=str(args.directory),
        LLM_CHAT_DATABASE_PATH=str(args.directory / "history.db"),
        MEMORY_INDEX_ENABLED="false",
        MEMORY_LOCAL_MODEL_DIR=str(args.model_directory.resolve()),
    )
    with asyncio.Runner() as runner:
        runner.get_loop()  # Windows creates a local wakeup socket before the network-denial assertion.
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network forbidden in offline embedding diagnostic")
        ):
            runner.run(evaluate(args))


if __name__ == "__main__":
    main()
