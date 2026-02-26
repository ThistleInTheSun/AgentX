"""
主入口：常驻运行，到达发送时间时按配置的「信息类别」逐类拉取 -> 总结 -> 推送到微信等。
用法：在项目根目录执行 python -m digest_info.run
"""
import sys
import time
from pathlib import Path

# 保证项目根目录在 path 中，这样无论从项目根还是 digest_info/ 内运行都能 import digest_info
_root = Path(__file__).resolve().parent
_project_root = _root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 注册所有内置实现（import 即注册）
from digest_info.sources import hackernews, rss_source, arxiv_source, tech_news_feeds  # noqa: F401
from digest_info.summarizers import plain_summarizer, llm_summarizer  # noqa: F401
from digest_info.notifiers import wechat                 # noqa: F401
from digest_info.categories import hackernews as cat_hn, rss as cat_rss, tech_daily  # noqa: F401

from digest_info.core import (
    CategoryRegistry,
    SummarizerRegistry,
    NotifierRegistry,
)
from digest_info.core.source import SearchResult
from digest_info.core.schedule import is_due, get_window_hours
from digest_info.config import load_config, load_category_config
from digest_info.sources.arxiv_source import fetch_one_paper_by_id, ArxivSource

# 常驻模式下：未到发送时间时每多少秒检查一次；执行完一次后冷却多少秒，避免同一时间窗口重复执行
CHECK_INTERVAL_SEC = 60
COOLDOWN_AFTER_RUN_SEC = 360


def is_any_category_due() -> bool:
    """检查是否有任意已配置类别当前处于发送时间窗口内。"""
    cfg = load_config()
    category_ids = cfg.get("categories", [])
    if category_ids and isinstance(category_ids[0], dict):
        category_ids = [c.get("id") for c in category_ids if c.get("id")]
    timezone = cfg.get("timezone") or None
    for cid in category_ids:
        cat_cfg = load_category_config(cid)
        schedule = cat_cfg.get("schedule")
        cat_tz = (schedule or {}).get("timezone") or timezone
        if is_due(schedule, timezone=cat_tz):
            return True
    return False


def run():
    cfg = load_config()
    # 启用的类别 id 列表；每个类别的 params、schedule 从该类别目录下的 config.yaml 读
    category_ids = cfg.get("categories", [])
    if category_ids and isinstance(category_ids[0], dict):
        category_ids = [c.get("id") for c in category_ids if c.get("id")]
    summarizer_id = cfg.get("summarizer", "plain")
    summarizer_params = cfg.get("summarizer_params") or {}
    notifiers_cfg = cfg.get("notifiers", [])
    timezone = cfg.get("timezone") or None

    summarizer = SummarizerRegistry.get(summarizer_id, **summarizer_params) or SummarizerRegistry.get("plain")
    parts: list[str] = []
    total_count = 0

    for cid in category_ids:
        cat_cfg = load_category_config(cid)
        params = cat_cfg.get("params") or {}
        schedule = cat_cfg.get("schedule")
        cat_tz = (schedule or {}).get("timezone") or timezone
        if not is_due(schedule, timezone=cat_tz):
            continue
        window_hours = get_window_hours(schedule)
        if window_hours is not None:
            params = {**params, "window_hours": window_hours}

        cat = CategoryRegistry.get(cid, **params)
        if cat is None:
            print(f"未知信息类别: {cid}，已跳过")
            continue
        try:
            results = cat.fetch(**params)
        except Exception as e:
            print(f"拉取类别 {cid} 失败: {e}")
            continue
        if not results:
            print(f"类别 {cid} 拉取 0 条（可能网络或源无数据），已跳过")
            continue
        total_count += len(results)
        prompt = cat.get_prompt()
        body_part = summarizer.summarize(
            results,
            source_id=cat.get_display_name(),
            category_id=cid,
            prompt=prompt,
        )
        parts.append(body_part)

    if not parts:
        print("本次没有拉取到任何内容，跳过推送")
        return

    body = "\n\n---\n\n".join(parts)
    title = f"信息摘要 · {total_count} 条"

    for item in notifiers_cfg:
        nid = item.get("id")
        params = item.get("params") or {}
        notifier = NotifierRegistry.get(nid, **params)
        if notifier is None:
            print(f"未知推送渠道: {nid}，已跳过")
            continue
        try:
            ok = notifier.send(title, body, **params)
            print(f"推送 {nid}: {'成功' if ok else '失败'}")
        except Exception as e:
            print(f"推送 {nid} 异常: {e}")


# 测试用：固定拉取一篇已知论文（Attention Is All You Need）并总结
_TEST_ARXIV_ID = "1706.03762"
# 完整验证：过去一周的 arXiv 论文数量
_TEST_FULL_LIMIT = 10


def run_test():
    """拉取一篇固定存在的论文并总结，用于验证网络与流程。"""
    print("测试模式：拉取单篇论文并总结…")
    results = fetch_one_paper_by_id(_TEST_ARXIV_ID)
    if not results:
        print("拉取失败：未获取到论文（请检查网络或代理）。")
        return
    summarizer = SummarizerRegistry.get("plain")
    body = summarizer.summarize(
        results,
        source_id="arXiv 测试",
        category_id="test",
        prompt="以下为单篇论文摘要（用于验证拉取与总结流程）。",
    )
    print("--- 测试摘要 ---")
    print(body)
    print("--- 测试完成 ---")


def run_test_full():
    """仅用 arXiv 验证完整流程：拉取过去一周论文并总结（不依赖 RSS、不推送）。"""
    print("完整验证：拉取过去一周 arXiv 论文并总结…")
    source = ArxivSource(
        categories=["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
        limit=_TEST_FULL_LIMIT,
    )
    results = source.fetch(window_hours=24 * 7, limit=_TEST_FULL_LIMIT)
    if not results:
        print("拉取失败：过去一周未获取到论文（请检查网络或代理）。")
        return
    summarizer = SummarizerRegistry.get("plain")
    body = summarizer.summarize(
        results,
        source_id="arXiv（过去一周）",
        category_id="test_full",
        prompt="以下为过去一周 arXiv CS 相关新论文摘要。",
    )
    print(f"--- 共 {len(results)} 条 ---")
    print(body)
    print("--- 完整验证完成 ---")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_test()
        sys.exit(0)
    if "--test-full" in sys.argv:
        run_test_full()
        sys.exit(0)
    print("信息摘要服务已启动，将按各类别 schedule 在发送时间执行拉取与推送。")
    while True:
        if not is_any_category_due():
            time.sleep(CHECK_INTERVAL_SEC)
            continue
        run()
        time.sleep(COOLDOWN_AFTER_RUN_SEC)
