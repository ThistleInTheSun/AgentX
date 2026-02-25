"""
主入口：按配置的「信息类别」逐类拉取 -> 用该类别的 prompt 总结 -> 合并推送到微信等。
用法：在项目根目录执行 python -m digest_info.run
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root.parent))

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
        print("没有拉取到任何内容，跳过推送")
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


if __name__ == "__main__":
    run()
