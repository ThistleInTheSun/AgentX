"""组装 Markdown：翻译 + 生词 + 长难句 + 艾宾浩斯复习表 + 版权声明。"""
from datetime import date, timedelta

REVIEW_OFFSETS = [("Day 1", 1), ("Day 2", 2), ("Day 4", 4), ("Day 7", 7), ("Day 15", 15)]


def ebbinghaus_table(publish_day: date) -> str:
    lines = [
        "| 复习节点 | 日期 | 复习内容 | 完成 |",
        "| --- | --- | --- | --- |",
    ]
    for label, offset in REVIEW_OFFSETS:
        day = publish_day + timedelta(days=offset)
        lines.append(f"| {label} | {day.strftime('%m-%d')} | 生词 10 个 + 长难句 | ☐ |")
    return "\n".join(lines)


def copyright_block(article) -> str:
    authors = "、".join(article.authors) or "Our World in Data"
    return (
        "---\n\n"
        f"> 原文：{article.title}\n"
        f"> 发布日期：{article.published}\n"
        f"> 作者：{authors}\n"
        "> 来源：Our World in Data\n"
        f"> 链接：{article.url}\n"
        "> 许可：CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）\n"
        "> 本文为中文翻译，仅供学习交流。"
    )


def assemble(article, translations: list[str], notes: dict, publish_day: date) -> str:
    parts = [f"# 每日外刊精读 | {article.title}", ""]
    if article.published:
        parts.append(f"原文发布于 {article.published} · Our World in Data")
        parts.append("")

    parts.append("## 原文 & 翻译")
    parts.append("")
    if article.truncated:
        parts.append("（原文较长，本篇为核心段落节选，完整原文见文末链接）")
        parts.append("")
    for i, para in enumerate(article.paragraphs):
        parts.append(para)
        parts.append("")
        if i < len(translations):
            parts.append(f"> {translations[i]}")
            parts.append("")

    parts.append("## 生词打卡（10 个）")
    parts.append("")
    for i, v in enumerate(notes["vocab"], 1):
        ipa = f" {v['ipa']}" if v.get("ipa") else ""
        parts.append(f"**{i}. {v['word']}**{ipa}")
        parts.append("")
        parts.append(f"- 释义：{v['meaning']}")
        parts.append(f"- 例句：{v['example']}")
        parts.append("")

    s = notes["sentence"]
    parts.append("## 长难句拆解")
    parts.append("")
    parts.append(f"**原句：** {s['original']}")
    parts.append("")
    parts.append(f"**拆解：** {s['analysis']}")
    parts.append("")
    parts.append(f"**翻译：** {s['translation']}")
    parts.append("")

    parts.append("## 艾宾浩斯复习表")
    parts.append("")
    parts.append(f"发布日：{publish_day.isoformat()}")
    parts.append("")
    parts.append(ebbinghaus_table(publish_day))
    parts.append("")
    parts.append("背完今天的 10 个词，来评论区扣「打卡 Day 1」，之后每个复习日回来接龙——坚持到 Day 15 的人，词是真的忘不掉。")
    parts.append("")

    parts.append(copyright_block(article))
    parts.append("")
    return "\n".join(parts)
