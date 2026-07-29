"""组装 Markdown：论文元信息 + 轻松解读 + 原文摘要 + 来源声明。"""
from datetime import date


def _authors_line(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    line = ", ".join(authors[:6])
    if len(authors) > 6:
        line += " 等"
    return line


def _categories_line(categories: list[str]) -> str:
    return " / ".join(categories[:5]) if categories else "未分类"


def assemble(paper, explanation: dict, publish_day: date) -> str:
    authors = _authors_line(paper.authors)
    categories = _categories_line(paper.categories)
    tags = " · ".join(explanation["tags"]) if explanation["tags"] else "好玩论文"

    parts = [
        f"# 今日好玩论文 | {paper.title}",
        "",
        f"> 作者：{authors}",
        f"> 发布时间：{paper.published}",
        f"> 分类：{categories}",
        f"> 标签：{tags}",
        f"> arXiv：{paper.url}",
        "",
        f"{explanation['hook']}",
        "",
        "## 这篇论文在聊什么",
        "",
        explanation["what_is_it"],
        "",
        "## 最有趣的点",
        "",
        explanation["coolest_part"],
        "",
        "## 跟我有什么关系",
        "",
        explanation["why_care"],
        "",
        "## 一个冷知识",
        "",
        explanation["fun_fact"],
        "",
        "## 原文摘要",
        "",
        paper.abstract,
        "",
        "---",
        "",
        f"> 本文来自 arXiv，版权归原作者所有。"
        f"本文仅为中文解读，供学习交流，原文链接：{paper.url}",
        "",
    ]
    return "\n".join(parts)
