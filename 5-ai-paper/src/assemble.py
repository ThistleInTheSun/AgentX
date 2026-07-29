"""组装 Markdown：论文元信息 + 中文解读 + 原文摘要 + 来源声明。"""
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
    tags = " · ".join(explanation["tags"]) if explanation["tags"] else "AI 论文"

    parts = [
        f"# AI 论文速读 | {paper.title}",
        "",
        f"> 作者：{authors}",
        f"> 发布时间：{paper.published}",
        f"> 分类：{categories}",
        f"> 标签：{tags}",
        f"> arXiv：{paper.url}",
        "",
        f"**{explanation['one_sentence']}**",
        "",
        "## 一句话总结",
        "",
        explanation["one_sentence"],
        "",
        "## 为什么值得关注",
        "",
        explanation["why_matters"],
        "",
        "## 核心创新",
        "",
        explanation["key_innovation"],
        "",
        "## 关键结果",
        "",
        explanation["key_results"],
        "",
        "## 适合谁读",
        "",
        explanation["audience"],
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
