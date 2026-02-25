"""
简单总结器：把多条结果格式化成标题+链接列表；支持按类别的 prompt 作为说明头。
"""
from ..core.source import SearchResult
from ..core.summarizer import Summarizer, SummarizerRegistry


@SummarizerRegistry.register("plain")
class PlainSummarizer(Summarizer):
    """纯文本列表摘要；若提供 prompt 则放在该类别块开头作为展示说明。"""
    summarizer_id = "plain"

    def summarize(
        self,
        results: list[SearchResult],
        source_id: str = "",
        category_id: str = "",
        prompt: str = "",
        **kwargs,
    ) -> str:
        if not results:
            return "（暂无新内容）"
        header = category_id or source_id or "摘要"
        lines = [f"【{header}】共 {len(results)} 条"]
        if prompt:
            lines.append(f"说明：{prompt.strip()}")
        lines.append("")
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.title}")
            if r.url:
                lines.append(f"   {r.url}")
            if r.summary:
                lines.append(f"   {r.summary[:150]}...")
            lines.append("")
        return "\n".join(lines).strip()
