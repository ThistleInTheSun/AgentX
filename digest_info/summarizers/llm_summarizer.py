"""
LLM 总结器：将多条 SearchResult 与类别 prompt 一并发给大模型，生成结构化摘要。
需在 config 中配置 summarizer: llm 及 summarizer_params（如 api_key、model、base_url）。
API Key 建议通过环境变量传递，勿写入仓库。
"""
import os
import json
import requests
from ..core.source import SearchResult
from ..core.summarizer import Summarizer, SummarizerRegistry


def _build_messages(results: list[SearchResult], source_id: str, category_id: str, prompt: str) -> list[dict]:
    raw_list = []
    for i, r in enumerate(results, 1):
        raw_list.append({
            "序号": i,
            "标题": r.title,
            "链接": r.url,
            "来源": r.source_id or source_id,
            "摘要": (r.summary or "")[:500],
        })
    user_content = (
        f"【类别】{category_id or source_id}\n\n"
        f"【整理要求】\n{prompt.strip()}\n\n"
        f"【原始条目】（请按上述要求整理成「科技新闻」与「新论文」两部分，每条保留标题、来源、链接与一句话摘要）\n"
        f"{json.dumps(raw_list, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": "你是一个科技信息摘要助手，负责把原始条目按用户要求整理成清晰、有条理的日报格式。"},
        {"role": "user", "content": user_content},
    ]


@SummarizerRegistry.register("llm")
class LLMSummarizer(Summarizer):
    """使用 OpenAI 或兼容 API 对条目进行总结。"""
    summarizer_id = "llm"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"

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
        if not self.api_key:
            return (
                f"【{category_id or source_id}】共 {len(results)} 条（未配置 OPENAI_API_KEY / summarizer_params.api_key，此处仅列标题）\n\n"
                + "\n".join(f"{i}. {r.title}\n   {r.url}" for i, r in enumerate(results, 1))
            )
        messages = _build_messages(results, source_id, category_id, prompt)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return content.strip() or "（模型未返回内容）"
        except Exception as e:
            return (
                f"【{category_id or source_id}】共 {len(results)} 条；LLM 调用失败: {e}\n\n"
                + "\n".join(f"{i}. {r.title}\n   {r.url}" for i, r in enumerate(results, 1))
            )
