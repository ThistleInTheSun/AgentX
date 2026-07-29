"""调用 LLM（OpenAI 兼容 API）把论文抽象解读成中文。"""
import json
import logging

import requests

from . import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _chat(messages: list[dict], *, json_mode: bool = False) -> str:
    cfg = config.llm_config()
    if not cfg["api_key"]:
        raise LLMError("LLM_API_KEY 未配置（见 .env.example）")
    body = {"model": cfg["model"], "messages": messages, "temperature": 0.4}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        json=body,
        timeout=300,
    )
    if resp.status_code != 200:
        raise LLMError(f"LLM API {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def explain(paper) -> dict:
    """生成面向 AI 从业者/爱好者的中文解读。"""
    categories = ", ".join(paper.categories[:5])
    authors = ", ".join(paper.authors[:6])
    if len(paper.authors) > 6:
        authors += " 等"
    prompt = f"""请对以下 arXiv 论文进行中文解读，目标读者是对 AI 感兴趣的技术从业者与爱好者。

标题：{paper.title}
作者：{authors}
分类：{categories}
发布时间：{paper.published}

摘要：
{paper.abstract}

请输出 JSON，格式如下：
{{
  "one_sentence": "一句话总结这篇论文（不超过 60 字）",
  "why_matters": "为什么这项研究值得关注？它解决了什么问题或填补了哪些空白？",
  "key_innovation": "核心创新点是什么？方法或思路上有何不同？",
  "key_results": "关键结果、实验结论或主要贡献",
  "audience": "适合哪些人读？需要哪些前置知识？",
  "tags": ["标签1", "标签2", "标签3"]
}}
"""
    content = _chat([
        {"role": "system", "content": "你是 AI 领域研究者，擅长把论文核心内容提炼成清晰、准确的中文解读。"},
        {"role": "user", "content": prompt},
    ], json_mode=True)
    data = json.loads(content)
    return {
        "one_sentence": data.get("one_sentence", "").strip(),
        "why_matters": data.get("why_matters", "").strip(),
        "key_innovation": data.get("key_innovation", "").strip(),
        "key_results": data.get("key_results", "").strip(),
        "audience": data.get("audience", "").strip(),
        "tags": [t.strip() for t in data.get("tags", []) if t.strip()],
    }
