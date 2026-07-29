"""调用 LLM（OpenAI 兼容 API）筛选有趣论文并生成轻松解读。"""
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
    body = {"model": cfg["model"], "messages": messages, "temperature": 0.7}
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


def pick_most_interesting(papers: list) -> object | None:
    """从候选论文中选出对大众最有趣的一篇，返回 Paper 对象。"""
    if not papers:
        return None
    if len(papers) == 1:
        return papers[0]

    items = []
    for i, paper in enumerate(papers[:12], 1):
        items.append(f"[{i}] 标题：{paper.title}\n分类：{', '.join(paper.categories[:3])}\n摘要：{paper.abstract[:500]}")
    numbered = "\n\n".join(items)

    prompt = f"""下面有 {len(items)} 篇最新论文的标题和摘要片段。
请从「大众读者会觉得好玩、有趣、猎奇、涨知识」的角度，给每篇打分（1-10 分），并选出最有趣的一篇。

{numbered}

请只输出 JSON：
{{
  "scores": [
    {{"index": 1, "score": 8, "reason": "为什么有趣或无趣"}},
    ...
  ],
  "winner_index": 1
}}
index 从 1 开始。
"""
    content = _chat([
        {"role": "system", "content": "你是科学传播编辑，擅长发现论文中对大众最有意思的点。"},
        {"role": "user", "content": prompt},
    ], json_mode=True)
    data = json.loads(content)
    winner = data.get("winner_index", 1)
    if not isinstance(winner, int) or winner < 1 or winner > len(papers):
        winner = 1
    return papers[winner - 1]


def explain(paper) -> dict:
    """生成轻松、面向大众的中文解读。"""
    categories = ", ".join(paper.categories[:5])
    authors = ", ".join(paper.authors[:6])
    if len(paper.authors) > 6:
        authors += " 等"
    prompt = f"""请用轻松、好奇、讲人话的语气，解读下面这篇论文。目标读者是普通大众，不要太学术。

标题：{paper.title}
作者：{authors}
分类：{categories}
发布时间：{paper.published}

摘要：
{paper.abstract}

请输出 JSON，格式如下：
{{
  "hook": "一个吸引人的开头，50 字以内，用来当文章导语",
  "what_is_it": "这篇论文到底在研究什么？用人话解释",
  "coolest_part": "最有趣、最 surprising 的点是什么？",
  "why_care": "这件事跟我们普通人有什么关系？",
  "fun_fact": "一个可以当谈资的延伸小知识或冷知识",
  "tags": ["标签1", "标签2", "标签3"]
}}
"""
    content = _chat([
        {"role": "system", "content": "你是科普写手，擅长把学术论文讲成有趣的中文短文，让人读完想转发。"},
        {"role": "user", "content": prompt},
    ], json_mode=True)
    data = json.loads(content)
    return {
        "hook": data.get("hook", "").strip(),
        "what_is_it": data.get("what_is_it", "").strip(),
        "coolest_part": data.get("coolest_part", "").strip(),
        "why_care": data.get("why_care", "").strip(),
        "fun_fact": data.get("fun_fact", "").strip(),
        "tags": [t.strip() for t in data.get("tags", []) if t.strip()],
    }
