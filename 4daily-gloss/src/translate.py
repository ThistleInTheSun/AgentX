"""调用 LLM（OpenAI 兼容 API，默认 DeepSeek）：翻译 + 生词 + 长难句。"""
import json
import logging
import re

import requests

from . import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _chat(messages: list[dict], *, json_mode: bool = False) -> str:
    cfg = config.llm_config()
    if not cfg["api_key"]:
        raise LLMError("LLM_API_KEY 未配置（见 .env.example）")
    body = {"model": cfg["model"], "messages": messages, "temperature": 0.3}
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


def translate(paragraphs: list[str]) -> list[str]:
    """逐段翻译，返回与原文段落对齐的中文段落列表。"""
    numbered = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))
    content = _chat([
        {"role": "system", "content": (
            "你是资深英译中翻译，风格准确、通顺、适合公众号阅读。"
            "将用户给出的编号段落逐段翻译成简体中文。"
            "只输出 JSON：{\"translations\": [\"第1段译文\", \"第2段译文\", ...]}，"
            "数组长度必须与原文段落数一致，不要合并或拆分段落。"
        )},
        {"role": "user", "content": numbered},
    ], json_mode=True)
    data = json.loads(content)
    trans = data.get("translations", [])
    if len(trans) != len(paragraphs):
        log.warning("译文段数 %d 与原文 %d 不一致，按顺序对齐", len(trans), len(paragraphs))
    return [str(t).strip() for t in trans]


def extract_study_notes(text: str) -> dict:
    """抽取 10 个考研向生词 + 1 个长难句拆解。"""
    content = _chat([
        {"role": "system", "content": (
            "你是考研英语辅导老师。从用户给出的英文文章中：\n"
            "1. 挑选 10 个考研核心词汇或高频短语（优先大纲词汇中的中高难度词，"
            "避免过于简单的词如 the/make/good）。\n"
            "2. 挑选 1 个最有代表性的长难句，做成分拆解并给出翻译。\n"
            "只输出 JSON，格式：\n"
            "{\"vocab\": [{\"word\": \"英文词\", \"ipa\": \"音标（可为空字符串）\", "
            "\"meaning\": \"中文释义（含词性）\", \"example\": \"该词所在的原文例句\"}],\n"
            " \"sentence\": {\"original\": \"原句\", \"analysis\": \"结构拆解（主干+修饰成分，中文说明）\", "
            "\"translation\": \"整句中文翻译\"}}\n"
            "vocab 恰好 10 项，example 必须摘自原文。"
        )},
        {"role": "user", "content": text},
    ], json_mode=True)
    data = json.loads(content)
    vocab = data.get("vocab", [])[:10]
    sentence = data.get("sentence", {})
    if len(vocab) < 10 or not sentence.get("original"):
        raise LLMError(f"生词/长难句抽取不完整：vocab={len(vocab)}")
    return {"vocab": vocab, "sentence": sentence}
