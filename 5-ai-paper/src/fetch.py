"""拉取 arXiv AI 相关论文并筛选。"""
import html
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from . import config

log = logging.getLogger(__name__)
ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass
class Paper:
    title: str
    url: str
    arxiv_id: str
    published: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.abstract


def _http_get(url: str) -> str:
    resp = requests.get(
        url, timeout=config.HTTP_TIMEOUT, headers={"User-Agent": config.USER_AGENT}
    )
    resp.raise_for_status()
    return resp.text


def _build_query() -> str:
    cats = "+OR+".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    return (
        "http://export.arxiv.org/api/query?"
        f"search_query={cats}&"
        "sortBy=submittedDate&"
        "sortOrder=descending&"
        f"max_results={config.ARXIV_MAX_RESULTS}"
    )


def fetch_papers() -> list[Paper]:
    """返回 arXiv 上最新的一批 AI 相关论文。"""
    url = _build_query()
    log.info("拉取 arXiv feed ...")
    root = ET.fromstring(_http_get(url))
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = html.unescape((entry.findtext(f"{ATOM_NS}title") or "")).strip()
        summary = html.unescape((entry.findtext(f"{ATOM_NS}summary") or "")).strip()
        published = (entry.findtext(f"{ATOM_NS}published") or "")[:10]
        link_el = entry.find(f"{ATOM_NS}id")
        url = (link_el.text or "").strip() if link_el is not None else ""
        arxiv_id = url.rstrip("/").split("/")[-1] if url else ""
        authors = [
            (a.findtext(f"{ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        categories = [
            c.get("term", "") for c in entry.findall(f"{ATOM_NS}category")
        ]
        if title and url:
            papers.append(Paper(
                title=title,
                url=url,
                arxiv_id=arxiv_id,
                published=published,
                authors=[a for a in authors if a],
                abstract=summary,
                categories=[c for c in categories if c],
            ))
    log.info("feed 共 %d 篇", len(papers))
    return papers


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) >= config.MIN_ASCII_RATIO


def pick_paper(papers: list[Paper], is_processed) -> Paper | None:
    """按时间顺序选第一篇：未处理过、英文、摘要长度合适。"""
    for paper in papers:
        if is_processed(paper.url):
            log.debug("跳过（已处理）：%s", paper.url)
            continue
        if not is_english(paper.abstract):
            log.info("跳过（非英文）：%s", paper.title)
            continue
        if len(paper.abstract) < config.MIN_ABSTRACT_CHARS:
            log.info("跳过（摘要过短 %d 字符）：%s", len(paper.abstract), paper.title)
            continue
        if len(paper.abstract) > config.MAX_ABSTRACT_CHARS:
            log.info("跳过（摘要过长 %d 字符）：%s", len(paper.abstract), paper.title)
            continue
        return paper
    return None
