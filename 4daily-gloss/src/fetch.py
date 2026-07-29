"""拉取 OWID atom feed 与文章全文。"""
import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from . import config

log = logging.getLogger(__name__)
ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass
class Article:
    title: str
    url: str
    published: str
    authors: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def _http_get(url: str) -> str:
    resp = requests.get(
        url, timeout=config.HTTP_TIMEOUT, headers={"User-Agent": config.USER_AGENT}
    )
    resp.raise_for_status()
    return resp.text


def fetch_feed() -> list[Article]:
    """返回 feed 中的条目（未拉正文）。"""
    root = ET.fromstring(_http_get(config.FEED_URL))
    articles = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
        link_el = entry.find(f"{ATOM_NS}link")
        url = link_el.get("href") if link_el is not None else ""
        published = (entry.findtext(f"{ATOM_NS}published") or "")[:10]
        authors = [
            (a.findtext(f"{ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        if title and url:
            articles.append(Article(title, url, published, [a for a in authors if a]))
    log.info("feed 共 %d 条", len(articles))
    return articles


_TAG_RE = re.compile(r"<[^>]+>")
_PARA_RE = re.compile(r'<p class="article-block__text[^"]*">(.*?)</p>', re.S)


def fetch_fulltext(article: Article) -> None:
    """抓取文章页正文段落，就地填充 article.paragraphs。"""
    page = _http_get(article.url)
    paras = []
    for raw in _PARA_RE.findall(page):
        text = html.unescape(_TAG_RE.sub("", raw)).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            paras.append(text)
    # 超长文章截断为节选，控制翻译成本与公众号篇幅
    total = 0
    kept = []
    for p in paras:
        if total + len(p) > config.MAX_TRANSLATE_CHARS and kept:
            article.truncated = True
            break
        kept.append(p)
        total += len(p)
    article.paragraphs = kept
    log.info("正文 %d 段 / %d 字符%s", len(kept), total, "（节选）" if article.truncated else "")


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) >= config.MIN_ASCII_RATIO


def pick_article(articles: list[Article], is_processed) -> Article | None:
    """按 feed 顺序选第一篇：未处理过、英文、足够长。"""
    for art in articles:
        if is_processed(art.url):
            log.debug("跳过（已处理）：%s", art.url)
            continue
        try:
            fetch_fulltext(art)
        except Exception as e:
            log.warning("拉取正文失败，跳过 %s: %s", art.url, e)
            continue
        if len(art.text) < config.MIN_ARTICLE_CHARS:
            log.info("跳过（过短 %d 字符）：%s", len(art.text), art.title)
        elif not is_english(art.text):
            log.info("跳过（非英文）：%s", art.title)
        else:
            return art
    return None
