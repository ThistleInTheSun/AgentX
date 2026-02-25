"""
通用 RSS 搜索源：可配置任意 RSS 地址，便于扩展新信息源。
"""
import xml.etree.ElementTree as ET
import requests
from ..core.source import Source, SearchResult, SourceRegistry


def _parse_rss_feed(url: str, limit: int = 10) -> list[SearchResult]:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "InfoDigest/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        return [SearchResult(title="RSS 获取失败", url=url, summary=str(e), source_id="rss")]
    # 兼容 RSS 2.0 与 Atom
    items = []
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item")[:limit]:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            summary = (desc_el.text or "").strip() if desc_el is not None else ""
            if title or link:
                items.append(SearchResult(title=title, url=link, summary=summary[:200], source_id="rss"))
    else:
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry")[:limit]:
            title_el = entry.find("{http://www.w3.org/2005/Atom}title")
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            summary_el = entry.find("{http://www.w3.org/2005/Atom}summary")
            title = (title_el.text or "").strip() if title_el is not None else ""
            link = ""
            if link_el is not None and link_el.get("href"):
                link = link_el.get("href", "")
            summary = (summary_el.text or "").strip() if summary_el is not None else ""
            if title or link:
                items.append(SearchResult(title=title, url=link, summary=summary[:200], source_id="rss"))
    return items


@SourceRegistry.register("rss")
class RSSSource(Source):
    """通过 RSS URL 拉取条目。"""
    source_id = "rss"

    def __init__(self, feed_url: str, limit: int = 10):
        self.feed_url = feed_url
        self.limit = limit

    def fetch(self, **kwargs) -> list[SearchResult]:
        url = kwargs.get("feed_url", self.feed_url)
        limit = kwargs.get("limit", self.limit)
        return _parse_rss_feed(url, limit)
