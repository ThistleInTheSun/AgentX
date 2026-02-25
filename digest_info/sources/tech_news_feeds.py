"""
科技新闻多 RSS 聚合源：从多个 RSS 拉取条目，并按「过去 N 小时」过滤。
默认包含 TechCrunch、The Verge、Ars Technica。
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import requests
from ..core.source import Source, SearchResult, SourceRegistry

ATOM_NS = "http://www.w3.org/2005/Atom"

# 默认科技新闻 RSS 列表（可被配置覆盖）
DEFAULT_FEED_URLS = [
    ("https://techcrunch.com/feed/", "TechCrunch"),
    ("https://www.theverge.com/rss/index.xml", "The Verge"),
    ("https://feeds.arstechnica.com/arstechnica/index", "Ars Technica"),
]


def _parse_date_rss(pub_date_text: str | None) -> datetime | None:
    """解析 RSS 2.0 pubDate (RFC 2822)。"""
    if not pub_date_text or not pub_date_text.strip():
        return None
    try:
        dt = parsedate_to_datetime(pub_date_text.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _parse_date_atom(published_el) -> datetime | None:
    """解析 Atom published/updated。"""
    if published_el is None:
        return None
    text = (published_el.text or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _fetch_one_feed(
    feed_url: str,
    source_label: str,
    window_hours: int | None,
    limit_per_feed: int,
) -> list[SearchResult]:
    cutoff = None
    if window_hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    try:
        r = requests.get(
            feed_url,
            timeout=15,
            headers={"User-Agent": "InfoDigest/1.0"},
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        return [
            SearchResult(
                title=f"{source_label} 获取失败",
                url=feed_url,
                summary=str(e),
                source_id=source_label,
            )
        ]
    results = []
    channel = root.find("channel")
    if channel is not None:
        # RSS 2.0
        for item in channel.findall("item")[: limit_per_feed * 2]:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")
            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            summary = (desc_el.text or "").strip() if desc_el is not None else ""
            pub_dt = _parse_date_rss(pub_el.text if pub_el is not None else None)
            if cutoff and pub_dt is not None and pub_dt < cutoff:
                continue
            if title or link:
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        summary=summary[:200],
                        source_id=source_label,
                    )
                )
            if len(results) >= limit_per_feed:
                break
    else:
        # Atom
        for entry in root.findall(f".//{{{ATOM_NS}}}entry"):
            if len(results) >= limit_per_feed:
                break
            title_el = entry.find(f"{{{ATOM_NS}}}title")
            link_el = entry.find(f"{{{ATOM_NS}}}link[@rel='alternate']") or entry.find(f"{{{ATOM_NS}}}link")
            summary_el = entry.find(f"{{{ATOM_NS}}}summary")
            published_el = entry.find(f"{{{ATOM_NS}}}published") or entry.find(f"{{{ATOM_NS}}}updated")
            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.get("href") or "").strip() if link_el is not None else ""
            summary = (summary_el.text or "").strip() if summary_el is not None else ""
            pub_dt = _parse_date_atom(published_el)
            if cutoff and pub_dt is not None and pub_dt < cutoff:
                continue
            if title or link:
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        summary=summary[:200],
                        source_id=source_label,
                    )
                )
    return results[:limit_per_feed]


@SourceRegistry.register("tech_news_feeds")
class TechNewsFeedsSource(Source):
    """多 RSS 聚合，仅保留过去 window_hours 内的条目。"""

    source_id = "tech_news_feeds"

    def __init__(
        self,
        feed_urls: list[tuple[str, str]] | list[str] | None = None,
        limit_per_feed: int = 8,
    ):
        if feed_urls is None:
            self.feed_list: list[tuple[str, str]] = list(DEFAULT_FEED_URLS)
        else:
            self.feed_list = []
            for u in feed_urls:
                if isinstance(u, (list, tuple)):
                    self.feed_list.append((str(u[0]), str(u[1])))
                else:
                    self.feed_list.append((str(u), u))
        self.limit_per_feed = limit_per_feed

    def fetch(
        self,
        window_hours: int | None = None,
        limit_per_feed: int | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        limit_per_feed = limit_per_feed or self.limit_per_feed
        all_results = []
        for feed_url, label in self.feed_list:
            all_results.extend(
                _fetch_one_feed(feed_url, label, window_hours, limit_per_feed)
            )
        # 按来源分组展示时保持顺序；可选按时间排序（这里简单按拉取顺序）
        return all_results
