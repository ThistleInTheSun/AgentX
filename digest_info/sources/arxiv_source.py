"""
arXiv 论文源：按提交时间范围拉取 CS 相关新论文（如 cs.AI, cs.LG, cs.CL, cs.CV）。
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import requests
from ..core.source import Source, SearchResult, SourceRegistry

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def _parse_arxiv_feed(xml_content: bytes, source_id: str) -> list[SearchResult]:
    results = []
    root = ET.fromstring(xml_content)
    for entry in root.findall(f".//{{{ATOM_NS}}}entry"):
        title_el = entry.find(f"{{{ATOM_NS}}}title")
        summary_el = entry.find(f"{{{ATOM_NS}}}summary")
        link_el = entry.find(f"{{{ATOM_NS}}}link[@rel='alternate']")
        if link_el is None:
            link_el = entry.find(f"{{{ATOM_NS}}}link")
        id_el = entry.find(f"{{{ATOM_NS}}}id")
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        link = link_el.get("href", "") if link_el is not None else ""
        if not link and id_el is not None and id_el.text:
            link = id_el.text.replace("/abs/", "/abs/").strip()
        if not title:
            continue
        results.append(
            SearchResult(
                title=title,
                url=link,
                summary=summary[:300] if summary else "",
                source_id=source_id,
                raw={"summary_full": summary},
            )
        )
    return results


@SourceRegistry.register("arxiv")
class ArxivSource(Source):
    """arXiv API：按 submittedDate 范围拉取 CS 分类新论文。"""

    source_id = "arxiv"

    def __init__(
        self,
        categories: str | list[str] | None = None,
        limit: int = 20,
    ):
        # 默认 CS 常见子类：AI, 机器学习, 计算与语言, 计算机视觉
        if categories is None:
            categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
        self.categories = [categories] if isinstance(categories, str) else list(categories)
        self.limit = limit

    def fetch(self, window_hours: int | None = None, limit: int | None = None, **kwargs) -> list[SearchResult]:
        limit = limit or self.limit
        now = datetime.now(timezone.utc)
        if window_hours is None:
            window_hours = 24
        start_dt = now - timedelta(hours=window_hours)
        # arXiv API 要求 GMT，格式 YYYYMMDDHHmm
        start_str = start_dt.strftime("%Y%m%d%H%M")
        end_str = now.strftime("%Y%m%d%H%M")
        # 构建 cat 查询：cat:cs.AI OR cat:cs.LG ...
        cat_query = "+OR+".join(f"cat:{c}" for c in self.categories)
        date_query = f"submittedDate:[{start_str}+TO+{end_str}]"
        search_query = f"({cat_query})+AND+{date_query}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = "http://export.arxiv.org/api/query"
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "InfoDigest/1.0"})
            r.raise_for_status()
            results = _parse_arxiv_feed(r.content, self.source_id)
            return results[:limit]
        except Exception as e:
            return [
                SearchResult(
                    title="arXiv 获取失败",
                    url=url,
                    summary=str(e),
                    source_id=self.source_id,
                )
            ]
