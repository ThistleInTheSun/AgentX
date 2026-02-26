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
        if not title or title.strip().lower() == "error":
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


def fetch_one_paper_by_id(paper_id: str) -> list[SearchResult]:
    """
    按 arXiv 编号拉取单篇论文（用于测试）。例如 paper_id="1706.03762"。
    返回与 _parse_arxiv_feed 相同格式的列表，失败或解析不到则返回 []。
    """
    url = "http://export.arxiv.org/api/query"
    try:
        r = requests.get(
            url,
            params={"id_list": paper_id},
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; InfoDigest/1.0)"},
        )
        r.raise_for_status()
        return _parse_arxiv_feed(r.content, "arXiv")
    except Exception:
        return []


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
        url = "http://export.arxiv.org/api/query"
        # 分类条件：cat:cs.AI+OR+cat:cs.LG+...（不加括号，兼容 API）
        cat_query = "+OR+".join(f"cat:{c}" for c in self.categories)
        try:
            # 先尝试带时间范围（arXiv 支持 submittedDate:[YYYYMMDD TO YYYYMMDD]）
            if window_hours is not None and window_hours > 0:
                now = datetime.now(timezone.utc)
                start_dt = now - timedelta(hours=window_hours)
                start_str = start_dt.strftime("%Y%m%d")
                end_str = now.strftime("%Y%m%d")
                date_query = f"submittedDate:[{start_str}+TO+{end_str}]"
                search_query = f"{cat_query}+AND+{date_query}"
                results = self._query_arxiv(url, search_query, limit)
            else:
                results = []
            # 时间窗口内 0 条时回退：只按分类取最近 N 条
            if not results:
                results = self._query_arxiv(url, cat_query, limit)
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

    def _query_arxiv(self, url: str, search_query: str, limit: int) -> list[SearchResult]:
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; InfoDigest/1.0)"})
        r.raise_for_status()
        return _parse_arxiv_feed(r.content, self.source_id)
