"""
arXiv 论文源：按提交时间范围拉取 CS 相关新论文（如 cs.AI, cs.LG, cs.CL, cs.CV）。
"""
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
import requests
from ..core.source import Source, SearchResult, SourceRegistry

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def _parse_updated(updated_el) -> datetime | None:
    """从 Atom <updated> 解析为 UTC datetime，如 2026-02-26T02:01:29Z"""
    if updated_el is None or not (updated_el.text or "").strip():
        return None
    try:
        # ISO 格式带 Z 或 +00:00，统一为 UTC
        s = (updated_el.text or "").strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


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
        updated_el = entry.find(f"{{{ATOM_NS}}}updated")
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        link = link_el.get("href", "") if link_el is not None else ""
        if not link and id_el is not None and id_el.text:
            link = id_el.text.replace("/abs/", "/abs/").strip()
        if not title or title.strip().lower() == "error":
            continue
        updated_dt = _parse_updated(updated_el)
        raw = {"summary_full": summary}
        if updated_dt is not None:
            raw["updated_parsed"] = updated_dt
        results.append(
            SearchResult(
                title=title,
                url=link,
                summary=summary[:300] if summary else "",
                source_id=source_id,
                raw=raw,
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
            # 过去 N 小时：不用 submittedDate 查询（该条件常返回 0），改为拉「按时间排序的最近多条」再在本地按 <updated> 过滤
            if window_hours is not None and window_hours > 0:
                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(hours=window_hours)
                seen_urls = set()
                results = []
                # 每类多拉一些，以便过滤后仍能凑够 limit（一周内每类可能有几十篇）
                fetch_per_cat = min(100, max(limit * 2, 50))
                for i, c in enumerate(self.categories):
                    if i > 0:
                        time.sleep(3)  # arXiv 建议连续请求间隔 3 秒
                    search_query = f"cat:{c}"
                    batch = self._query_arxiv(url, search_query, fetch_per_cat)
                    for r in batch:
                        if r.url in seen_urls:
                            continue
                        updated = r.raw.get("updated_parsed")
                        if updated is not None and updated < cutoff:
                            continue  # 超出时间窗口，跳过
                        seen_urls.add(r.url)
                        results.append(r)
                    if len(results) >= limit:
                        break
                # 按更新时间倒序（API 已是倒序，合并后保持即可），取前 limit
                results = results[:limit]
            else:
                results = []
            # 时间窗口内 0 条时回退：只按分类取最近 N 条（arXiv 建议连续请求间隔 3 秒）
            if not results:
                time.sleep(3)
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
        # 整段编码：+ -> %2B，( ) -> %28 %29，否则服务端收不到 search_query 或把 + 当空格
        qs = f"search_query={quote(search_query, safe='')}&start=0&max_results={min(limit, 100)}&sortBy=submittedDate&sortOrder=descending"
        r = requests.get(
            f"{url}?{qs}",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; InfoDigest/1.0)"},
        )
        r.raise_for_status()
        return _parse_arxiv_feed(r.content, self.source_id)
