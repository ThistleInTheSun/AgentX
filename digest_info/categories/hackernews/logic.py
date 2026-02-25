"""
Hacker News 信息类别：独立拉取逻辑 + 本目录下的 prompt.txt 定制展示方式。
"""
from digest_info.core.category import Category, CategoryRegistry
from digest_info.core.source import SearchResult
from digest_info.sources.hackernews import HackerNewsSource


@CategoryRegistry.register("hackernews", display_name="Hacker News 热门")
class HackerNewsCategory(Category):
    """HN 热门/最新/最佳，逻辑在 sources.hackernews，prompt 见同目录 prompt.txt。"""

    def __init__(self, feed: str = "top", limit: int = 10, **kwargs):
        super().__init__(feed=feed, limit=limit, **kwargs)
        self._source = HackerNewsSource(feed=feed, limit=limit)

    def fetch(self, **kwargs) -> list[SearchResult]:
        feed = kwargs.get("feed", self._source.feed)
        limit = kwargs.get("limit", self._source.limit)
        return self._source.fetch(feed=feed, limit=limit)
