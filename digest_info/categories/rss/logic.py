"""
RSS 信息类别：通用 RSS，需在配置中提供 feed_url；展示方式由本目录 prompt.txt 定制。
"""
from digest_info.core.category import Category, CategoryRegistry
from digest_info.core.source import SearchResult
from digest_info.sources.rss_source import RSSSource


@CategoryRegistry.register("rss", display_name="RSS 订阅")
class RSSCategory(Category):
    """任意 RSS 地址，params 需含 feed_url，可选 limit。"""

    def __init__(self, feed_url: str = "", limit: int = 10, **kwargs):
        super().__init__(feed_url=feed_url, limit=limit, **kwargs)
        self._source = RSSSource(feed_url=feed_url or "", limit=limit)

    def fetch(self, **kwargs) -> list[SearchResult]:
        url = kwargs.get("feed_url", self._source.feed_url)
        limit = kwargs.get("limit", self._source.limit)
        if not url:
            return []
        return self._source.fetch(feed_url=url, limit=limit)
