"""
示例搜索源：Hacker News 热门/最新。
"""
import requests
from ..core.source import Source, SearchResult, SourceRegistry


@SourceRegistry.register("hackernews")
class HackerNewsSource(Source):
    """HN Top 或 New 列表。"""
    source_id = "hackernews"

    def __init__(self, feed: str = "top", limit: int = 10):
        # feed: top | new | best
        self.feed = feed
        self.limit = limit

    def fetch(self, **kwargs) -> list[SearchResult]:
        feed = kwargs.get("feed", self.feed)
        limit = kwargs.get("limit", self.limit)
        url = f"https://hacker-news.firebaseio.com/v0/{feed}stories.json"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            ids = r.json()[:limit]
        except Exception as e:
            return [SearchResult(
                title="Hacker News 获取失败",
                url="",
                summary=str(e),
                source_id=self.source_id,
            )]
        results = []
        for i, sid in enumerate(ids):
            try:
                item_r = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{sid}.json",
                    timeout=5,
                )
                item_r.raise_for_status()
                item = item_r.json()
                title = item.get("title", "(无标题)")
                link = item.get("url") or f"https://news.ycombinator.com/item?id={sid}"
                score = item.get("score", 0)
                results.append(SearchResult(
                    title=title,
                    url=link,
                    summary=f"Score: {score}",
                    source_id=self.source_id,
                    raw=item,
                ))
            except Exception:
                continue
        return results
