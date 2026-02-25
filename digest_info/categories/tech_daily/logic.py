"""
每日科技摘要：过去 24 小时内科技圈重大新闻 + arXiv CS 新论文。
第一步：从 TechCrunch、The Verge、Ars Technica、arXiv CS 搜集信息；
第二步：按 window_hours 过滤并合并；
第三步：由 summarizer（含可选大模型）按 prompt 总结并推送。
"""
from digest_info.core.category import Category, CategoryRegistry
from digest_info.core.source import SearchResult
from digest_info.sources.tech_news_feeds import TechNewsFeedsSource
from digest_info.sources.arxiv_source import ArxivSource


@CategoryRegistry.register("tech_daily", display_name="每日科技摘要")
class TechDailyCategory(Category):
    """科技新闻（多 RSS）+ arXiv CS 新论文，每日 9:15 推送过去 24 小时内容。"""

    def __init__(
        self,
        limit_news: int = 15,
        limit_papers: int = 10,
        feed_urls: list | None = None,
        arxiv_categories: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            limit_news=limit_news,
            limit_papers=limit_papers,
            feed_urls=feed_urls,
            arxiv_categories=arxiv_categories,
            **kwargs,
        )
        self._news_source = TechNewsFeedsSource(
            feed_urls=feed_urls,
            limit_per_feed=limit_news // 3 + 2,
        )
        self._arxiv_source = ArxivSource(
            categories=arxiv_categories or ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
            limit=limit_papers,
        )

    def fetch(self, **kwargs) -> list[SearchResult]:
        window_hours = kwargs.get("window_hours", 24)
        limit_news = kwargs.get("limit_news", 15)
        limit_papers = kwargs.get("limit_papers", 10)

        news = self._news_source.fetch(
            window_hours=window_hours,
            limit_per_feed=limit_news // 3 + 2,
        )
        papers = self._arxiv_source.fetch(
            window_hours=window_hours,
            limit=limit_papers,
        )
        # 先新闻，后论文；若需按时间混排可在此扩展
        return news + papers
