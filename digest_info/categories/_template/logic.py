"""
【模板】新信息类别逻辑 — 复制本目录并重命名后使用。

使用步骤：
1. 复制整个 _template 目录，重命名为你的类别 id（如 cocoon_break、tech_news）
2. 在本文件中：把 MyCategory / my_category 替换为你的类名和 id，实现 fetch()
3. 编辑同目录下的 prompt.txt，写该类别的展示/筛选说明
4. 在 categories/__init__.py 中增加: from . import 你的目录名
5. 在 config.yaml 的 categories 里添加一项，并设置 schedule（推送时间）和 params
"""
from digest_info.core.category import Category, CategoryRegistry
from digest_info.core.source import SearchResult


@CategoryRegistry.register("my_category", display_name="我的类别")
class MyCategory(Category):
    """类别说明：例如「信息茧房破圈：少见的、改变认知的、对生活有价值的信息」"""

    def __init__(self, limit: int = 10, **kwargs):
        super().__init__(limit=limit, **kwargs)
        self._limit = limit
        # 若有现成 Source 可复用，例如：
        # from digest_info.sources.rss_source import RSSSource
        # self._source = RSSSource(feed_url=..., limit=limit)

    def fetch(self, **kwargs) -> list[SearchResult]:
        limit = kwargs.get("limit", self._limit)
        # TODO: 实现拉取逻辑，返回 list[SearchResult]
        # 若配置了 window_hours（最近 N 小时），可从 kwargs 取: kwargs.get("window_hours")
        return []
