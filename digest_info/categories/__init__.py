# 信息类别：每个子目录为一类，含独立逻辑 + prompt.txt，便于定制
from . import hackernews
from . import rss
from . import tech_daily

__all__ = ["hackernews", "rss", "tech_daily"]
