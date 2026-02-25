"""
搜索源抽象：所有「从哪里搜」的实现都实现此接口，便于扩展新内容。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """单条搜索结果"""
    title: str
    url: str
    summary: str = ""
    source_id: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "source_id": self.source_id,
        }


class Source(ABC):
    """搜索源抽象基类。新增搜索内容时，继承此类并注册即可。"""

    source_id: str = ""  # 唯一标识，用于配置和日志

    @abstractmethod
    def fetch(self, **kwargs) -> list[SearchResult]:
        """拉取/搜索信息，返回结果列表。kwargs 可传该源需要的参数（如关键词、条数）。"""
        pass

    def get_display_name(self) -> str:
        return self.source_id or self.__class__.__name__


class SourceRegistry:
    """搜索源注册表：按 source_id 查找，方便配置里只写字符串即可启用某几个源。"""
    _sources: dict[str, type[Source]] = {}

    @classmethod
    def register(cls, source_id: str):
        def decorator(klass: type[Source]):
            klass.source_id = source_id
            cls._sources[source_id] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, source_id: str, **init_kwargs) -> Source | None:
        K = cls._sources.get(source_id)
        if K is None:
            return None
        return K(**init_kwargs)

    @classmethod
    def list_ids(cls) -> list[str]:
        return list(cls._sources.keys())
