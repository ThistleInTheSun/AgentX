"""
总结器抽象：把多条 SearchResult 整理成一段可读摘要，便于扩展（简单拼接 / LLM 总结等）。
"""
from abc import ABC, abstractmethod
from .source import SearchResult


class Summarizer(ABC):
    """总结器抽象基类。"""

    summarizer_id: str = ""

    @abstractmethod
    def summarize(
        self,
        results: list[SearchResult],
        source_id: str = "",
        category_id: str = "",
        prompt: str = "",
        **kwargs,
    ) -> str:
        """将多条搜索结果总结成一段文本。category_id / prompt 为该类别专属，便于定制。"""
        pass

    def get_display_name(self) -> str:
        return self.summarizer_id or self.__class__.__name__


class SummarizerRegistry:
    """总结器注册表。"""
    _summarizers: dict[str, type[Summarizer]] = {}

    @classmethod
    def register(cls, summarizer_id: str):
        def decorator(klass: type[Summarizer]):
            klass.summarizer_id = summarizer_id
            cls._summarizers[summarizer_id] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, summarizer_id: str, **init_kwargs) -> Summarizer | None:
        K = cls._summarizers.get(summarizer_id)
        if K is None:
            return None
        return K(**init_kwargs)

    @classmethod
    def list_ids(cls) -> list[str]:
        return list(cls._summarizers.keys())
