"""
信息类别抽象：每个类别 = 独立拉取逻辑 + 专属 prompt，便于按类别定制。
"""
from abc import ABC, abstractmethod
from pathlib import Path
from .source import Source, SearchResult


class Category(ABC):
    """
    一个信息类别：负责该类别的数据拉取 + 提供该类别的总结/展示 prompt。
    新增类别时在 categories/<name>/ 下实现逻辑并放 prompt.txt，再注册即可。
    """
    category_id: str = ""
    display_name: str = ""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @abstractmethod
    def fetch(self, **kwargs) -> list[SearchResult]:
        """拉取该类别下的信息。"""
        pass

    def get_prompt(self) -> str:
        """
        返回该类别用于总结/展示的 prompt 文本。
        默认从同目录下的 prompt.txt 读取；子类可重写。
        """
        return _read_prompt_for_class(self.__class__)

    def get_display_name(self) -> str:
        return self.display_name or self.category_id or self.__class__.__name__


def _read_prompt_for_class(klass: type) -> str:
    """从类别所在目录读取 prompt.txt（如 digest_info/categories/hackernews/prompt.txt）。"""
    try:
        mod = klass.__module__
        if "categories." in mod:
            parts = mod.split(".")
            idx = parts.index("categories")
            if idx + 1 < len(parts):
                # __file__ = digest_info/core/category.py -> parent.parent = digest_info
                base = Path(__file__).resolve().parent.parent
                cat_dir = base / "categories" / parts[idx + 1]
                prompt_file = cat_dir / "prompt.txt"
                if prompt_file.exists():
                    return prompt_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


class CategoryRegistry:
    """信息类别注册表：按 category_id 查找。"""
    _categories: dict[str, type[Category]] = {}

    @classmethod
    def register(cls, category_id: str, display_name: str = ""):
        def decorator(klass: type[Category]):
            klass.category_id = category_id
            klass.display_name = display_name or category_id
            cls._categories[category_id] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, category_id: str, **init_kwargs) -> Category | None:
        K = cls._categories.get(category_id)
        if K is None:
            return None
        return K(**init_kwargs)

    @classmethod
    def list_ids(cls) -> list[str]:
        return list(cls._categories.keys())
