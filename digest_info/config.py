"""
配置加载：主 config.yaml + 每个类别各自目录下的 config.yaml。
"""
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "config.yaml"
CATEGORIES_DIR = CONFIG_DIR / "categories"


def load_config() -> dict:
    """加载主 config.yaml；若不存在则尝试 config.example.yaml 或返回默认。"""
    if CONFIG_PATH.exists():
        return _load_yaml(CONFIG_PATH)
    example = CONFIG_DIR / "config.example.yaml"
    if example.exists():
        return _load_yaml(example)
    return _default_config()


def load_category_config(category_id: str) -> dict:
    """
    加载该类别自己的 config.yaml（categories/<id>/config.yaml）。
    返回至少包含 params、schedule 的 dict；文件不存在或为空则返回 {}。
    """
    path = CATEGORIES_DIR / category_id / "config.yaml"
    if not path.exists():
        return {}
    data = _load_yaml(path)
    if not isinstance(data, dict):
        return {}
    return data


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return _default_config()


def _default_config() -> dict:
    return {
        "timezone": "Asia/Shanghai",
        "categories": ["hackernews"],  # 启用的类别 id，参数从各类别目录 config.yaml 读
        "summarizer": "plain",
        "notifiers": [
            {
                "id": "wechat",
                "params": {
                    "webhook_url": os.environ.get("WECHAT_WEBHOOK_URL", ""),
                },
            },
        ],
    }
