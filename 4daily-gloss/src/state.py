"""状态持久化：JSON 记录已处理文章 URL，避免重复。"""
import json
from datetime import datetime, timezone

from . import config


def load() -> dict:
    if config.STATE_FILE.exists():
        return json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
    return {"processed": {}}


def is_processed(state: dict, url: str) -> bool:
    return url in state["processed"]


def mark_processed(state: dict, url: str, title: str, status: str) -> None:
    state["processed"][url] = {
        "title": title,
        "status": status,  # drafted / published_to_wechat / skipped
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
