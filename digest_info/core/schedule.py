"""
每类别推送时间：根据 schedule 配置判断当前是否该推送该类别。
支持：每日固定时刻、指定星期几（如每周五）、可选内容时间窗口 window（24h/7d）。
"""
from datetime import datetime, time, timedelta
from typing import Any


def _get_now(tz_name: str | None) -> datetime:
    """当前时间，若配置了时区则转为该时区。"""
    now = datetime.now()
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            return now.astimezone(ZoneInfo(tz_name))
        except Exception:
            pass
    return now


def _parse_time(s: str) -> time | None:
    """解析 "HH:MM" 或 "H:MM"。"""
    if not s or ":" not in s:
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return time(h, m)
    except ValueError:
        pass
    return None


def is_due(
    schedule: dict[str, Any] | None,
    now: datetime | None = None,
    timezone: str | None = None,
    *,
    minute_window: int = 5,
) -> bool:
    """
    判断当前是否该推送该类别。
    - schedule 为 None 或空：视为不限制，始终 due（方便未配置时兼容）。
    - schedule.time: "09:15" 表示每天 9:15 左右推送。
    - schedule.weekday: 0=周一 … 6=周日，仅该星期几推送；与 time 同时生效。
    - schedule.days: [0,1,…,6] 同 weekday，可多日。
    - minute_window: 与 schedule 时间相差在此分钟内视为 due（应对 cron 略漂移）。
    """
    if not schedule:
        return True
    now = now or _get_now(timezone)
    t = _parse_time(schedule.get("time") or "")
    if t is None:
        # 未配置具体时间则只按星期几判断；若星期几也未配置则视为 due
        weekday = now.weekday()  # Monday=0, Sunday=6
        days = schedule.get("days")
        w = schedule.get("weekday")
        if days is not None:
            return weekday in list(days)
        if w is not None:
            return weekday == int(w)
        return True
    # 今天 schedule 时刻
    scheduled_today = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    start = scheduled_today - timedelta(minutes=minute_window)
    end = scheduled_today + timedelta(minutes=minute_window)
    if not (start <= now <= end):
        return False
    weekday = now.weekday()
    days = schedule.get("days")
    w = schedule.get("weekday")
    if days is not None:
        return weekday in list(days)
    if w is not None:
        return weekday == int(w)
    return True


def get_window_hours(schedule: dict[str, Any] | None) -> int | None:
    """
    从 schedule.window 解析「内容时间窗口」小时数，供拉取时过滤「最近 N 小时」等。
    - "24h" -> 24, "7d" -> 168
    """
    if not schedule:
        return None
    w = (schedule.get("window") or "").strip().lower()
    if not w:
        return None
    if w.endswith("h"):
        try:
            return int(w[:-1])
        except ValueError:
            return None
    if w.endswith("d"):
        try:
            return int(w[:-1]) * 24
        except ValueError:
            return None
    return None
