"""加载 .env 与全局配置。不引入 python-dotenv，自行解析。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS_DIR = ROOT / "drafts"
LOGS_DIR = ROOT / "logs"
STATE_FILE = ROOT / "state.json"

FEED_URL = "https://ourworldindata.org/atom.xml"

# 文章筛选阈值
MIN_ARTICLE_CHARS = 1200      # 正文太短跳过（多为公告类）
MAX_TRANSLATE_CHARS = 6000    # 超长文章截取前若干段（节选）
MIN_ASCII_RATIO = 0.9         # 英文判定

HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (daily-gloss; +https://ourworldindata.org)"


def load_env() -> None:
    """读取项目根目录 .env，写入 os.environ（不覆盖已存在的变量）。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def llm_config() -> dict:
    return {
        "api_key": get("LLM_API_KEY"),
        "base_url": get("LLM_BASE_URL", "https://api.deepseek.com"),
        "model": get("LLM_MODEL", "deepseek-chat"),
    }


def wechat_config() -> dict:
    return {
        "appid": get("WECHAT_APPID"),
        "secret": get("WECHAT_APPSECRET"),
        "author": get("WECHAT_AUTHOR", "每日外刊精读"),
    }
