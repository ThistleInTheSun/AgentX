"""加载 .env 与全局配置。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS_DIR = ROOT / "drafts"
LOGS_DIR = ROOT / "logs"
STATE_FILE = ROOT / "state.json"

# 跨学科、容易产生「好玩发现」的 arXiv 分类
ARXIV_CATEGORIES = [
    "q-bio",           # 定量生物学
    "physics.pop-ph",  # 大众物理
    "cs.CY",           # 计算机与社会
    "cs.HC",           # 人机交互
    "cs.SD",           # 语音
    "astro-ph.EP",     # 地球与行星天体物理
    "stat.AP",         # 统计应用
    "eess.SY",         # 系统与控制
]
ARXIV_MAX_RESULTS = 25

# 摘要筛选阈值
MIN_ABSTRACT_CHARS = 300
MAX_ABSTRACT_CHARS = 4000
MIN_ASCII_RATIO = 0.85

HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (fun-paper-explainer; research digest)"


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
        "author": get("WECHAT_AUTHOR", "好玩论文挖掘机"),
    }
