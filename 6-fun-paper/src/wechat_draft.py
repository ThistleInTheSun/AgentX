"""微信公众号草稿箱：Markdown 转 HTML 后写入 draft/add。"""
import html as html_mod
import json
import logging
import re

import requests

from . import config

log = logging.getLogger(__name__)
API = "https://api.weixin.qq.com/cgi-bin"


class WeChatError(RuntimeError):
    pass


def credentials_ready() -> bool:
    cfg = config.wechat_config()
    return bool(cfg["appid"] and cfg["secret"])


def _access_token() -> str:
    cfg = config.wechat_config()
    resp = requests.get(
        f"{API}/token",
        params={"grant_type": "client_credential", "appid": cfg["appid"], "secret": cfg["secret"]},
        timeout=config.HTTP_TIMEOUT,
    ).json()
    if "access_token" not in resp:
        raise WeChatError(f"获取 access_token 失败: {resp}")
    return resp["access_token"]


def upload_thumb(image_path: str) -> str:
    """上传永久封面图，返回 media_id（写入 .env 的 WECHAT_THUMB_MEDIA_ID）。"""
    token = _access_token()
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{API}/material/add_material",
            params={"access_token": token, "type": "image"},
            files={"media": f},
            timeout=60,
        ).json()
    if "media_id" not in resp:
        raise WeChatError(f"上传封面失败: {resp}")
    return resp["media_id"]


_INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _inline(text: str) -> str:
    return _INLINE_BOLD.sub(r"<strong>\1</strong>", html_mod.escape(text))


def markdown_to_html(md: str) -> str:
    """够用即可的转换器：只处理本项目产出的 Markdown 子集。"""
    out = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(_inline(lines[i].lstrip("> ").rstrip()))
                i += 1
            out.append("<blockquote>" + "<br/>".join(quote) + "</blockquote>")
            continue
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not set("".join(cells)) <= set("- :"):
                    rows.append(cells)
                i += 1
            trs = []
            for r_idx, row in enumerate(rows):
                tag = "th" if r_idx == 0 else "td"
                tds = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in row)
                trs.append(f"<tr>{tds}</tr>")
            out.append('<table border="1" cellpadding="4" style="border-collapse:collapse">'
                       + "".join(trs) + "</table>")
            continue
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:].rstrip())}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line == "---":
            out.append("<hr/>")
        else:
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "".join(out)


def create_draft(title: str, markdown: str, source_url: str, digest: str = "") -> str:
    """写入草稿箱，返回 media_id。"""
    cfg = config.wechat_config()
    thumb = config.get("WECHAT_THUMB_MEDIA_ID")
    if not thumb:
        raise WeChatError(
            "缺少 WECHAT_THUMB_MEDIA_ID（草稿必须有封面图）。"
            "先运行: python run.py --upload-thumb cover.jpg"
        )
    article = {
        "title": title[:60],
        "author": cfg["author"],
        "digest": digest[:110],
        "content": markdown_to_html(markdown),
        "content_source_url": source_url,
        "thumb_media_id": thumb,
        "need_open_comment": 0,
        "only_fans_can_comment": 0,
    }
    token = _access_token()
    resp = requests.post(
        f"{API}/draft/add",
        params={"access_token": token},
        data=json.dumps({"articles": [article]}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=60,
    ).json()
    if "media_id" not in resp:
        raise WeChatError(f"写入草稿箱失败: {resp}")
    log.info("已写入公众号草稿箱 media_id=%s", resp["media_id"])
    return resp["media_id"]


def publish(media_id: str) -> str:
    """把草稿发布到公众号主页（不推送给粉丝），返回 publish_id。"""
    token = _access_token()
    resp = requests.post(
        f"{API}/freepublish/submit",
        params={"access_token": token},
        json={"media_id": media_id},
        timeout=60,
    ).json()
    if resp.get("errcode") != 0:
        raise WeChatError(f"发布失败: {resp}")
    log.info("已提交发布 publish_id=%s", resp.get("publish_id"))
    return str(resp.get("publish_id", ""))
