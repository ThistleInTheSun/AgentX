"""
微信推送：企业微信机器人 Webhook。
创建方式：企业微信群 -> 群设置 -> 群机器人 -> 添加 -> 复制 Webhook URL。
"""
import requests
from ..core.notifier import Notifier, NotifierRegistry


@NotifierRegistry.register("wechat")
class WeChatNotifier(Notifier):
    """企业微信机器人 Webhook 推送。"""
    notifier_id = "wechat"

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = (webhook_url or "").rstrip()

    def send(self, title: str, body: str, **kwargs) -> bool:
        if not self.webhook_url or "qyapi.weixin.qq.com" not in self.webhook_url:
            return False
        # 企业微信 markdown 消息有长度限制，过长则截断
        max_len = kwargs.get("max_body_len", 2000)
        if len(body) > max_len:
            body = body[:max_len] + "\n...(内容已截断)"
        content = f"## {title}\n\n{body}"
        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            },
        }
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            ok = r.status_code == 200
            if ok:
                j = r.json()
                if j.get("errcode") != 0:
                    return False
            return ok
        except Exception:
            return False
