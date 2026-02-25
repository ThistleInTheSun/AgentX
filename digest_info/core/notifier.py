"""
推送渠道抽象：把摘要发到哪里（微信、邮件等），便于扩展新渠道。
"""
from abc import ABC, abstractmethod


class Notifier(ABC):
    """推送渠道抽象基类。"""

    notifier_id: str = ""

    @abstractmethod
    def send(self, title: str, body: str, **kwargs) -> bool:
        """发送消息。返回是否成功。"""
        pass

    def get_display_name(self) -> str:
        return self.notifier_id or self.__class__.__name__


class NotifierRegistry:
    """推送渠道注册表。"""
    _notifiers: dict[str, type[Notifier]] = {}

    @classmethod
    def register(cls, notifier_id: str):
        def decorator(klass: type[Notifier]):
            klass.notifier_id = notifier_id
            cls._notifiers[notifier_id] = klass
            return klass
        return decorator

    @classmethod
    def get(cls, notifier_id: str, **init_kwargs) -> Notifier | None:
        K = cls._notifiers.get(notifier_id)
        if K is None:
            return None
        return K(**init_kwargs)

    @classmethod
    def list_ids(cls) -> list[str]:
        return list(cls._notifiers.keys())
