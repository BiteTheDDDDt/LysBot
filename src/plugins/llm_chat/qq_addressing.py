"""QQ message addressing selected from trusted event metadata."""
from dataclasses import dataclass

from nonebot.adapters.onebot.v11 import Message, MessageSegment


@dataclass(frozen=True)
class MessageTarget:
    message_id: int | None
    user_id: int
    text: str


@dataclass(frozen=True)
class QQAddressing:
    reply_to: int | None = None
    mention: int | None = None

    def prefix(self) -> Message:
        message = Message()
        if self.reply_to is not None:
            message.append(MessageSegment.reply(self.reply_to))
        if self.mention is not None:
            message.append(MessageSegment.at(self.mention))
            message.append(MessageSegment.text(" "))
        return message

    def text(self, value: str, *, first: bool = True) -> Message | str:
        prefix = self.prefix() if first else Message()
        if prefix or "[CQ:" in value:
            return prefix + MessageSegment.text(value)
        return value


def select_addressing(payload: dict, candidates: list[MessageTarget], *, bot_self_id: str,
                      allow_reply: bool) -> QQAddressing:
    quoted = None
    if allow_reply:
        quoted = next((row for row in candidates if row.message_id is not None
                       and str(row.message_id) == str(payload.get("reply_to_message_id"))), None)
    mention = next((row.user_id for row in candidates if row.user_id > 0
                    and str(row.user_id) != bot_self_id
                    and str(row.user_id) == str(payload.get("mention_user_id"))), None)
    return QQAddressing(quoted.message_id if quoted else None,
                        None if quoted and quoted.user_id == mention else mention)
