"""QQ message addressing from events and verified group membership."""
from dataclasses import dataclass, field
import json
import re
import unicodedata

from nonebot.adapters.onebot.v11 import Message, MessageSegment


MENTION_SLOT = re.compile(r"⟦at[0-9]+⟧")


@dataclass(frozen=True)
class MessageTarget:
    message_id: int | None
    user_id: int
    text: str
    names: tuple[str, ...] = ()


@dataclass
class QQAddressing:
    reply_to: int | None = None
    mention: int | None = None
    additional_mentions: tuple[int, ...] = ()
    audience: tuple[int, ...] = ()
    candidates: tuple[MessageTarget, ...] = field(default=(), repr=False, compare=False)
    bot_self_id: str = field(default="", repr=False, compare=False)
    allow_reply: bool = field(default=True, repr=False, compare=False)

    @property
    def mentions(self) -> tuple[int, ...]:
        return ((self.mention,) if self.mention is not None else ()) + self.additional_mentions

    def as_payload(self) -> dict:
        return {"audience_user_ids": list(self.audience),
                "reply_to_message_id": self.reply_to, "mention_user_ids": list(self.mentions)}

    def system_hint(self) -> str:
        return ("【本轮说话对象与送达】\n" + json.dumps(self.as_payload(), ensure_ascii=False)
                + "\n audience_user_ids 是正文主要对谁说，空列表表示面向当前话题或群体。"
                "引用用于承接原话，@用于叫到人，两者不等于正文的全部受众。"
                "把任务交给某人时直接对他说；评论某人时可以说他。程序添加引用和@，正文无需拼QQ标签。")

    def mention_slots(self) -> dict[str, int]:
        return {f"⟦at{index}⟧": user_id for index, user_id in enumerate(self.mentions, 1)}

    def _display_names(self) -> dict[int, str]:
        return {row.user_id: row.names[0] for row in self.candidates if row.names and row.names[0]}

    def rendering_hint(self, *, allow_inline: bool = True) -> str:
        """Describe who hears the prose without passing QQ routing fields to chat."""
        hint = "写群聊正文，从要说的话开始；系统会添加规划好的引用和 @。"
        if self.audience:
            names = self._display_names()
            audience = [names[user_id] for user_id in self.audience if user_id in names]
            if audience:
                hint += "已知收话人的称呼（资料）：" + json.dumps(audience, ensure_ascii=False) + "。"
            hint += "直接对交接里指定的收话人说话。"
        else:
            hint += "这句面向群里正在聊的话题；QQ 引用只标记接的是哪句。"
        if self.mentions and allow_inline:
            names = self._display_names()
            slots = {marker: names.get(user_id, f"第{index}位待通知的对象")
                     for index, (marker, user_id) in enumerate(self.mention_slots().items(), 1)}
            hint += ("下面的占位符对应规划好的原生 @。直接把占位符写在你想 @ 对方的位置，"
                     "例如把‘让某人来讲’写成‘那就让⟦at1⟧来讲’。同一占位符用一次，省略时系统会在开头 @。"
                     "保留占位符原样，无需另写 @昵称。可用占位符（资料）："
                     + json.dumps(slots, ensure_ascii=False))
        return hint

    def add_verified_member(self, user_id: int, *names: str) -> None:
        names = tuple(name for name in names if name)
        target = MessageTarget(None, user_id, names[0] if names else "群成员", names)
        if str(user_id) != self.bot_self_id and target not in self.candidates:
            self.candidates += (target,)

    def strip_repeated_mentions(self, text: str) -> str:
        """Drop leading textual @s that duplicate native mentions in this turn."""
        if not self.mentions or not (text or "").lstrip().startswith(("@", "＠")):
            return text
        names = {str(user_id) for user_id in self.mentions}
        for target in self.candidates:
            if target.user_id not in self.mentions:
                continue
            for name in target.names:
                alias = name.strip()
                names.add(alias)
                # The renderer sometimes leaves off a decorative final emoji.
                while alias and unicodedata.category(alias[-1]) in {"So", "Mn", "Sk", "Cf"}:
                    alias = alias[:-1].rstrip()
                names.add(alias)
        alternatives = "|".join(re.escape(name) for name in sorted(names - {""}, key=lambda name: (-len(name), name)))
        prefix = re.compile(rf"^\s*[@＠](?:{alternatives})(?=$|[\s,，:：、])[\s,，:：、]*", re.IGNORECASE)
        body = text
        while match := prefix.match(body):
            body = body[match.end():]
        return body if body.strip() else text

    def revise(self, payload: dict) -> None:
        """Refine this turn's shared delivery object before any media is sent."""
        if not isinstance(payload, dict):
            return
        merged = {**self.as_payload(), **payload}
        updated = select_addressing(merged, list(self.candidates),
                                    bot_self_id=self.bot_self_id, allow_reply=self.allow_reply)
        self.reply_to, self.mention = updated.reply_to, updated.mention
        self.additional_mentions, self.audience = updated.additional_mentions, updated.audience

    def prefix(self, *, skip_mentions: frozenset[int] = frozenset()) -> Message:
        message = Message()
        if self.reply_to is not None:
            message.append(MessageSegment.reply(self.reply_to))
        for user_id in self.mentions:
            if user_id in skip_mentions:
                continue
            message.append(MessageSegment.at(user_id))
            message.append(MessageSegment.text(" "))
        return message

    def text(self, value: str, *, first: bool = True) -> Message | str:
        prefix = self.prefix() if first else Message()
        if prefix or "[CQ:" in value:
            return prefix + MessageSegment.text(value)
        return value


    def place_text(self, value: str, *, first: bool, reserved: frozenset[int],
                   mentioned: set[int]) -> tuple[str, Message | str]:
        """Resolve approved markers into QQ segments and keep readable text for memory."""
        slots = self.mention_slots()
        if not slots or not MENTION_SLOT.search(value):
            prefix = self.prefix(skip_mentions=reserved) if first else Message()
            return value, prefix + MessageSegment.text(value) if prefix or "[CQ:" in value else value

        message = self.prefix(skip_mentions=reserved) if first else Message()
        names = self._display_names()
        plain = []
        offset = 0
        for match in MENTION_SLOT.finditer(value):
            before = value[offset:match.start()]
            if before:
                message.append(MessageSegment.text(before))
                plain.append(before)
            user_id = slots.get(match[0])
            if user_id is not None:
                if user_id in mentioned:
                    name = names.get(user_id, "对方")
                    message.append(MessageSegment.text(name))
                    plain.append(name)
                else:
                    message.append(MessageSegment.at(user_id))
                    mentioned.add(user_id)
                    plain.append("@" + names.get(user_id, str(user_id)))
            offset = match.end()
        if offset < len(value):
            message.append(MessageSegment.text(value[offset:]))
            plain.append(value[offset:])
        return "".join(plain), message


def select_addressing(payload: dict, candidates: list[MessageTarget], *, bot_self_id: str,
                      allow_reply: bool) -> QQAddressing:
    quoted = None
    if allow_reply:
        quoted = next((row for row in candidates if row.message_id is not None
                       and str(row.message_id) == str(payload.get("reply_to_message_id"))), None)
    def user_ids(value):
        if not isinstance(value, list):
            return ()
        allowed = {str(row.user_id): row.user_id for row in candidates
                   if row.user_id > 0 and str(row.user_id) != bot_self_id}
        return tuple(dict.fromkeys(allowed[str(item)] for item in value if str(item) in allowed))

    mentions = user_ids(payload.get("mention_user_ids", [payload.get("mention_user_id")]))
    mentions = tuple(user for user in mentions if not quoted or quoted.user_id != user)
    return QQAddressing(reply_to=quoted.message_id if quoted else None,
                        mention=mentions[0] if mentions else None,
                        additional_mentions=mentions[1:], audience=user_ids(payload.get("audience_user_ids", [])),
                        candidates=tuple(candidates), bot_self_id=bot_self_id, allow_reply=allow_reply)
