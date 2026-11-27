"""Account ownership for a single-process OneBot deployment."""

import json
import os
import time
from pathlib import Path

from nonebot import get_bots, logger
from nonebot.adapters.onebot.v11 import Adapter, Bot, Event
from nonebot.exception import IgnoredException
from nonebot.message import event_preprocessor


class Accounts:
    def __init__(self, primary, standby, path):
        self.primary, self.standby, self.path = primary, standby, path
        self.enabled = bool(primary or standby)
        if self.enabled and (not primary.isdecimal() or not standby.isdecimal() or primary == standby):
            raise ValueError("Configure two different numeric ONEBOT_PRIMARY_QQ and ONEBOT_STANDBY_QQ")
        self.active = primary
        self.groups = set()
        self.blocked = set()
        self.cutover = 0
        self.pending = []
        self.down_since = None
        self.next_refresh = 0
        self.awaiting_preferred = False
        self.served = set()
        self.outage_reported = False
        if self.enabled and path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state["pair"] == [primary, standby]:
                self.active = state["active"]
                self.groups = set(state["groups"])
                self.blocked = set(state["blocked"])
                self.cutover = state["cutover"]
                self.pending = state["pending"]
                self.awaiting_preferred = state.get("awaiting_preferred", False)
                self.served = set(state.get("served", self.groups))
                self.outage_reported = state.get("outage_reported", False)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                dict(
                    pair=[self.primary, self.standby],
                    active=self.active,
                    groups=sorted(self.groups),
                    blocked=sorted(self.blocked),
                    cutover=self.cutover,
                    pending=self.pending,
                    awaiting_preferred=self.awaiting_preferred,
                    served=sorted(self.served),
                    outage_reported=self.outage_reported,
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def notice(self, message):
        logger.warning(message)
        self.pending.append(message)
        self.save()

    def forced_offline(self, account):
        if account in (self.primary, self.standby) and account not in self.blocked:
            self.blocked.add(account)
            self.notice(f"QQ {account} 收到平台强制下线通知，请检查该账号；已暂停它的群投递。")

    def owns(self, account, group=None):
        return not self.enabled or (
            account == self.active
            and account not in self.blocked
            and (not self.cutover or group is None or int(group) in self.served)
        )

    async def observe(self, bots, states, now):
        if not self.groups and not self.cutover and not self.blocked:
            if states.get(self.primary) is not True and states.get(self.standby) is True:
                groups = await memberships(bots[self.standby])
                if groups:
                    self.active = self.standby
                    self.groups = self.served = groups
                    self.awaiting_preferred = True
                    self.save()
        if states.get(self.active) is True and not self.cutover and now >= self.next_refresh:
            self.next_refresh = now + 60
            groups = await memberships(bots[self.active])
            if groups is not None and groups != self.groups:
                self.groups = self.served = groups
                self.save()
        other = self.standby if self.active == self.primary else self.primary
        prefer = self.awaiting_preferred and other == self.primary
        if states.get(self.active) is True or self.active in self.blocked:
            self.down_since = None
        elif self.active not in bots or states.get(self.active) is False:
            if self.down_since is None:
                self.down_since = now
        else:
            self.down_since = None
        due = self.down_since is not None and now - self.down_since >= 90
        if (due or prefer) and states.get(other) is True and not self.blocked:
            groups = await memberships(bots[other])
            if groups and not self.blocked and (common := self.groups & groups):
                previous = self.active
                self.active = other
                self.served = common
                self.cutover = int(time.time())
                self.down_since = None
                self.awaiting_preferred = False
                reason = "首选账号就绪" if prefer else f"{previous} 已持续离线 90 秒"
                self.notice(f"{reason}，QQ {other} 已接管群 " + "、".join(map(str, sorted(common))))
        unavailable = self.down_since is not None and now - self.down_since >= 90
        if unavailable and not self.outage_reported:
            self.outage_reported = True
            self.notice(f"QQ {self.active} 离线，暂时没有可接管的账号或共同群。")
        elif self.outage_reported and not unavailable:
            self.outage_reported = False
            self.save()
        if self.pending:
            sender = next(
                (
                    bots[i]
                    for i in (self.active, self.standby, self.primary)
                    if states.get(i) is True and i not in self.blocked
                ),
                None,
            )
            if sender is not None:
                message = self.pending.pop(0)
                self.save()
                try:
                    await sender.send_private_msg(user_id=int(os.environ["MASTER_QQ"]), message=message, _timeout=5)
                except Exception as exc:
                    logger.warning(f"主备提醒投递未确认: {type(exc).__name__}")


async def memberships(bot):
    try:
        items = await bot.get_group_list(no_cache=True, _timeout=4)
        return {int(item["group_id"]) for item in items}
    except Exception as exc:
        logger.warning(f"主备群列表读取失败: {type(exc).__name__}")
        return None


accounts = Accounts(
    os.getenv("ONEBOT_PRIMARY_QQ", ""),
    os.getenv("ONEBOT_STANDBY_QQ", ""),
    Path(os.getenv("LLM_CHAT_DATA_DIR", "data")) / "onebot-accounts.json",
)


def active_bots():
    bots = get_bots()
    return {k: v for k, v in bots.items() if accounts.owns(k)} if accounts.enabled else bots


@event_preprocessor
async def account_events(bot: Bot, event: Event):
    if not accounts.enabled:
        return
    if getattr(event, "notice_type", None) == "bot_offline":
        accounts.forced_offline(bot.self_id)
    group = getattr(event, "group_id", None)
    if group is not None and (
        not accounts.owns(bot.self_id, group)
        or getattr(event, "user_id", None) in (int(accounts.primary), int(accounts.standby))
        or event.time <= accounts.cutover
    ):
        raise IgnoredException("group event outside account ownership")


class AccountAdapter(Adapter):
    async def _call_api(self, bot, api, **data):
        group = data.get("group_id")
        if group is not None and not api.startswith(("get_", "can_")) and not accounts.owns(bot.self_id, group):
            raise RuntimeError("group action outside account ownership")
        return await super()._call_api(bot, api, **data)
