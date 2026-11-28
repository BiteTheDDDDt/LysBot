"""Account ownership for a single-process OneBot deployment."""

import asyncio
import json
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nonebot import get_bots, logger
from nonebot.adapters.onebot.v11 import Adapter, Bot, Event
from nonebot.exception import IgnoredException
from nonebot.message import event_preprocessor


_sending_account_alert = ContextVar("sending_account_alert", default=False)


class AccountStatus(StrEnum):
    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


@dataclass(frozen=True, eq=False)
class AccountSession:
    bot: Bot | None
    status: AccountStatus


async def probe_status(bot):
    try:
        async with asyncio.timeout(3):
            result = await bot.get_status()
    except Exception:
        return AccountStatus.UNKNOWN
    if isinstance(result, dict):
        if result.get("online") is False:
            return AccountStatus.OFFLINE
        if result.get("online") is True:
            return AccountStatus.DEGRADED if result.get("good") is False else AccountStatus.ONLINE
    return AccountStatus.UNKNOWN


class Accounts:
    def __init__(self, primary, standby, path):
        self.primary, self.standby, self.path = primary, standby, path
        self.enabled = bool(primary or standby)
        if self.enabled and (not primary.isdecimal() or not standby.isdecimal() or primary == standby):
            raise ValueError("Configure two different numeric ONEBOT_PRIMARY_QQ and ONEBOT_STANDBY_QQ")
        self.active = primary
        self.groups = set()
        self._sessions: dict[str, AccountSession] = {}
        self._observe_lock = asyncio.Lock()
        self.connections_changed = asyncio.Event()
        self.cutover = 0
        self.pending = []
        self.down_since = None
        self.next_refresh = 0
        self.awaiting_preferred = False
        self.served = set()
        self.outage_reported = False
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            for account in state["blocked"]:
                self._sessions[account] = AccountSession(None, AccountStatus.BLOCKED)
            if self.enabled and state["pair"] == [primary, standby]:
                self.active = state["active"]
                self.groups = set(state["groups"])
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

    @property
    def blocked(self):
        return {key for key, session in self._sessions.items() if session.status == AccountStatus.BLOCKED}

    def statuses(self):
        accounts = (self.primary, self.standby) if self.enabled else self._sessions
        return {key: self.status(key) for key in accounts}

    def status(self, account):
        session = self._sessions.get(account)
        return session.status if session else AccountStatus.UNKNOWN

    def connection_opened(self, bot):
        status = AccountStatus.BLOCKED if self.status(bot.self_id) == AccountStatus.BLOCKED else AccountStatus.UNKNOWN
        self._sessions[bot.self_id] = AccountSession(bot, status)
        self.connections_changed.set()

    def connection_closed(self, bot):
        session = self._sessions.get(bot.self_id)
        if session and session.bot is bot:
            status = AccountStatus.BLOCKED if session.status == AccountStatus.BLOCKED else AccountStatus.OFFLINE
            self._sessions[bot.self_id] = AccountSession(None, status)
            self.connections_changed.set()

    def session_offline(self, bot):
        session = self._sessions.get(bot.self_id)
        if session and session.bot is bot and session.status != AccountStatus.BLOCKED:
            self._sessions[bot.self_id] = AccountSession(bot, AccountStatus.OFFLINE)

    def forced_offline(self, account):
        if self.enabled and account not in (self.primary, self.standby):
            return
        session = self._sessions.get(account)
        if self.status(account) != AccountStatus.BLOCKED:
            self._sessions[account] = AccountSession(session.bot if session else None, AccountStatus.BLOCKED)
            self.connections_changed.set()
            message = f"QQ {account} 收到平台强制下线通知，已暂停该账号的群、私聊及后台业务。"
            if self.enabled:
                self.notice(message)
            else:
                logger.warning(message)
                self.save()

    def owns(self, account, group=None):
        if self.status(account) == AccountStatus.BLOCKED:
            return False
        if not self.enabled:
            return True
        return account == self.active and (not self.cutover or group is None or int(group) in self.served)

    def is_current(self, bot):
        session = self._sessions.get(bot.self_id)
        return bool(session and session.bot is bot)

    def is_online(self, bot):
        session = self._sessions.get(bot.self_id)
        return bool(session and session.bot is bot and session.status == AccountStatus.ONLINE)

    def can_serve(self, bot, group=None):
        return self.is_online(bot) and self.owns(bot.self_id, group)

    async def observe(self, bots, now):
        async with self._observe_lock:
            await asyncio.gather(*(self._probe(bot) for bot in bots.values()))
            if self.enabled:
                await self._update_route(now)

    async def _probe(self, bot):
        session = self._sessions.get(bot.self_id)
        if session is None or session.bot is not bot or session.status == AccountStatus.BLOCKED:
            return
        status = await probe_status(bot)
        # Connection changes and offline notices supersede an outstanding probe.
        if self._sessions.get(bot.self_id) is session:
            self._sessions[bot.self_id] = AccountSession(bot, status)

    async def _memberships(self, account):
        # Routing depends on both accounts. Discard this read if either changes while waiting.
        before = tuple(self._sessions.get(key) for key in (self.primary, self.standby))
        session = self._sessions.get(account)
        if session is None or session.status != AccountStatus.ONLINE:
            return None
        groups = await memberships(session.bot)
        if before == tuple(self._sessions.get(key) for key in (self.primary, self.standby)):
            return groups
        return None

    def _disconnected(self, account):
        session = self._sessions.get(account)
        return session is None or session.status == AccountStatus.OFFLINE

    async def _update_route(self, now):
        if not self.groups and not self.cutover and self._disconnected(self.primary):
            groups = await self._memberships(self.standby)
            if groups:
                self.active = self.standby
                self.groups = self.served = groups
                self.awaiting_preferred = True
                self.save()
        if self.status(self.active) == AccountStatus.ONLINE and not self.cutover and now >= self.next_refresh:
            self.next_refresh = now + 60
            groups = await self._memberships(self.active)
            if groups is not None and groups != self.groups:
                self.groups = self.served = groups
                self.save()
        other = self.standby if self.active == self.primary else self.primary
        prefer = self.awaiting_preferred and other == self.primary
        if self._disconnected(self.active):
            if self.down_since is None:
                self.down_since = now
        else:
            self.down_since = None
        due = self.down_since is not None and now - self.down_since >= 90
        if (due or prefer) and self.status(self.active) != AccountStatus.BLOCKED:
            groups = await self._memberships(other)
            if groups and (common := self.groups & groups):
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
                    self._sessions[i].bot for i in (self.active, other)
                    if self.status(i) == AccountStatus.ONLINE
                ),
                None,
            )
            if sender is not None:
                message = self.pending.pop(0)
                self.save()
                token = _sending_account_alert.set(True)
                try:
                    await sender.send_private_msg(user_id=int(os.environ["MASTER_QQ"]), message=message, _timeout=5)
                except Exception as exc:
                    logger.warning(f"主备提醒投递未确认: {type(exc).__name__}")
                finally:
                    _sending_account_alert.reset(token)


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
    return {key: bot for key, bot in get_bots().items() if accounts.can_serve(bot)}


@event_preprocessor
async def account_events(bot: Bot, event: Event):
    if getattr(event, "notice_type", None) == "bot_offline":
        accounts.forced_offline(bot.self_id)
    elif getattr(event, "notice_type", None) == "bot_status" and getattr(event, "sub_type", None) == "offline":
        accounts.session_offline(bot)
    group = getattr(event, "group_id", None)
    if event.post_type != "meta_event" and (
        not accounts.can_serve(bot, group)
        or (accounts.enabled and (
            getattr(event, "user_id", None) in (int(accounts.primary), int(accounts.standby))
            or event.time <= accounts.cutover
        ))
    ):
        raise IgnoredException("event outside service account ownership")


class AccountAdapter(Adapter):
    def bot_connect(self, bot):
        super().bot_connect(bot)
        accounts.connection_opened(bot)

    def bot_disconnect(self, bot):
        if self.bots.get(bot.self_id) is bot:
            accounts.connection_closed(bot)
            super().bot_disconnect(bot)

    async def _call_api(self, bot, api, **data):
        if not accounts.is_current(bot):
            raise RuntimeError("action from an expired account connection")
        if api not in {"get_status", "get_login_info"} and not accounts.can_serve(bot, data.get("group_id")):
            alert = (api == "send_private_msg" and _sending_account_alert.get()
                     and str(data.get("user_id")) == os.getenv("MASTER_QQ"))
            if not (accounts.is_online(bot) and (api == "get_group_list" or alert)):
                raise RuntimeError("action outside available service account")
        return await super()._call_api(bot, api, **data)
