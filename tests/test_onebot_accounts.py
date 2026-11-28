import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from nonebot.adapters.onebot.v11 import Adapter
from nonebot.adapters.onebot.v11.exception import ActionFailed
from nonebot.exception import IgnoredException

from lysbot import get_service_bot
from src.onebot_accounts import Accounts, AccountAdapter, account_events, active_bots


def make_bot(account, groups):
    return Mock(
        self_id=account,
        get_status=AsyncMock(return_value={"online": True, "good": True}),
        get_group_list=AsyncMock(return_value=[{"group_id": i} for i in groups]),
        send_private_msg=AsyncMock(),
    )


def message(group):
    return Mock(group_id=group, user_id=33, time=int(time.time()) + 1,
                post_type="message", notice_type=None)


class AccountTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "state.json"
        self.main, self.spare = make_bot("11", [1, 2]), make_bot("22", [2, 3])
        self.bots = {"11": self.main, "22": self.spare}
        self.accounts = Accounts("11", "22", self.path)
        self.bind_accounts()
        self.enterContext(patch("src.onebot_accounts.get_bots", return_value=self.bots))
        self.enterContext(patch.dict("os.environ", MASTER_QQ="33"))
        self.wire = self.enterContext(patch.object(Adapter, "_call_api", new_callable=AsyncMock))
        self.adapter = object.__new__(AccountAdapter)

    def bind_accounts(self):
        self.enterContext(patch("src.onebot_accounts.accounts", self.accounts))
        for bot in self.bots.values():
            self.accounts.connection_opened(bot)

    async def assert_business_access(self, bot, allowed):
        for group in (2, None):
            event = message(group)
            api, params = ("send_group_msg", {"group_id": group}) if group else ("send_private_msg", {"user_id": 33})
            with self.subTest(account=bot.self_id, group=group):
                self.assertEqual(self.accounts.can_serve(bot, group), allowed)
                if allowed:
                    await account_events(bot, event)
                    await self.adapter._call_api(bot, api, **params)
                    self.wire.assert_awaited_once()
                else:
                    with self.assertRaises(IgnoredException):
                        await account_events(bot, event)
                    with self.assertRaises(RuntimeError):
                        await self.adapter._call_api(bot, api, **params)
                    self.wire.assert_not_awaited()
                self.wire.reset_mock()

    async def test_offline_or_unknown_stops_group_private_background_and_extension_sends(self):
        await self.accounts.observe(self.bots, 0)
        self.assertIs(get_service_bot(), self.main)
        await self.assert_business_access(self.main, True)
        await self.assert_business_access(self.spare, False)
        for raw, expected in (({"online": False}, "offline"), ({}, "unknown"), ({"online": True}, "online")):
            self.main.get_status.return_value = raw
            await self.accounts.observe(self.bots, 10)
            self.assertEqual(self.accounts.status("11"), expected)
            ready = expected == "online"
            self.assertEqual(active_bots(), {"11": self.main} if ready else {})
            self.assertIs(get_service_bot(), self.main if ready else None)
            await self.assert_business_access(self.main, ready)

    async def test_switch_after_grace_and_keep_owner_after_restart(self):
        await self.accounts.observe(self.bots, 0)
        self.main.get_status.return_value = {"online": False}
        for t in (1, 90):
            await self.accounts.observe(self.bots, t)
            self.assertEqual(self.accounts.active, "11")
        await self.accounts.observe(self.bots, 91)
        self.spare.send_private_msg.assert_awaited_once()
        self.accounts = Accounts("11", "22", self.path)
        self.bind_accounts()
        self.main.get_status.return_value = {"online": True}
        await self.accounts.observe(self.bots, 120)
        self.assertIs(get_service_bot(), self.spare)
        self.assertFalse(self.accounts.owns("11", 2))
        self.assertTrue(self.accounts.can_serve(self.spare, 2))
        self.assertFalse(self.accounts.can_serve(self.spare, 3))
        await self.assert_business_access(self.spare, True)
        await self.assert_business_access(self.main, False)
        self.spare.get_status.return_value = {"online": False}
        await self.accounts.observe(self.bots, 150)
        await self.accounts.observe(self.bots, 240)
        self.assertIs(get_service_bot(), self.main)
        self.assertTrue(self.accounts.can_serve(self.main, 1))

    async def test_startup_waits_for_connected_primary_with_unknown_or_degraded_status(self):
        for start, raw, expected in ((1, {}, "unknown"), (130, {"online": True, "good": False}, "degraded")):
            self.main.get_status.return_value = raw
            for now in (start, start + 100):
                await self.accounts.observe(self.bots, now)
                self.assertEqual(self.accounts.status("11"), expected)
                self.assertEqual(self.accounts.active, "11")
                self.assertIsNone(get_service_bot())
                self.spare.get_group_list.assert_not_awaited()
        self.main.get_status.return_value = {"online": False}
        await self.accounts.observe(self.bots, 240)
        self.assertTrue(self.accounts.can_serve(self.spare, 2))

    async def test_degraded_owner_after_startup_never_donates_work_to_the_standby(self):
        await self.accounts.observe(self.bots, 0)
        self.main.get_status.return_value = {"online": True, "good": False}
        for now in (10, 150):
            await self.accounts.observe(self.bots, now)
            self.assertEqual(self.accounts.active, "11")
            self.assertEqual(self.accounts.status("11"), "degraded")
            self.assertIsNone(get_service_bot())
            await self.assert_business_access(self.main, False)
        self.spare.get_group_list.assert_not_awaited()

    async def test_first_online_account_serves_until_preferred_is_ready(self):
        self.bots.pop("11")
        self.accounts.connection_closed(self.main)
        await self.accounts.observe(self.bots, 1)
        self.assertTrue(self.accounts.can_serve(self.spare, 2))
        self.bots["11"] = self.main
        self.accounts = Accounts("11", "22", self.path)
        self.bind_accounts()
        await self.accounts.observe(self.bots, 10)
        self.assertEqual(self.accounts.active, "11")
        self.assertFalse(self.accounts.awaiting_preferred)
        self.assertFalse(self.accounts.can_serve(self.main, 1))

    async def test_forced_offline_during_startup_membership_lookup_prevents_takeover(self):
        async def read_groups(**kwargs):
            self.accounts.forced_offline("11")
            return [{"group_id": 2}]

        self.spare.get_group_list.side_effect = read_groups
        self.main.get_status.return_value = {"online": False}
        await self.accounts.observe(self.bots, 1)
        self.spare.get_group_list.assert_awaited()
        self.spare.send_private_msg.assert_awaited_once()
        for accounts in (self.accounts, Accounts("11", "22", self.path)):
            self.assertEqual(accounts.active, "11")
            self.assertEqual(accounts.status("11"), "blocked")
            self.assertFalse(accounts.owns("11", 2))
            self.assertFalse(accounts.owns("22", 2))

    async def test_legacy_block_survives_restart_and_reconnection(self):
        self.path.write_text(json.dumps({
            "pair": ["11", "22"], "active": "11", "groups": [2], "served": [2],
            "blocked": ["11"], "cutover": 0, "pending": [],
        }), encoding="utf-8")
        self.accounts = Accounts("11", "22", self.path)
        self.bind_accounts()
        for now in (10, 120):
            await self.accounts.observe(self.bots, now)
            self.assertEqual(self.accounts.status("11"), "blocked")
            self.assertIsNone(get_service_bot())
            await self.assert_business_access(self.main, False)
        self.assertEqual(self.accounts.status("22"), "online")
        self.assertEqual(self.accounts.active, "11")

    async def test_old_connection_probe_and_disconnect_cannot_change_replacement_session(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_status(**kwargs):
            started.set()
            await release.wait()
            return {"online": True, "good": True}

        self.main.get_status.side_effect = slow_status
        pending = asyncio.create_task(self.accounts.observe(self.bots.copy(), 1))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            self.accounts.connection_closed(self.main)
            replacement = self.bots["11"] = make_bot("11", [1, 2])
            self.accounts.connection_opened(replacement)
        finally:
            release.set()
            await pending
        self.assertEqual(self.accounts.status("11"), "unknown")
        await self.accounts.observe(self.bots, 2)
        self.accounts.connection_closed(self.main)
        self.assertEqual(self.accounts.status("11"), "online")
        self.assertIs(get_service_bot(), replacement)
        await self.assert_business_access(self.main, False)
        await self.assert_business_access(replacement, True)
        with self.assertRaises(RuntimeError):
            await self.adapter._call_api(self.main, "get_status")
        self.wire.assert_not_awaited()
        await self.adapter._call_api(replacement, "get_status")
        self.wire.assert_awaited_once()

    async def test_offline_result_takes_effect_while_other_account_probe_is_pending(self):
        await self.accounts.observe(self.bots, 0)
        standby_started, main_sampled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def slow_standby(**kwargs):
            standby_started.set()
            await release.wait()
            return {"online": True}

        async def offline_main(**kwargs):
            await standby_started.wait()
            main_sampled.set()
            return {"online": False}

        self.main.get_status.side_effect = offline_main
        self.spare.get_status.side_effect = slow_standby
        pending = asyncio.create_task(self.accounts.observe(self.bots, 1))
        try:
            await asyncio.wait_for(main_sampled.wait(), timeout=2)
            self.assertFalse(pending.done())
            self.assertEqual(self.accounts.status("11"), "offline")
            self.assertIsNone(get_service_bot())
            await self.assert_business_access(self.main, False)
        finally:
            release.set()
            await pending

    async def test_offline_notice_beats_an_older_probe_then_fresh_probe_can_recover(self):
        await self.accounts.observe(self.bots, 0)
        notice = Mock(notice_type="bot_status", sub_type="offline", post_type="notice",
                      time=int(time.time()), group_id=None, user_id=11)

        async def stale_online(**kwargs):
            try:
                await account_events(self.main, notice)
            except IgnoredException:
                pass
            return {"online": True, "good": True}

        self.main.get_status.side_effect = stale_online
        await self.accounts.observe(self.bots, 2)
        self.assertEqual(self.accounts.status("11"), "offline")
        self.assertIsNone(get_service_bot())
        self.main.get_status.side_effect = None
        await self.accounts.observe(self.bots, 3)
        self.assertIs(get_service_bot(), self.main)

    async def test_single_account_block_from_notice_beats_probe_and_persists(self):
        del self.bots["22"]
        self.accounts = Accounts("", "", self.path)
        self.bind_accounts()
        await self.accounts.observe(self.bots, 0)
        self.assertIs(get_service_bot(), self.main)
        notice = Mock(notice_type="bot_offline", post_type="notice", time=int(time.time()),
                      group_id=None, user_id=11)

        async def late_online(**kwargs):
            try:
                await account_events(self.main, notice)
            except IgnoredException:
                pass
            return {"online": True}

        self.main.get_status.side_effect = late_online
        await self.accounts.observe(self.bots, 1)
        self.assertEqual(self.accounts.status("11"), "blocked")
        await self.assert_business_access(self.main, False)
        self.accounts = Accounts("", "", self.path)
        self.bind_accounts()
        self.main.get_status.side_effect = None
        await self.accounts.observe(self.bots, 2)
        self.assertEqual(self.accounts.status("11"), "blocked")
        self.assertIsNone(get_service_bot())
        await self.assert_business_access(self.main, False)

    async def test_blocked_standby_does_not_disable_healthy_owner(self):
        await self.accounts.observe(self.bots, 0)
        self.accounts.forced_offline("22")
        await self.accounts.observe(self.bots, 1)
        self.assertEqual(self.accounts.status("22"), "blocked")
        self.assertIs(get_service_bot(), self.main)
        await self.assert_business_access(self.main, True)

    async def test_group_permission_error_leaves_account_online_for_other_destinations(self):
        await self.accounts.observe(self.bots, 0)
        self.wire.side_effect = ActionFailed(status="failed", retcode=100, wording="群内禁言")
        with self.assertRaises(ActionFailed):
            await self.adapter._call_api(self.main, "send_group_msg", group_id=1, message="one")
        self.wire.side_effect = None
        self.wire.reset_mock()
        self.assertEqual(self.accounts.status("11"), "online")
        await self.assert_business_access(self.main, True)

    async def test_block_stops_non_message_writes_and_still_allows_status_diagnostics(self):
        await self.accounts.observe(self.bots, 0)
        self.accounts.forced_offline("11")
        with self.assertRaises(RuntimeError):
            await self.adapter._call_api(self.main, "set_group_card", group_id=2, user_id=11)
        with self.assertRaises(RuntimeError):
            await self.adapter._call_api(self.main, "get_group_msg_history", group_id=2)
        self.wire.assert_not_awaited()
        await self.adapter._call_api(self.main, "get_status")
        self.wire.assert_awaited_once()

    async def test_account_notice_only_allows_connected_standby_to_notify_master(self):
        checked = False

        async def check_alert(user_id, **kwargs):
            nonlocal checked
            self.assertEqual(user_id, 33)
            await self.adapter._call_api(self.spare, "send_private_msg", user_id=user_id, message="alert")
            self.wire.assert_awaited_once()
            self.wire.reset_mock()
            for bot, api, params in (
                (self.spare, "send_private_msg", {"user_id": 44}),
                (self.spare, "send_group_msg", {"group_id": 2}),
                (self.main, "send_private_msg", {"user_id": 33}),
            ):
                with self.assertRaises(RuntimeError):
                    await self.adapter._call_api(bot, api, **params)
            self.wire.assert_not_awaited()
            checked = True

        await self.accounts.observe(self.bots, 0)
        self.accounts.forced_offline("11")
        self.spare.send_private_msg.side_effect = check_alert
        await self.accounts.observe(self.bots, 1)
        self.spare.send_private_msg.assert_awaited_once()
        self.main.send_private_msg.assert_not_awaited()
        self.assertTrue(checked, "The notice must pass all checks at the transport boundary")
