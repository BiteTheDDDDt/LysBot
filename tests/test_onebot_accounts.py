import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from nonebot.exception import IgnoredException
from src.onebot_accounts import AccountAdapter, account_events

from src.onebot_accounts import Accounts


class AccountTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "state.json"
        self.accounts = Accounts("11", "22", self.path)
        self.main, self.spare = AsyncMock(), AsyncMock()
        self.main.get_group_list.return_value = [{"group_id": 1}, {"group_id": 2}]
        self.spare.get_group_list.return_value = [{"group_id": 2}, {"group_id": 3}]
        self.bots = {"11": self.main, "22": self.spare}

    async def test_switch_after_grace_and_keep_owner_after_restart(self):
        with patch.dict("os.environ", MASTER_QQ="33"):
            await self.accounts.observe(self.bots, {"11": True, "22": True}, 0)
            for t in (1, 90):
                await self.accounts.observe(self.bots, {"11": False, "22": True}, t)
                self.assertEqual(self.accounts.active, "11")
            await self.accounts.observe(self.bots, {"11": False, "22": True}, 91)
            self.spare.send_private_msg.assert_awaited_once()
            restored = Accounts("11", "22", self.path)
            self.assertFalse(restored.owns("11", 2))
            self.assertTrue(restored.owns("22", 2))
            self.assertFalse(restored.owns("22", 3))
            await restored.observe(self.bots, {"11": True, "22": True}, 120)
            self.assertEqual(restored.active, "22")
            await restored.observe(self.bots, {"11": True, "22": False}, 150)
            await restored.observe(self.bots, {"11": True, "22": False}, 240)
            self.assertEqual(restored.active, "11")
            self.assertTrue(restored.owns("11", 1))

    async def test_forced_offline_survives_disconnect(self):
        self.accounts.forced_offline("11")
        restored = Accounts("11", "22", self.path)
        restored.groups = {2}
        with patch.dict("os.environ", MASTER_QQ="33"):
            for t in (10, 120):
                await restored.observe({"22": self.spare}, {"22": True}, t)
        self.assertFalse(restored.owns("11", 2))
        self.assertFalse(restored.owns("22", 2))
        self.spare.send_private_msg.assert_awaited_once()

    async def test_standby_event_and_stale_delivery_are_rejected(self):
        event = Mock(group_id=2, user_id=33, time=100, notice_type=None)
        with patch("src.onebot_accounts.accounts", self.accounts):
            await account_events(Mock(self_id="11"), event)
            with self.assertRaises(IgnoredException):
                await account_events(Mock(self_id="22"), event)
            self.accounts.active = "22"
            self.accounts.groups = {2}
            with self.assertRaises(RuntimeError):
                await AccountAdapter._call_api(None, Mock(self_id="11"), "send_group_msg", group_id=2)

    async def test_switch_moves_private_messages_and_plugin_sender_together(self):
        from lysbot import get_service_bot
        event = Mock(group_id=None, user_id=33, time=200, post_type="message")
        with patch("src.onebot_accounts.accounts", self.accounts), patch("src.onebot_accounts.get_bots", return_value=self.bots):
            for active, other in (("11", "22"), ("22", "11")):
                self.accounts.active = active
                self.accounts.online = {active: True, other: True}
                self.assertIs(get_service_bot(), self.bots[active])
                await account_events(Mock(self_id=active), event)
                with self.assertRaises(IgnoredException):
                    await account_events(Mock(self_id=other), event)
                with self.assertRaises(RuntimeError):
                    await AccountAdapter._call_api(None, Mock(self_id=other), "send_private_msg", user_id=33)
                self.accounts.online[active] = False
                self.assertIsNone(get_service_bot())

    async def test_first_online_account_serves_until_preferred_is_ready(self):
        with patch.dict("os.environ", MASTER_QQ="33"):
            await self.accounts.observe({"22": self.spare}, {"22": True}, 1)
            self.assertTrue(self.accounts.owns("22", 2))
            restored = Accounts("11", "22", self.path)
            await restored.observe(self.bots, {"11": True, "22": True}, 10)
            self.assertEqual(restored.active, "11")
            self.assertFalse(restored.awaiting_preferred)
            self.assertFalse(restored.owns("11", 1))
