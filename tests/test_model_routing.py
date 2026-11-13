import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

nonebot.init()

from model_env import TEST_MODEL_ENV
from src.plugins.llm_chat import commands, models, roles
from src.plugins.llm_chat.config import config


class ModelRoutingTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, TEST_MODEL_ENV, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_general_tasks_share_one_configuration(self):
        os.environ['MODEL_GENERAL'] = 'openai:general-first | openai:general-second'
        for role in ('briefing', 'news_interest', 'jrrp', 'keyword_extract'):
            with self.subTest(role=role):
                self.assertEqual(roles.resolve_role(role), (
                    'openai:general-first', ['openai:general-second', 'openai:test-fallback'],
                ))
                self.assertEqual(roles.role_config_source(role), 'MODEL_GENERAL')

    def test_high_volume_tasks_inherit_general_when_unset_or_blank(self):
        for role in ('chat', 'planner', 'summary', 'important_info'):
            key = roles.env_key_for(role)
            for empty in ('', '  ', None):
                with self.subTest(role=role, empty=empty):
                    if empty is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = empty
                    self.assertEqual(roles.resolve_role(role), (
                        'cliproxyapi:test-general', ['openai:test-fallback'],
                    ))

    def test_explicit_task_chain_uses_fallback_without_adding_general(self):
        os.environ['MODEL_SUMMARY'] = 'openai:summary-primary | openai:summary-secondary'
        self.assertEqual(roles.resolve_role('summary'), (
            'openai:summary-primary', ['openai:summary-secondary', 'openai:test-fallback'],
        ))
        self.assertEqual(roles.role_config_source('summary'), 'MODEL_SUMMARY')

    def test_duplicates_remain_ordered_retry_slots(self):
        os.environ['MODEL_GENERAL'] = 'openai:a | openai:a | openai:b'
        os.environ['MODEL_FALLBACK'] = 'openai:b | openai:c | openai:a'
        self.assertEqual(roles.resolve_role('jrrp'), (
            'openai:a', ['openai:a', 'openai:b', 'openai:b', 'openai:c', 'openai:a'],
        ))
        self.assertEqual(roles.resolve_role('fallback'), ('openai:b', ['openai:c', 'openai:a']))

    def test_no_role_keeps_duplicate_global_fallbacks_including_primary(self):
        os.environ['MODEL_CHAT'] = 'openai:a'
        os.environ['MODEL_FALLBACK'] = 'openai:a | openai:b | openai:a'
        manager = models.ModelManager()
        for explicit in (None, 'openai:a', 'openai:b'):
            with self.subTest(model=explicit):
                primary = explicit or 'openai:a'
                self.assertEqual(manager._resolve_candidates(role=None, model=explicit), (
                    primary, [primary, 'openai:a', 'openai:b', 'openai:a'],
                ))

    def test_session_override_keeps_every_configured_retry_slot(self):
        os.environ['MODEL_CHAT'] = 'openai:a | openai:b | openai:a'
        os.environ['MODEL_FALLBACK'] = 'openai:b | openai:a'
        manager = models.ModelManager()
        configured = ['openai:a', 'openai:b', 'openai:a', 'openai:b', 'openai:a']
        for explicit, expected in ((None, configured), ('openai:a', configured),
                                   ('openai:b', ['openai:b', *configured])):
            with self.subTest(model=explicit):
                primary, candidates = manager._resolve_candidates(role='chat', model=explicit)
                self.assertEqual(primary, explicit or 'openai:a')
                self.assertEqual(candidates, expected)

    def test_tools_never_inherit_text_models(self):
        os.environ['MODEL_FALLBACK'] = 'openai:ordinary-text'
        for role in ('search', 'vision', 'tts', 'image_gen'):
            key = roles.env_key_for(role)
            self.assertEqual(roles.resolve_role(role), (TEST_MODEL_ENV[key], []))
            os.environ.pop(key)
            with self.assertRaisesRegex(ValueError, key):
                roles.resolve_role(role)

    def test_missing_general_and_empty_chain_fail_explicitly(self):
        os.environ.pop('MODEL_GENERAL')
        self.assertTrue(any('MODEL_GENERAL' in error for error in roles.validate_all()))
        os.environ['MODEL_GENERAL'] = ' | '
        with self.assertRaisesRegex(ValueError, '不能为空'):
            roles.resolve_role('jrrp')

    def test_unknown_role_does_not_silently_use_general(self):
        with self.assertRaisesRegex(ValueError, '未知'):
            roles.resolve_role('summmary')
        self.assertNotIn('apology', roles.all_roles())

    def test_legacy_settings_are_reported_instead_of_silently_ignored(self):
        os.environ['ROLE_BRIEFING'] = 'openai:legacy'
        errors = roles.validate_all()
        self.assertEqual(len(errors), 1)
        self.assertIn('ROLE_BRIEFING', errors[0])
        self.assertIn('迁移', errors[0])

    def test_startup_rejects_incompatible_tool_fallback(self):
        os.environ['MODEL_SEARCH'] = 'dashscope:test-search | openai:ordinary-text'
        errors = roles.validate_all()
        self.assertTrue(any('MODEL_SEARCH' in e and 'ordinary-text' in e for e in errors))

    def test_global_provider_key_does_not_make_search_usable(self):
        with patch.object(roles, 'provider_configured', side_effect=lambda p: p != 'dashscope'):
            errors = roles.validate_all()
        self.assertTrue(any('MODEL_SEARCH' in e and 'API key' in e for e in errors))

    def test_disabled_media_does_not_require_media_models(self):
        for key in ('MODEL_TTS', 'MODEL_IMAGE_GEN'):
            os.environ.pop(key)
        with patch.dict("os.environ", {"MODEL_TTS": ""}), patch.dict("os.environ", {"MODEL_IMAGE_GEN": ""}):
            self.assertEqual(roles.validate_all(), [])

    def test_only_cliproxyapi_images_need_no_responses_driver(self):
        os.environ["MODEL_GENERAL"]="openai:text-only"
        with patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}):
            self.assertFalse(roles.role_is_required("image_driver"))
            self.assertEqual(roles.validate_all(),[])

    def test_image_validation_is_independent_of_general_model(self):
        os.environ["MODEL_GENERAL"]="openai:text-only"
        with patch.dict("os.environ", {"MODEL_IMAGE_GEN": "cliproxyapi:gpt-image-test"}):
            self.assertFalse(roles.role_is_required("image_driver"))
            self.assertEqual(roles.validate_all(),[])

    def test_driver_reuses_general_and_global_fallback(self):
        os.environ['MODEL_GENERAL'] = 'google:text-only | cliproxyapi:driver-primary'
        os.environ['MODEL_FALLBACK'] = 'cliproxyapi:driver-backup | openai:text-backup'
        self.assertEqual(roles.role_config_source('image_driver'), 'MODEL_GENERAL')
        self.assertEqual(roles.resolve_role('image_driver'), roles.resolve_role('general'))
        self.assertEqual(roles.validate_all(), [])

    def test_direct_image_backup_can_cover_incompatible_general_models(self):
        os.environ['MODEL_GENERAL'] = 'openai:text-only'
        os.environ['MODEL_IMAGE_GEN'] = 'cliproxyapi:gpt-image-test | openai:gpt-image-backup'
        self.assertEqual(roles.validate_all(), [])

    def test_removed_driver_setting_is_reported(self):
        os.environ['MODEL_IMAGE_DRIVER'] = 'cliproxyapi:old-driver'
        self.assertTrue(any('MODEL_IMAGE_DRIVER 已合并' in e for e in roles.validate_all()))

    def test_status_shows_general_inheritance_without_driver_setting(self):
        os.environ.pop('MODEL_PLANNER')
        status = roles.model_config_status()
        self.assertIn('MODEL_PLANNER（继承 MODEL_GENERAL）: cliproxyapi:test-general', status)
        self.assertIn('MODEL_GENERAL: cliproxyapi:test-general', status)
        self.assertNotIn('MODEL_IMAGE_DRIVER', status)
        self.assertNotIn('ROLE_APOLOGY', status)

    def test_runtime_search_filter_also_rejects_explicit_text_override(self):
        _, candidates = models.model_manager._resolve_candidates(
            role='search', model='openai:session-model',
        )
        self.assertEqual(candidates, ['dashscope:test-search'])

    def test_chat_session_override_still_keeps_configured_backups(self):
        primary, candidates = models.model_manager._resolve_candidates(
            role='chat', model='openai:session-model',
        )
        self.assertEqual(primary, 'openai:session-model')
        self.assertEqual(candidates, ['openai:session-model', 'openai:test-chat', 'openai:test-fallback'])


class RoutingExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_same_model_after_other_candidate_and_stop_at_first_success(self):
        for role, first, second in (
            ('chat', 'openai:a', 'openai:b'),
            ('vision', 'google:gemma-a', 'google:gemma-b'),
        ):
            for retry in (False, True):
                with self.subTest(role=role, retry=retry):
                    env = {**TEST_MODEL_ENV, roles.env_key_for(role): f'{first} | {second} | {first}'}
                    manager = models.ModelManager()
                    outcomes = ([RuntimeError('first attempt unavailable')] if retry else [])
                    first_call = AsyncMock(side_effect=[*outcomes, models.ChatResponse('成功')])
                    second_call = AsyncMock(side_effect=RuntimeError('second candidate unavailable'))
                    adapters = {
                        first: SimpleNamespace(chat=first_call, chat_with_images=first_call),
                        second: SimpleNamespace(chat=second_call, chat_with_images=second_call),
                    }
                    with (
                        patch.dict(os.environ, env, clear=True),
                        patch.object(manager, 'get_or_build_adapter', side_effect=adapters.__getitem__) as build,
                        patch.object(models, 'start_call') as start,
                        patch.object(models, 'finish_call') as finish,
                    ):
                        messages = [models.Message(role='user', content='测试请求')]
                        if role == 'vision':
                            result = await manager.chat_with_images(messages, ['test-image'])
                        else:
                            result = await manager.chat(messages, role=role)
                    expected = [first, second, first] if retry else [first]
                    self.assertEqual([call.args[0] for call in build.call_args_list], expected)
                    self.assertEqual(first_call.await_count, 2 if retry else 1)
                    self.assertEqual(second_call.await_count, 1 if retry else 0)
                    self.assertEqual(str(result), '成功')
                    self.assertEqual(result.model_spec, first)
                    self.assertEqual(result.primary_spec, first)
                    self.assertEqual(result.fallback_attempt, len(expected) - 1)
                    self.assertEqual([call.kwargs['extra']['attempt'] for call in start.call_args_list],
                                     list(range(len(expected))))
                    self.assertEqual([call.kwargs['fallback_attempt'] for call in finish.call_args_list],
                                     list(range(len(expected))))

    async def test_search_failure_does_not_call_plain_text_fallback(self):
        env = {**TEST_MODEL_ENV, 'MODEL_SEARCH': 'dashscope:first | dashscope:second'}
        adapter = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError('search unavailable')))
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(models.model_manager, 'get_or_build_adapter', return_value=adapter) as build,
            patch.object(models, 'start_call'),
            patch.object(models, 'finish_call'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'search unavailable'):
                await models.model_manager.chat([models.Message(role='user', content='search')], role='search')
        self.assertEqual([c.args[0] for c in build.call_args_list], ['dashscope:first', 'dashscope:second'])
        for call in adapter.chat.await_args_list:
            self.assertTrue(call.kwargs['extra_body']['enable_search'])

    async def test_jrrp_uses_planned_chat_without_a_session_override(self):
        event = SimpleNamespace(user_id=123, sender=SimpleNamespace(card='', nickname='Tester'))
        with (
            patch.dict(os.environ, TEST_MODEL_ENV, clear=True),
            patch.object(commands.memory_manager, 'get_session_model', return_value='openai:chat-override'),
            patch.object(commands, '_build_jrrp_memory_context', AsyncMock(return_value='')),
            patch.object(commands, '_record_command_interaction', AsyncMock()),
            patch.object(commands.jrrp_cmd, 'send', AsyncMock()),
            patch.object(commands.model_manager, 'run_turn', AsyncMock(return_value='50')) as chat,
        ):
            await commands.handle_jrrp(event)
        kwargs = chat.await_args.kwargs
        self.assertEqual(kwargs['role'], 'chat')
        self.assertEqual(kwargs['reply_policy'].scene, 'jrrp')
        self.assertIsNone(kwargs.get('model'))

    async def test_status_command_displays_the_new_configuration(self):
        event = SimpleNamespace(user_id=123)
        with (
            patch.dict(os.environ, TEST_MODEL_ENV, clear=True),
            patch.object(commands.memory_manager, 'get_history', AsyncMock(return_value=[])),
            patch.object(commands.memory_manager, 'get_session_model', return_value='openai:chat-override'),
            patch.object(commands.status_cmd, 'finish', AsyncMock()) as finish,
        ):
            await commands.handle_status(event)
        text = finish.await_args.args[0]
        self.assertIn('openai:chat-override', text)
        self.assertIn('MODEL_GENERAL', text)
        self.assertNotIn('MODEL_IMAGE_DRIVER', text)
        self.assertNotIn('绘图外层调用', text)
        self.assertNotIn('ROLE_APOLOGY', text)
