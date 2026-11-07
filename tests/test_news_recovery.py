import json
import socket
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import nonebot

nonebot.init()
from src.plugins.llm_chat import news_broadcast as news, news_sources as sources, public_net
from src.plugins.llm_chat.memory import MemoryManager


def dns(address):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))]


class NewsSourceTests(unittest.IsolatedAsyncioTestCase):
    async def inspect(self, url, handler, *, address='198.18.0.8'):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(sources.httpx, 'AsyncClient', return_value=client), \
             patch.object(public_net.socket, 'getaddrinfo', return_value=dns(address)):
            return await sources.inspect_publication_date(url)

    async def test_fake_dns_is_resolved_independently_and_public_connection_is_pinned(self):
        requests = []
        def handle(request):
            requests.append(request)
            if request.url.host == 'dns.google':
                self.assertEqual(request.url.params['name'], 'example.com')
                return httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': '93.184.216.34'}]})
            self.assertEqual(request.url.host, '93.184.216.34')
            self.assertEqual(request.headers['host'], 'example.com')
            self.assertEqual(request.extensions['sni_hostname'], 'example.com')
            return httpx.Response(200, headers={'content-type': 'text/html'},
                                  text='<meta property="article:published_time" content="2026-09-12T10:00:00+08:00">')
        result = await self.inspect('https://example.com/news', handle)
        self.assertEqual(result, {'status': 'verified', 'published_at': '2026-09-12'})
        self.assertEqual(len(requests), 2)

    async def test_private_destinations_and_private_dns_answers_are_not_fetched(self):
        for url, address in [('http://127.0.0.1/a', '198.18.0.8'),
                             ('http://198.18.0.8/a', '198.18.0.8'),
                             ('http://metadata.internal/a', '198.18.0.8'),
                             ('http://example.com:8000/a', '93.184.216.34'),
                             ('http://example.com/a', '192.168.1.1')]:
            calls = []
            result = await self.inspect(url, lambda r: calls.append(r), address=address)
            self.assertEqual(result['published_at'], '')
            self.assertEqual(calls, [])
        requests = []
        def private_answer(request):
            requests.append(request)
            return httpx.Response(200, json={'Status': 0, 'Answer': [{'type': 1, 'data': '169.254.169.254'}]})
        result = await self.inspect('https://example.com/a', private_answer)
        self.assertEqual(result['published_at'], '')
        self.assertEqual(len(requests), 1)

    async def test_redirect_is_validated_before_fetch(self):
        requests = []
        def redirect(request):
            requests.append(request)
            return httpx.Response(302, headers={'location': 'http://169.254.169.254/latest/meta-data/'})
        result = await self.inspect('https://example.com/a', redirect, address='93.184.216.34')
        self.assertEqual(result['published_at'], '')
        self.assertEqual(len(requests), 1)

    async def test_unavailable_or_missing_metadata_remains_distinguishable(self):
        for status in [200, 403]:
            result = await self.inspect('https://example.com/a',
                lambda r: httpx.Response(status, headers={'content-type': 'text/html'}, text='<html>no date</html>'),
                address='93.184.216.34')
            self.assertEqual(result['status'], 'missing_metadata' if status == 200 else 'http_403')
    async def test_unknown_dates_trigger_fallback_not_successful_empty_search(self):
        source = {'index': 1, 'url': 'https://example.com/a', 'title': 'Update', 'site_name': 'Example', 'published_at': ''}
        native = {'content': '{"items":[{"source_index":1,"summary":"Update"}]}', 'sources': [source]}
        with patch.object(news, 'resolve_role', return_value=('dashscope:first', ['dashscope:second'])), \
             patch.object(news, '_dashscope_native_search', AsyncMock(return_value=native)) as search, \
             patch.object(sources, 'fetch_public_page', AsyncMock()) as fetch:
            with self.assertRaisesRegex(RuntimeError, '未能核验'):
                await news.search_news_interest({'label': 'AI'}, now=datetime(2026,9,12))
        self.assertEqual(search.await_count, 2)
        fetch.assert_not_awaited()


class NativeNewsDateTests(unittest.TestCase):
    def test_native_dates_need_known_url_explicit_confirmation_and_recent_date(self):
        url = "https://example.com/a"
        valid = {"url": url, "published_at": "2026-09-12", "date_confirmed": True,
                 "date_quote": "Published September 12, 2026"}
        for changes in (
            {"published_at": ""},
            {"published_at": "今天"},
            {"published_at": "2026-02-30"},
            {"date_confirmed": False},
            {"date_quote": ""},
            {"url": "https://invented.example/a"},
            {"url": "https://invented.example/a", "source_index": 1},
            {"published_at": "2026-09-01"},
            {"published_at": "2026-09-13"},
        ):
            with self.subTest(changes=changes):
                source = {"index": 1, "url": url, "title": "Update", "site_name": "Example", "published_at": ""}
                native = {"content": json.dumps({"findings": [{**valid, **changes}]}), "sources": [source]}
                news._apply_native_date_evidence(native)
                items = news.parse_native_news_selection(
                    '{"items":[{"source_index":1,"published_at":"2026-09-12","summary":"Update"}]}',
                    native["sources"], interest_label="AI",
                    earliest_date=datetime(2026, 9, 10).date(), latest_date=datetime(2026, 9, 12).date(),
                )
                self.assertEqual(items, [])

    def test_conflicting_search_dates_remain_unknown_and_provider_metadata_is_preserved(self):
        source = {"index": 1, "url": "https://example.com/a", "published_at": ""}
        claim = {"source_index": 1, "published_at": "2026-09-12", "date_confirmed": True, "date_quote": "Published Sep 12"}
        native = {"sources": [source], "content": json.dumps({
            "items": [claim, {**claim, "published_at": "2026-09-11"}],
        })}
        news._apply_native_date_evidence(native)
        self.assertEqual(source["published_at"], "")
        source.update(published_at="2026-09-10", date_evidence="search_provider")
        news._apply_native_date_evidence(native)
        self.assertEqual(source["published_at"], "2026-09-10")
        self.assertEqual(source["date_evidence"], "search_provider")


class NewsRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = object.__new__(MemoryManager)
        self.manager._init()
        self.manager._db_path = Path(self.temp.name) / 'history.db'
        self.now = datetime(2026,9,12,8)
        self.cutoff = self.now - timedelta(hours=36)
        await self.manager.save_published_briefing(71, {'topics': ['开发工具']}, sent_as='image',
            published_at=self.now-timedelta(hours=8), briefing_date='2026-09-11')

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    async def pending(self, now=None):
        return await self.manager.get_pending_briefing_for_news(71, published_after=self.cutoff, now=now or self.now)

    async def test_legacy_skip_can_recover_but_sent_records_cannot(self):
        brief = await self.pending()
        await self.manager.save_news_broadcast(brief['id'],71,{'items':[]},sent_as='skipped_no_news')
        self.assertIsNotNone(await self.pending())
        await self.manager.save_news_broadcast(brief['id'],71,{'items':[{'url':'https://example.com/a'}]},sent_as='image')
        self.assertIsNone(await self.pending())

    async def test_retries_wait_survive_restart_and_stop_at_limit(self):
        for attempt in range(3):
            now = self.now+timedelta(minutes=90*attempt)
            brief = await self.pending(now)
            self.assertEqual(brief['news_attempts'], attempt)
            with patch.object(news, 'memory_manager', self.manager), \
                 patch.object(news.config, 'news_max_attempts', 3), \
                 patch.object(news.config, 'news_retry_interval_minutes', 90):
                await news._record_news_retry(brief, now=now, reason='verification_failed')
            self.assertIsNone(await self.pending(now+timedelta(minutes=89)))
            self.assertFalse(await self.manager.has_pending_briefing_for_news(published_after=self.cutoff, now=now))
            await self.manager.close()
            self.manager = object.__new__(MemoryManager)
            self.manager._init()
            self.manager._db_path = Path(self.temp.name) / 'history.db'
        self.assertIsNone(await self.pending(self.now+timedelta(hours=10)))

    async def test_only_latest_briefing_is_eligible_and_context_stays_in_group(self):
        await self.manager.save_published_briefing(71, {'topics':['游戏硬件'],'active_members':['private name']},
            sent_as='image', published_at=self.now, briefing_date='2026-09-12')
        await self.manager.save_published_briefing(72, {'topics':['other group']},
            sent_as='image', published_at=self.now, briefing_date='2026-09-12')
        brief=await self.pending()
        self.assertEqual(brief['briefing_date'], '2026-09-12')
        await self.manager.save_news_broadcast(brief['id'],71,{'items':[]},sent_as='retry_exhausted')
        self.assertIsNone(await self.pending())
        topics=await self.manager.recent_news_topics(71,since=self.cutoff,before=self.now)
        self.assertEqual(topics,['游戏硬件','开发工具'])

    async def test_failed_generation_schedules_retry_without_sending(self):
        bot=type('Bot', (), {'send_group_msg': AsyncMock()})()
        with patch.object(news, 'memory_manager', self.manager), \
             patch.object(news, 'ensure_send_allowed'), \
             patch.object(news, 'generate_news_payload', AsyncMock(return_value=None)) as generate:
            await news.run_daily_news_broadcast(bot,[{'group_id':71}],now=self.now)
        self.assertIsNone(await self.pending())
        self.assertIsNotNone(await self.pending(self.now+timedelta(hours=2)))
        self.assertEqual(generate.await_args.kwargs['recent_topics'], ['开发工具'])
        bot.send_group_msg.assert_not_awaited()

    async def test_broad_plan_receives_recent_topics_and_has_no_generated_opening(self):
        response='{"intro":"大家最近都在关注……","interests":[{"label":"游戏与硬件","query":"游戏 新作 硬件"}]}'
        with patch.object(news.model_manager, 'chat', AsyncMock(return_value=response)) as chat:
            result=await news.extract_news_plan({'topics':['某游戏一闪技巧'],'active_members':['private name']},
                recent_topics=['显卡驱动','另一款游戏'])
        prompt='\n'.join(m.content for m in chat.await_args.args[0])
        self.assertIn('另一款游戏',prompt)
        self.assertIn('不必',prompt)
        self.assertNotIn('private name',prompt)
        self.assertEqual(result['intro'],'')
        self.assertEqual(result['interests'][0]['label'],'游戏与硬件')
