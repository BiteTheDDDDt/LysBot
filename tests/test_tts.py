import base64
import io
import json
import os
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import nonebot


nonebot.init()

from src.plugins.llm_chat import chat, tools, tts
from src.plugins.llm_chat.config import TOOL_USAGE_PROMPT, config
from src.plugins.llm_chat.providers import supports_tts
from src.plugins.llm_chat.roles import resolve_role


class GeminiTTSTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        tools.clear_voice_context()
        tools._voice_last_sent_at.clear()

    def test_pcm_is_wrapped_as_24khz_mono_wav(self):
        pcm = b"\x01\x02" * 240

        wav_bytes = tts._pcm_to_wav(pcm, sample_rate=24000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 24000)
            self.assertEqual(wav_file.readframes(wav_file.getnframes()), pcm)

    async def test_generate_speech_uses_native_google_audio_endpoint(self):
        captured = {}
        pcm = b"\x00\x01" * 120

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["api_key"] = request.headers.get("x-goog-api-key")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "inlineData": {
                                    "mimeType": "audio/L16;codec=pcm;rate=24000",
                                    "data": base64.b64encode(pcm).decode("ascii"),
                                }
                            }]
                        }
                    }],
                    "usageMetadata": {"candidatesTokenCount": 25},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with (
                patch.object(config, "google_api_key", "test-key"),
                patch.object(config, "google_tts_base_url", "https://example.test/v1beta"),
                patch.object(config, "google_tts_voice", "Leda"),
                patch.dict(os.environ, {"MODEL_TTS": "google:gemini-3.1-flash-tts-preview"}),
            ):
                result = await tts.generate_speech("你好", "cheerful", client=client)

        self.assertEqual(
            captured["url"],
            "https://example.test/v1beta/models/gemini-3.1-flash-tts-preview:generateContent",
        )
        self.assertEqual(captured["api_key"], "test-key")
        prompt = captured["body"]["contents"][0]["parts"][0]["text"]
        self.assertIn("你好", prompt)
        self.assertIn("A youthful, androgynous, bright and crystal-clear voice", prompt)
        self.assertIn("Let a smile enter the voice", prompt)
        self.assertIn("must not replace the core vocal identity", prompt)
        self.assertNotIn("Role: 磷叶石", prompt)
        self.assertTrue(result.wav_bytes.startswith(b"RIFF"))
        self.assertEqual(result.source_mime_type, "audio/L16;codec=pcm;rate=24000")
        self.assertEqual(result.model_spec, "google:gemini-3.1-flash-tts-preview")

    def test_custom_voice_persona_replaces_builtin_profile(self):
        prompt = tts._build_tts_prompt("测试", "neutral", "A custom bright voice.")

        self.assertIn("A custom bright voice.", prompt)
        self.assertNotIn("intensely curious", prompt)
        self.assertIn("<TRANSCRIPT>\n测试\n</TRANSCRIPT>", prompt)

    async def test_role_tts_falls_back_only_to_explicit_tts_chain(self):
        calls = []
        pcm = b"\x00\x01" * 40

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if "primary-tts" in str(request.url):
                return httpx.Response(500, json={"error": {"message": "temporary"}})
            return httpx.Response(
                200,
                json={
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "inlineData": {
                                    "mimeType": "audio/L16;rate=24000",
                                    "data": base64.b64encode(pcm).decode("ascii"),
                                }
                            }]
                        }
                    }]
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with (
                patch.object(config, "google_api_key", "test-key"),
                patch.object(config, "google_tts_base_url", "https://example.test/v1beta"),
                patch.object(config, "google_tts_voice", "Leda"),
                patch.dict(
                    os.environ,
                    {
                        "MODEL_TTS": "google:primary-tts | google:fallback-tts",
                        "MODEL_FALLBACK": "nvidia:not-an-audio-model",
                    },
                ),
            ):
                result = await tts.generate_speech("回退测试", client=client)
                primary, fallback = resolve_role("tts")

        self.assertEqual(primary, "google:primary-tts")
        self.assertEqual(fallback, ["google:fallback-tts"])
        self.assertEqual(result.model_spec, "google:fallback-tts")
        self.assertEqual(len(calls), 2)

    def test_tts_capability_is_separate_from_text_models(self):
        self.assertTrue(supports_tts("google:gemini-3.1-flash-tts-preview"))
        self.assertFalse(supports_tts("google:gemini-3.1-flash-lite"))
        self.assertFalse(supports_tts("nvidia:some-tts-model"))

    async def test_send_voice_sends_record_and_records_round_state(self):
        bot = SimpleNamespace(
            send_group_msg=AsyncMock(),
            send_private_msg=AsyncMock(return_value={"message_id": 123}),
        )
        tools.set_voice_context(group_id=None, user_id=42, bot=bot)
        generated = tts.GeneratedSpeech(
            wav_bytes=tts._pcm_to_wav(b"\x00\x00" * 20),
            source_mime_type="audio/L16;rate=24000",
        )

        with (
            patch.dict("os.environ", {"MODEL_TTS": "google:test-tts"}),
            patch.object(config, "google_tts_max_chars", 200),
            patch.object(config, "google_tts_cooldown", 0),
            patch.object(tts, "generate_speech", AsyncMock(return_value=generated)) as generate_mock,
        ):
            result = await tools.send_voice("这句只发语音", emotion="neutral")

        tool_result = json.loads(result)
        self.assertEqual(tool_result["status"], "sent")
        self.assertEqual(tool_result["media_type"], "voice")
        self.assertEqual(tool_result["transcript"], "这句只发语音")
        generate_mock.assert_awaited_once_with("这句只发语音", emotion="neutral")
        bot.send_private_msg.assert_awaited_once()
        sent_message = bot.send_private_msg.await_args.kwargs["message"]
        self.assertEqual(sent_message[0].type, "record")
        self.assertTrue(tools.voice_was_sent_this_round())
        memory_text = tools.format_sent_voice_for_memory(tools.get_last_sent_voice())
        self.assertIn("这句只发语音", memory_text)
        self.assertIn("【生成语音", memory_text)
        self.assertIn("实际台词：这句只发语音", memory_text)

    async def test_send_voice_uses_group_record_api_in_group_context(self):
        bot = SimpleNamespace(
            send_group_msg=AsyncMock(return_value={"message_id": 456}),
            send_private_msg=AsyncMock(),
        )
        tools.set_voice_context(group_id=99, user_id=42, bot=bot)
        generated = tts.GeneratedSpeech(
            wav_bytes=tts._pcm_to_wav(b"\x00\x00" * 20),
            source_mime_type="audio/L16;rate=24000",
        )

        with (
            patch.dict("os.environ", {"MODEL_TTS": "google:test-tts"}),
            patch.object(config, "google_tts_max_chars", 200),
            patch.object(config, "google_tts_cooldown", 0),
            patch.object(tts, "generate_speech", AsyncMock(return_value=generated)),
        ):
            await tools.send_voice("群里的语音", emotion="cheerful")

        bot.send_group_msg.assert_awaited_once()
        self.assertEqual(bot.send_group_msg.await_args.kwargs["group_id"], 99)
        bot.send_private_msg.assert_not_awaited()

    async def test_voice_tool_turn_sends_and_records_non_repeating_followup(self):
        tools.set_voice_context(group_id=99, user_id=42, bot=object())
        context = tools._voice_context.get()
        context["sent_in_round"] = True
        context["last_sent"] = {"text": "记到聊天历史", "emotion": "neutral"}
        tools._voice_context.set(context)

        add_message = AsyncMock()
        send_followup = AsyncMock()
        with (
            patch.object(chat.memory_manager, "add_message", add_message),
            patch.object(chat.proactive_manager, "mark_just_spoke") as mark_spoke,
        ):
            finalized = await chat._finalize_voice_tool_turn(
                user_id=42,
                group_id=99,
                followup_text="只准听一遍。",
                send_followup=send_followup,
            )

        self.assertTrue(finalized)
        add_message.assert_awaited_once()
        self.assertIn("记到聊天历史", add_message.await_args.args[2])
        self.assertIn("只准听一遍", add_message.await_args.args[2])
        send_followup.assert_awaited_once_with("只准听一遍")
        mark_spoke.assert_called_once_with(99)

    def test_voice_tool_is_exposed_to_chat_models(self):
        names = {item["function"]["name"] for item in tools.AVAILABLE_TOOLS}
        self.assertIn("send_voice", names)

    def test_voice_tool_instructions_cover_implicit_audio_requests(self):
        description = tools.VOICE_TOOL["function"]["description"]

        self.assertIn("想听你的声音", description)
        self.assertIn("哄我睡觉", description)
        self.assertIn("普通的‘你说说’", description)
        self.assertIn("不要求用户必须说出“生成语音”", TOOL_USAGE_PROMPT)


if __name__ == "__main__":
    unittest.main()
