"""Exercise real LysBot startup with a local model stub and simulated OneBot.

No QQ login, external model, production configuration or production database.
"""
import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websockets


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def exercise(root, directory, model_port, requests):
    port = unused_port()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MODEL_", "ROLE_", "PROVIDER_PROXY_"))}
    environment.update(
        PERSONA_FILE="personas/phosphophyllite.toml",
        KNOWLEDGE_DIR="knowledge",
        HTTP_PROXY="", HTTPS_PROXY="", ALL_PROXY="", NO_PROXY="127.0.0.1,localhost",
        MASTER_QQ="42", ACCESS_TOKEN="smoke-token", HOST="127.0.0.1", PORT=str(port),
        ONEBOT_PRIMARY_QQ="", ONEBOT_STANDBY_QQ="",
        MODEL_GENERAL="openai:smoke-general", OPENAI_API_KEY="smoke-not-a-real-key",
        OPENAI_BASE_URL=f"http://127.0.0.1:{model_port}/v1", PYTHONUTF8="1",
        LLM_CHAT_DATA_DIR=str(directory / "data"),
        LLM_CHAT_DATABASE_PATH=str(directory / "data/history.db"),
        AGGREGATION_WINDOW="0",
        ENABLE_IMPORTANT_INFO="false",
        ENABLE_DAILY_BRIEFING="false", ENABLE_DAILY_NEWS_BROADCAST="false",
        ENABLE_MODEL_CARD="false", NETWORK_MAX_RETRIES="0",
    )
    checkout = directory / "checkout"
    checkout.mkdir()
    for name in ("src", "lysbot", "personas"):
        shutil.copytree(root / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__"))
    (checkout / "knowledge").mkdir()
    shutil.copy2(root / "knowledge/project.md", checkout / "knowledge/project.md")
    for name in ("bot.py", "pyproject.toml"):
        shutil.copy2(root / name, checkout / name)
    extension_dir = checkout / "extensions"
    extension_dir.mkdir()
    (extension_dir / "smoke_extension.py").write_text(
        'from lysbot import register_tool\n'
        'async def smoke_owner(context):\n'
        '    assert context.is_master\n'
        '    return "extension_owner_verified"\n'
        'register_tool("smoke_owner", "Verify owner", {"type": "object", "properties": {}}, '
        'smoke_owner, master_only=True, read_only=True)\n', encoding="utf-8"
    )
    log_path = directory / "bot.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "bot.py", cwd=checkout, env=environment, stdout=log, stderr=asyncio.subprocess.STDOUT
        )
        try:
            for _ in range(150):
                if process.returncode is not None:
                    raise RuntimeError("Bot exited during startup")
                try:
                    _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(0.1)
            else:
                raise TimeoutError("Bot listener did not start")

            async with websockets.connect(
                f"ws://127.0.0.1:{port}/onebot/v11/ws",
                additional_headers={"X-Self-ID": "999", "X-Client-Role": "Universal",
                                    "Authorization": "Bearer smoke-token"},
                proxy=None,
            ) as connection:
                event = {
                    "time": int(time.time()), "self_id": 999, "post_type": "message",
                    "message_type": "group", "sub_type": "normal", "message_id": 1,
                    "group_id": 7, "user_id": 42, "anonymous": None,
                    "message": [{"type": "at", "data": {"qq": "999"}},
                                {"type": "text", "data": {"text": "你好，你当前的群名片是什么"}}],
                    "raw_message": "[CQ:at,qq=999]你好，你当前的群名片是什么", "font": 0,
                    "sender": {"user_id": 42, "nickname": "测试成员", "card": "测试成员",
                               "sex": "unknown", "age": 0, "area": "", "level": "",
                               "role": "member", "title": ""},
                }
                health = directory / "data" / "health.json"

                async def send_when_online():
                    while not (health.exists() and json.loads(health.read_text(encoding="utf-8")).get("qq_online") is True):
                        await asyncio.sleep(0.1)
                    await connection.send(json.dumps(event))

                reply = None
                async with asyncio.timeout(35), asyncio.TaskGroup() as traffic:
                    traffic.create_task(send_when_online())
                    while reply is None:
                        action = json.loads(await connection.recv())
                        name = action.get("action")
                        data = {}
                        if name == "get_login_info":
                            data = {"user_id": 999, "nickname": "磷叶石"}
                        elif name == "get_status":
                            data = {"online": True, "good": True}
                        elif name == "get_group_list":
                            data = [{"group_id": 7, "group_name": "测试群"}]
                        elif name == "get_group_member_info":
                            params = action["params"]
                            if int(params["user_id"]) == 999:
                                assert int(params["group_id"]) == 7 and params["no_cache"] is True
                                data = {"user_id": 999, "nickname": "磷叶石", "card": "磷叶石 [smoke-live-model]", "role": "member"}
                            else:
                                data = {"user_id": 42, "nickname": "测试成员", "card": "测试成员"}
                        elif name == "get_group_member_list":
                            data = [{"user_id": 42, "nickname": "测试成员", "card": "测试成员"}]
                        elif name in {"send_group_msg", "send_msg"}:
                            data = {"message_id": 123}
                            reply = action.get("params", {}).get("message")
                        await connection.send(json.dumps({"status": "ok", "retcode": 0,
                                                          "data": data, "echo": action.get("echo")}))
                assert "你好" in json.dumps(reply, ensure_ascii=False), "Expected generated greeting"
                assert "磷叶石 [smoke-live-model]" in json.dumps(reply, ensure_ascii=False)
                assert any(message.get("role") == "tool" and "smoke-live-model" in str(message.get("content"))
                           for item in requests for message in item.get("messages", [])), "Live group facts were not read"
                assert requests, "Model endpoint was not called"
                assert all(item["model"] == "smoke-general" for item in requests)
                for item in requests:
                    names = {tool["function"]["name"] for tool in item.get("tools", [])}
                    assert not names & {"search_web", "send_voice", "generate_image"}
                assert any(message.get("role") == "tool" and "extension_owner_verified" in str(message.get("content"))
                           for item in requests for message in item.get("messages", [])), "Extension was not executed"
                await connection.send(json.dumps({
                    "time": int(time.time()), "self_id": 999, "post_type": "notice",
                    "notice_type": "bot_offline", "user_id": 999, "tag": "smoke", "message": "test offline",
                }))
                async with asyncio.timeout(20):
                    while True:
                        snapshot = json.loads(health.read_text(encoding="utf-8"))
                        if snapshot["account_statuses"].get("999") == "blocked":
                            break
                        await asyncio.sleep(0.1)
                assert snapshot["qq_online"] is False
                state = directory / "data" / "onebot-accounts.json"
                assert "999" in json.loads(state.read_text(encoding="utf-8"))["blocked"]
                print(f"PASS: real startup, OneBot message/reply, {len(requests)} general-model calls, live group card read, master extension executed, forced offline persisted")
        except Exception:
            print(f"Smoke-test log: {log_path}")
            print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
            raise
        finally:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=5)


def main():
    root = Path(__file__).resolve().parents[1]
    requests = []

    class ModelStub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            messages = payload.get("messages", [])
            text = "\n".join(str(message.get("content", "")) for message in messages)
            has_tool_results = any(message.get("role") == "tool" for message in messages)
            is_execution_plan = has_tool_results or any(
                tool.get("function", {}).get("name") == "smoke_owner" for tool in payload.get("tools", [])
            )
            if "参与规划器" in text:
                content = json.dumps({"action": "reply", "reply_mode": "sentence",
                                      "parts": [{"type": "text"}], "reason": "回应问候"})
            elif is_execution_plan:
                content = json.dumps({"response_mode": "factual", "verify_group_identity": False,
                                      "content": "回复问候，并给出 QQ 返回的当前群名片 磷叶石 [smoke-live-model]",
                                      "reply_mode": "sentence", "text_parts": 1,
                                      "reply_requirements": [], "should_reply": True})
            else:
                content = "你好，我现在的群名片是磷叶石 [smoke-live-model]"
            message = {"role": "assistant", "content": content}
            if is_execution_plan and not has_tool_results:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "smoke-tool-call", "type": "function",
                    "function": {"name": "smoke_owner", "arguments": "{}"},
                }, {
                    "id": "smoke-group-facts", "type": "function",
                    "function": {"name": "get_group_facts", "arguments": '{"scope":"bot"}'},
                }]}
            response = json.dumps({
                "id": "smoke-completion", "object": "chat.completion", "created": int(time.time()),
                "model": "smoke-general", "choices": [{"index": 0, "finish_reason": "stop",
                "message": message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="lysbot-smoke-") as directory:
            asyncio.run(exercise(root, Path(directory), server.server_port, requests))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
