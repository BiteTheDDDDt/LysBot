"""AI-operated NapCat login/status/connection setup. Never print credentials.

Uses the NapCat WebUI backend API. Reference:
https://github.com/NapNeko/NapCatQQ/tree/main/packages/napcat-webui-backend/src
"""
import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import httpx
from dotenv import dotenv_values


class NapCatError(RuntimeError):
    pass


class NapCatClient:
    def __init__(self, url, token, *, transport=None):
        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=15, trust_env=False, transport=transport)
        self.token = token

    def close(self):
        self.client.close()

    def request(self, route, payload=None):
        response = self.client.post("/api/" + route, json=payload or {})
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise NapCatError("napcat_unexpected_response")
        if body.get("code") != 0:
            raise NapCatError("napcat_api_rejected")
        return body.get("data")

    def login(self):
        hashed = hashlib.sha256((self.token + ".napcat").encode()).hexdigest()
        data = self.request("auth/login", {"hash": hashed})
        if not isinstance(data, dict):
            raise NapCatError("napcat_unexpected_response")
        if data.get("require2FA"):
            raise NapCatError("napcat_requires_2fa")
        credential = data.get("Credential")
        if not credential:
            raise NapCatError("napcat_credential_missing")
        self.client.headers["Authorization"] = "Bearer " + credential

    def status(self):
        state = self.request("QQLogin/CheckLoginStatus")
        if not isinstance(state, dict):
            raise NapCatError("napcat_unexpected_response")
        return {"logged_in": state.get("isLogin") is True,
                "offline": state.get("isOffline") is True,
                "core_ready": state.get("coreReady") is True,
                "phase": state.get("loginPhase")}

    def qr(self, output):
        import qrcode
        data = self.request("QQLogin/GetQQLoginQrcode")
        if not isinstance(data, dict):
            raise NapCatError("napcat_unexpected_response")
        value = data.get("qrcode")
        if not isinstance(value, str) or not value:
            raise NapCatError("napcat_qr_not_ready")
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        qrcode.make(value).save(output)
        output.chmod(0o600)
        return {"path": str(output), "needs_user_action": "scan_in_qq"}

    def connect(self, url, token):
        if not self.status()["logged_in"]:
            return {"changed": False, "needs_user_action": "scan_in_qq"}
        current = self.request("OB11Config/GetConfig")
        desired = connection_config(current, url, token)
        if current == desired:
            return {"changed": False, "configured": True}
        self.request("OB11Config/SetConfig", {"config": json.dumps(desired, ensure_ascii=False)})
        actual = self.request("OB11Config/GetConfig")
        if not isinstance(actual, dict):
            raise NapCatError("napcat_unexpected_configuration")
        expected = next(item for item in desired["network"]["websocketClients"] if item["name"] == "lysbot")
        if not any(all(item.get(key) == value for key, value in expected.items())
                   for item in actual.get("network", {}).get("websocketClients", [])):
            raise NapCatError("napcat_configuration_not_verified")
        return {"changed": True, "configured": True}


def connection_config(current, url, token):
    if not isinstance(current, dict):
        raise NapCatError("napcat_unexpected_configuration")
    if not token or token.startswith("replace-with"):
        raise NapCatError("onebot_token_missing_or_placeholder")
    if not url.startswith(("ws://", "wss://")):
        raise NapCatError("invalid_websocket_url")
    desired = deepcopy(current)
    network = desired.setdefault("network", {})
    if not isinstance(network, dict):
        raise NapCatError("napcat_unexpected_configuration")
    clients = network.setdefault("websocketClients", [])
    if not isinstance(clients, list) or any(not isinstance(item, dict) for item in clients):
        raise NapCatError("napcat_unexpected_configuration")
    owned = {"name": "lysbot", "enable": True, "url": url, "messagePostFormat": "array",
             "reportSelfMessage": False, "reconnectInterval": 5000, "token": token,
             "debug": False, "heartInterval": 30000}
    network["websocketClients"] = [item for item in clients if item.get("name") != "lysbot" and item.get("url") != url] + [owned]
    return desired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "qr", "connect"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--url")
    parser.add_argument("--ws-url", default="ws://bot:8080/onebot/v11/ws")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true", help="JSON is the default output format")
    args = parser.parse_args()
    root = args.root.resolve()
    values = {**dotenv_values(root / ".env"), **os.environ}
    data_dir = Path(values.get("LLM_CHAT_DATA_DIR") or root / "data")
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    client = None
    try:
        settings = json.loads((data_dir / "napcat/config/webui.json").read_text(encoding="utf-8"))
        token = settings.get("token")
        if not token:
            raise NapCatError("napcat_webui_token_missing")
        default_url = "http://napcat:6099" if Path("/.dockerenv").exists() else "http://127.0.0.1:16099"
        client = NapCatClient(args.url or default_url, token)
        client.login()
        if args.action == "status":
            data = client.status()
        elif args.action == "qr":
            data = client.qr(args.output or data_dir / "private/qq-login.png")
        else:
            data = client.connect(args.ws_url, str(values.get("ACCESS_TOKEN") or ""))
        result = {"schema_version": 1, "ok": True, "action": args.action, **data}
        code = 2 if data.get("needs_user_action") else 0
    except (OSError, ValueError, httpx.HTTPError, NapCatError) as error:
        error_code = str(error) if isinstance(error, NapCatError) else type(error).__name__
        result = {"schema_version": 1, "ok": False, "action": args.action, "code": error_code,
                  "hint": "Check the NapCat service, private webui.json and API version; do not print tokens"}
        code = 1
    finally:
        if client:
            client.close()
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
