"""Read-only, secret-free JSON diagnostics for AI-operated LysBot deployments."""
import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import time
import tomllib
from pathlib import Path

from dotenv import dotenv_values


def diagnose(root, environment=None, *, runtime=False):
    root = Path(root).resolve()
    environment = dict(os.environ if environment is None else environment)
    values = {**dotenv_values(root / ".env"), **environment}
    checks = []
    providers = runpy.run_path(str(root / "src/plugins/llm_chat/providers.py"))

    def check(name, status, code, hint=""):
        checks.append({"name": name, "status": status, "code": code, "hint": hint})

    master = str(values.get("MASTER_QQ") or "").strip()
    check("master", "ok" if master.isdecimal() and int(master) > 0 else "error",
          "configured" if master.isdecimal() and int(master) > 0 else "master_missing_or_invalid",
          "Set MASTER_QQ in .env" if not master.isdecimal() or int(master) <= 0 else "")
    token = str(values.get("ACCESS_TOKEN") or "").strip()
    token_ok = bool(token) and not token.startswith("replace-with")
    check("onebot_auth", "ok" if token_ok else "error",
          "configured" if token_ok else "onebot_token_missing_or_placeholder", "Use one shared random ACCESS_TOKEN")
    capabilities = {}
    inherited = {"CHAT", "PLANNER", "SUMMARY", "IMPORTANT_INFO"}
    for role in ("GENERAL", "CHAT", "PLANNER", "SUMMARY", "IMPORTANT_INFO", "FALLBACK",
                 "SEARCH", "VISION", "TTS", "IMAGE_GEN"):
        key = "MODEL_" + role
        raw = str(values.get(key) or "").strip()
        if role not in inherited | {"FALLBACK"}:
            capabilities[role.lower()] = bool(raw)
        if not raw:
            check(key, "error" if role == "GENERAL" else "ok",
                  "model_required" if role == "GENERAL" else "inherits_general" if role in inherited
                  else "no_fallback" if role == "FALLBACK" else "disabled")
            continue
        try:
            candidates = [providers["parse_spec"](part) for part in providers["parse_chain"](raw)]
        except ValueError:
            check(key, "error", "invalid_model_spec", "Use provider:model_id")
            continue
        compatible = {"SEARCH": "supports_web_search", "VISION": "supports_vision",
                      "TTS": "supports_tts", "IMAGE_GEN": "supports_image_generation"}.get(role)
        if compatible and not all(providers[compatible](f"{provider}:{model}") for provider, model in candidates):
            check(key, "error", "incompatible_model_capability", "Check the supported protocol/model combination")
            continue
        keys = [providers["PROVIDERS"][provider].api_key_attr.upper() for provider, _ in candidates]
        usable = [str(values.get(name) or "").strip() for name in keys]
        if not any(value and not value.startswith("your_") for value in usable):
            check(key, "error", "provider_credentials_missing_or_placeholder", "Set the configured provider API key")
        elif any(model.startswith("your-") for _, model in candidates):
            check(key, "error", "model_id_placeholder", "Replace the example model ID")
        else:
            check(key, "ok", "configured_not_network_verified")
    news = str(values.get("ENABLE_DAILY_NEWS_BROADCAST") or "false").lower() in {"1", "true", "yes"}
    briefing = str(values.get("ENABLE_DAILY_BRIEFING") or "false").lower() in {"1", "true", "yes"}
    if news and (not briefing or not capabilities["search"]):
        check("news", "error", "news_dependencies_missing", "Enable briefing and configure MODEL_SEARCH")
    persona = Path(values.get("PERSONA_FILE") or "personas/phosphophyllite.toml")
    if not persona.is_absolute():
        persona = root / persona
    try:
        with persona.open("rb") as stream:
            data = tomllib.load(stream)
        valid = all(isinstance(data.get(key), str) and data[key].strip() for key in ("name", "prompt"))
        check("persona", "ok" if valid else "error", "configured" if valid else "invalid_persona")
    except (OSError, ValueError):
        check("persona", "error", "persona_unreadable", "Check PERSONA_FILE")
    data_dir = Path(values.get("LLM_CHAT_DATA_DIR") or "data")
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    model_dir = Path(values.get("MEMORY_LOCAL_MODEL_DIR") or data_dir / "models/bge-small-zh-v1.5")
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    spec = runpy.run_path(str(root / "src/plugins/llm_chat/local_embedding.py"))
    try:
        for filename, digest in spec["FILES"].items():
            with (model_dir / filename).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("hash mismatch")
        check("embedding", "ok", "artifacts_verified")
    except (OSError, ValueError):
        check("embedding", "warning", "embedding_not_ready_keyword_available", "Run the setup service; basic chat remains available")
    for module in ("nonebot", "httpx", "onnxruntime", "qrcode"):
        available = importlib.util.find_spec(module) is not None
        check("dependency:" + module, "ok" if available else "error",
              "installed" if available else "dependency_missing", "Install this project or run inside its container")
    if runtime:
        try:
            state = json.loads((data_dir / "health.json").read_text(encoding="utf-8"))
            fresh = -10 <= time.time() - float(state["updated_at"]) <= 120
            for key in ("plugin_ready", "database_ok", "onebot_connected"):
                ready = fresh and state.get(key) is True
                check(key, "ok" if ready else "error", "ready" if ready else "not_ready")
        except (OSError, ValueError, KeyError, TypeError):
            check("runtime", "error", "heartbeat_unavailable", "Start bot, then check runtime status")
    return {"schema_version": 1, "ok": not any(item["status"] == "error" for item in checks),
            "network_verified": False, "capabilities": capabilities, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="JSON is the default output format")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    try:
        result = diagnose(args.root, runtime=args.runtime)
    except (OSError, ValueError):
        result = {"schema_version": 1, "ok": False, "checks": [
            {"name": "doctor", "status": "error", "code": "diagnostic_source_unavailable",
             "hint": "Run from a complete LysBot checkout or container"}
        ]}
    print(json.dumps(result, ensure_ascii=False))
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
