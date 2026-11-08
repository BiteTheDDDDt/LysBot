"""Run unittest with temporary data, dummy credentials and disabled HTTP transport."""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    # The parent process also prints UTF-8 child logs on Windows failure paths.
    # Setting PYTHONUTF8 only in the child's environment leaves the parent on
    # cp1252 and can hide the actual test failure with UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--pattern", default="test*.py")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = Path(tempfile.mkdtemp(prefix="lysbot-tests-"))
    env = os.environ.copy()
    env.update(LLM_CHAT_DATA_DIR=str(output / "data"), LLM_CHAT_DATABASE_PATH=str(output / "data/history.db"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               MEMORY_INDEX_ENABLED="false", MEMORY_EMBEDDING_MODEL="dashscope:text-embedding-v4",
               MEMORY_EMBEDDING_REVISION="v4", MEMORY_EMBEDDING_DIMENSIONS="512")
    # Retry behavior is tested explicitly; unrelated failure tests must not sleep or consume mock outcomes twice.
    env.update(NETWORK_MAX_RETRIES="0")
    # Test-only fixtures keep concrete model names out of production routing code.
    tree = ast.parse((project / "tests/model_env.py").read_text(encoding="utf-8"))
    defaults = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(
                        getattr(target, "id", "") == "TEST_MODEL_ENV" for target in node.targets))
    for key in list(env):
        if key.startswith("ROLE_") or key == "MODEL_IMAGE_DRIVER":
            del env[key]
    env.update(defaults)
    for provider in ("deepseek", "openai", "cliproxyapi", "nvidia", "dashscope", "google", "openrouter",
                     "siliconflow", "tencentmaas", "anthropic", "volc", "xiaomimimo"):
        env[f"{provider.upper()}_API_KEY"] = "test-only-not-a-real-key"
    script = """
import sys, unittest, importlib.util
from contextlib import ExitStack
from unittest.mock import patch
# Each IsolatedAsyncioTestCase owns a loop. Drain shared usage writers before
# that loop closes, otherwise aiosqlite can retain a dead worker across cases.
original_teardown = unittest.IsolatedAsyncioTestCase._callTearDown
async def close_test_usage():
    logger = sys.modules.get('src.plugins.llm_chat.llm_logger')
    usage = sys.modules.get('src.plugins.llm_chat.usage_db')
    if logger is not None:
        await logger.drain_usage_tasks()
    if usage is not None:
        await usage.close_db()
def teardown_with_usage(self):
    try:
        original_teardown(self)
    finally:
        self._callAsync(close_test_usage)
unittest.IsolatedAsyncioTestCase._callTearDown = teardown_with_usage
with ExitStack() as stack:
    for package in ('httpx', 'httpx2'):
        if importlib.util.find_spec(package) is None:
            continue
        transport = __import__(package)
        stack.enter_context(patch.object(transport.AsyncHTTPTransport, 'handle_async_request', side_effect=RuntimeError('HTTP disabled in tests')))
        stack.enter_context(patch.object(transport.HTTPTransport, 'handle_request', side_effect=RuntimeError('HTTP disabled in tests')))
    suite = unittest.defaultTestLoader.discover('tests', pattern=sys.argv[1])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
    log_path = output / "output.log"
    print(f"Isolated test data and logs: {output}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-c", script, args.pattern], cwd=project,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"Tests exceeded 180 seconds. Inspect {log_path}")
        return 1
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if result.returncode:
        print("\n".join(lines[-120:]))
    else:
        print("\n".join(line for line in lines if line.startswith(("Ran ", "OK"))))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
