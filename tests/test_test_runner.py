import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestRunnerOutputTests(unittest.TestCase):
    def test_failed_tests_print_unicode_and_preserve_exit_code_on_legacy_console(self):
        project = Path(__file__).resolve().parents[1]
        script = """
import runpy, subprocess, sys
from unittest.mock import patch

def failed_tests(args, **kwargs):
    kwargs['stdout'].write('FAIL: 中文错误\\n')
    return subprocess.CompletedProcess(args, 7)

sys.argv = ['scripts/run_tests.py']
with patch('subprocess.run', side_effect=failed_tests):
    runpy.run_path('scripts/run_tests.py', run_name='__main__')
"""
        env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="cp1252")
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=project, env=env,
            capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 7, result.stderr.decode("utf-8", errors="replace"))
        self.assertIn("FAIL: 中文错误", result.stdout.decode("utf-8"))
        self.assertNotIn(b"UnicodeEncodeError", result.stderr)
