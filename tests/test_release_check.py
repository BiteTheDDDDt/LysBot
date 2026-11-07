import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_release import audit, main, private_path


class ReleaseCheckTests(unittest.TestCase):
    def test_private_paths_and_safe_templates(self):
        for path in ('.env', '.env.local', 'data/history.db', 'a/history.db-wal', 'a/run.log', '.agents/task.txt'):
            self.assertTrue(private_path(path), path)
        for path in ('.env.example', '.env.quickstart.example', 'src/config.py', 'tests/fixtures/synthetic.json'):
            self.assertFalse(private_path(path), path)

    def test_index_and_history_are_checked_without_exposing_content(self):
        token = ('sk-' + 'X' * 32).encode()

        def fake_git(root, *args):
            if args[0] == 'ls-files':
                return b'.env\0config.py\0'
            if args[0] == 'show':
                return token
            return b'\ndata/history.db\0\n.env\0.env.example\0'

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.py').write_text('# corrected worktree\n', encoding='utf-8')
            with patch('scripts.check_release.git', side_effect=fake_git):
                findings = audit(root)
        self.assertIn(('tracked-private-file', '.env'), findings)
        self.assertIn(('historical-private-file', 'data/history.db'), findings)
        self.assertIn(('possible-credential', 'config.py'), findings)
        self.assertIn(('missing-license', 'LICENSE'), findings)
        self.assertNotIn(token.decode(), str(findings))
        self.assertNotIn(('historical-private-file', '.env.example'), findings)

    def test_worktree_is_checked_even_when_index_is_clean(self):
        def fake_git(root, *args):
            return b'config.py\0' if args[0] == 'ls-files' else b''

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.py').write_text('sk-' + 'Y' * 32, encoding='utf-8')
            (root / 'LICENSE').write_text('fixture', encoding='utf-8')
            with patch('scripts.check_release.git', side_effect=fake_git):
                self.assertEqual(audit(root), [('possible-credential', 'config.py')])

    def test_exit_status_reports_blockers(self):
        with patch('sys.argv', ['check_release.py']), patch('scripts.check_release.audit', return_value=[
            ('historical-private-file', '.env')
        ]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 1)
        self.assertIn('historical-private-file: .env', output.getvalue())


if __name__ == '__main__':
    unittest.main()
