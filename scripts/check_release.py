"""Read-only release preflight. Report paths/categories, never matched content.

This is a small guard for known private paths and common credential formats,
not a complete secret scanner or a privacy clearance.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath

SECRET_PATTERN = re.compile(
    rb"\b(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{20,}|"
    rb"AIza[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,})\b"
    rb"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    base = path.name.lower()
    example = base == '.env.example' or (base.startswith('.env.') and base.endswith('.example'))
    return (
        (base == '.env' or base.startswith('.env.')) and not example
        or base.endswith(('.db', '.db-wal', '.db-shm', '.sqlite', '.sqlite3', '.log'))
        or bool(path.parts and path.parts[0].lower() in {'data', '.agents', '.codex', '.tmp'})
        or base == 'notes.txt'
    )


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def audit(root: Path) -> list[tuple[str, str]]:
    findings: set[tuple[str, str]] = set()
    tracked = git(root, 'ls-files', '-z').decode('utf-8').split('\0')
    for name in filter(None, tracked):
        if private_path(name):
            findings.add(('tracked-private-file', name))
            continue
        # Inspect both the index and worktree: an unstaged correction does not
        # remove a credential that is still staged for commit.
        contents = [git(root, 'show', ':' + name)]
        path = root / name
        if path.is_file():
            contents.append(path.read_bytes())
        if any(SECRET_PATTERN.search(content) for content in contents):
            findings.add(('possible-credential', name))
    # Include deleted paths, renames and branches; do not print commit messages.
    history = git(root, 'log', '--all', '--format=', '--name-only', '-z').decode('utf-8')
    for name in history.split('\0'):
        name = name.strip('\r\n')
        if name and private_path(name):
            findings.add(('historical-private-file', name))
    if not any((root / name).is_file() for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING')):
        findings.add(('missing-license', 'LICENSE'))
    return sorted(findings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        findings = audit(args.root.resolve())
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        print('Release check could not complete; verify the Git repository and readable files.')
        return 2
    for category, name in findings:
        print(f'{category}: {name}')
    print('Manual review required: chat examples, screenshots, archived diffs, assets and historical secrets.')
    print('Scope: tracked paths/content and historical private paths; no full historical content scan.')
    print(f'{len(findings)} blocking finding(s).')
    return int(bool(findings))


if __name__ == '__main__':
    raise SystemExit(main())
