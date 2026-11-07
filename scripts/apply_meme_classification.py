"""Apply a completed visual review with a reversible metadata journal; never delete images."""
import argparse
import json
from pathlib import Path
import sqlite3
from contextlib import closing

FIELDS = ['meme_kind', 'classification_reason', 'text_sentence_count', 'random_eligible']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--restore', action='store_true')
    args = p.parse_args()
    journal = args.report_dir / 'classification-backup.json'
    results = [] if args.restore else [json.loads(line) for line in (args.report_dir / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    if not args.restore:
        manifest = json.loads((args.report_dir / 'manifest.json').read_text(encoding='utf-8'))
        images = manifest['images'] if isinstance(manifest, dict) else manifest
        if not {r['hash'] for r in images}.issubset({r['hash'] for r in results}):
            raise ValueError('Review incomplete')
        if journal.exists():
            raise ValueError('Backup already exists; do not overwrite an applied review')
    with closing(sqlite3.connect(args.database, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        cols = {r[1] for r in db.execute('PRAGMA table_info(memes)')}
        for name, typ in [('meme_kind', 'TEXT'), ('classification_reason', 'TEXT'), ('text_sentence_count', 'INTEGER')]:
            if name not in cols:
                db.execute(f'ALTER TABLE memes ADD COLUMN {name} {typ}')
        db.commit()
        db.execute('BEGIN IMMEDIATE')
        if args.restore:
            backup = json.loads(journal.read_text(encoding='utf-8'))
            for row in backup:
                db.execute('UPDATE memes SET meme_kind=?,classification_reason=?,text_sentence_count=?,random_eligible=? WHERE id=? AND hash=?',
                           (*[row[f] for f in FIELDS], row['id'], row['hash']))
        else:
            backup = []
            for result in results:
                old = db.execute('SELECT id,hash,meme_kind,classification_reason,text_sentence_count,random_eligible FROM memes WHERE id=? AND hash=?',
                                 (result['id'], result['hash'])).fetchone()
                if not old:
                    continue
                count = result.get('text_sentence_count')
                kind = result.get('meme_kind')
                valid = (result.get('is_meme') is True and kind in {'reaction','joke'}
                         and type(count) is int and 0 <= count < 3)
                eligible = int(valid and result.get('random_eligible') is True)
                backup.append(dict(old))
                db.execute('UPDATE memes SET meme_kind=?,classification_reason=?,text_sentence_count=?,random_eligible=? WHERE id=? AND hash=?',
                           (kind if valid else None, result.get('reason') or '未确认有效分类，停止随机发送', count,
                            eligible, old['id'], old['hash']))
            # A crash before commit leaves only an unused backup, not unjournaled changes.
            journal.write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding='utf-8')
        db.commit()
        print(json.dumps({'reviewed_existing_rows': len(backup), 'categories': [list(r) for r in db.execute('SELECT meme_kind,random_eligible,count(*) FROM memes GROUP BY meme_kind,random_eligible')]},ensure_ascii=False))


if __name__ == '__main__':
    main()
