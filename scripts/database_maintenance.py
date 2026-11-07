"""Inspect/snapshot SQLite; attempt index repair only on a newly created copy.

No subcommand replaces a running database or deletes application records.
"""
import argparse
import json
from pathlib import Path
import sqlite3
from contextlib import closing


def connect_readonly(path: Path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    db.execute("PRAGMA query_only=ON")
    return db


def integrity(path: Path) -> list[str]:
    with closing(connect_readonly(path)) as db:
        return [row[0] for row in db.execute("PRAGMA integrity_check")]


def table_counts(path: Path) -> dict:
    with closing(connect_readonly(path)) as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        result = {}
        for name in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            try:
                result[name] = db.execute(f"SELECT COUNT(*) FROM {quoted} NOT INDEXED").fetchone()[0]
            except sqlite3.DatabaseError:
                result[name] = None
        return result


def snapshot(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing snapshots and the live database.
    with destination.open("xb"):
        pass
    with closing(connect_readonly(source)) as original, closing(sqlite3.connect(destination)) as copy:
        original.backup(copy, pages=256, sleep=0.05)


def reindex_copy(source: Path, destination: Path) -> dict:
    snapshot(source, destination)
    before = table_counts(destination)
    with closing(sqlite3.connect(destination)) as db:
        db.execute("REINDEX")
        db.commit()
    checks = integrity(destination)
    after = table_counts(destination)
    valid = checks == ["ok"] and before == after and all(value is not None for value in after.values())
    return {"valid": valid, "integrity": checks, "counts_before": before, "counts_after": after}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "snapshot", "reindex-copy"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        result = {"integrity": integrity(args.source), "counts": table_counts(args.source)}
    else:
        if args.destination is None:
            parser.error("destination is required")
        if args.action == "snapshot":
            snapshot(args.source, args.destination)
            result = {"snapshot": str(args.destination), "integrity": integrity(args.destination)}
        else:
            result = reindex_copy(args.source, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("valid", result.get("integrity") == ["ok"]) else 1)
