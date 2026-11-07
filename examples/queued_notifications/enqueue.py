"""Standalone stdlib producer; run this file directly without loading NoneBot."""
import argparse
import json
from pathlib import Path

from storage import enqueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/queued_notifications"))
    parser.add_argument("--id", required=True, help="Stable task ID, reused when a producer retries")
    parser.add_argument("--text", required=True, help="Notification text, at most 1800 characters")
    parser.add_argument("--ttl", type=float, default=3600, help="Expires after this many seconds")
    args = parser.parse_args()
    try:
        result = enqueue(args.directory, args.id, args.text, args.ttl)
    except (ValueError, OSError):
        parser.exit(1, "Queue submission failed; check the directory, task ID, text and TTL.\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
