"""Explicitly download and verify the pinned local embedding artifacts; runtime is offline."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/models/bge-small-zh-v1.5"))
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()
    spec = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/plugins/llm_chat/local_embedding.py"))
    args.directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(proxy=args.proxy, timeout=120, follow_redirects=True) as client:
        for filename, expected in spec["FILES"].items():
            destination = args.directory / filename
            if destination.is_file():
                with destination.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                        print(json.dumps({"file": filename, "status": "verified"}), flush=True)
                        continue
            temporary = destination.with_suffix(destination.suffix + ".partial")
            url = f"https://huggingface.co/{spec['REPOSITORY']}/resolve/{spec['REVISION']}/{filename}"
            digest = hashlib.sha256()
            size = 0
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > 150 * 1024 * 1024:
                            raise ValueError("embedding_download_too_large")
                        digest.update(chunk)
                        stream.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("embedding_download_hash_mismatch:" + filename)
            temporary.replace(destination)
            print(json.dumps({"file": filename, "status": "downloaded_and_verified", "bytes": size}), flush=True)


if __name__ == "__main__":
    main()
