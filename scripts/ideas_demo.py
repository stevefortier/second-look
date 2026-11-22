#!/usr/bin/env python3
"""Restore the photos of datasets/ideas-birds from COCO; they are not kept in Git.

Every file is checked against the SHA-256 in photos.jsonl. Existing correct files are kept.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path('datasets/ideas-birds'))
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in (args.root / 'photos.jsonl').read_text().splitlines() if line.strip()]
    fetched = 0
    for row in rows:
        path = args.root / row['file']
        if path.resolve().parent.parent != args.root.resolve():
            sys.exit(f"unexpected path in photos.jsonl: {row['file']}")
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']:
            continue
        with urlopen(row['url'], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            sys.exit(f"{row['url']} no longer matches its recorded SHA-256")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        fetched += 1
    print(f'{len(rows)} photos verified in {args.root} ({fetched} downloaded)')


if __name__ == '__main__':
    main()
