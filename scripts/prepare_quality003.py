"""Rebuild the reviewed 003 pilot from original example/train photos, without modifying them."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(source, study, dest):
    originals = {r['image']: r for r in map(json.loads, (source/'ratings.jsonl').read_text().splitlines())}
    rows = list(map(json.loads, (study/'ratings.jsonl').read_text().splitlines()))
    hashes = json.loads((study/'image-sha256.json').read_text())
    for row in rows:
        name = row['image']
        if row['group'] != originals[name]['group'] or hashlib.sha256((source/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'original group or bytes changed: {name}')
    dest.mkdir(parents=True, exist_ok=False)
    for row in rows:
        name = row['image']; (dest/name).parent.mkdir(exist_ok=True)
        shutil.copyfile(source/name, dest/name)
    for name in ('axes.yaml', 'ratings.jsonl'):
        shutil.copyfile(study/name, dest/name)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k, d in [('source','datasets/example/train'),('study','studies/overnight/003'),('dest','work/quality003')]:
        p.add_argument('--'+k,type=Path,default=Path(d))
    prepare(**vars(p.parse_args()))
