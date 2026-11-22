"""Assemble a separate 008 bank; never edit or traverse the source dataset."""
import hashlib
import json
import shutil
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[3]
    study = Path(__file__).resolve().parent
    source = root/'datasets/example'
    dest = root/'work/quality008/bank'
    images = dest/'train/images'
    images.mkdir(parents=True, exist_ok=False)
    spec = json.loads((study.parent/'007/protocol.json').read_text())
    sha = lambda b: hashlib.sha256(b).hexdigest()
    original = (source/'train/ratings.jsonl').read_bytes()
    axes = (source/'axes.yaml').read_bytes()
    assert sha(original) == spec['inputs']['ratings.jsonl'] and sha(axes) == spec['inputs']['axes.yaml']
    keys = list(yaml.safe_load(axes))
    fit = {r['image']: r for r in map(json.loads, (study/'fit-ratings.jsonl').read_text().splitlines())}
    assert len(fit) == 16 == len((study/'fit-ratings.jsonl').read_text().splitlines())
    lines = []
    for line in original.splitlines(keepends=True):
        r = json.loads(line); n = r['image']; p = source/'train'/n
        assert sha(p.read_bytes()) == spec['photos'][n]
        shutil.copyfile(p, dest/'train'/n)
        if r['group'] in spec['dev_groups']:
            assert n not in fit
            lines.append(line)  # exact bytes, including original nulls
        else:
            new = fit.pop(n)
            assert set(new) == set(r) and new['group'] == r['group'] and list(new['scores']) == keys
            lines.append((json.dumps(new)+'\n').encode())
    assert not fit and len(lines) == len(spec['photos']) == 24
    joined = b''.join(lines)
    (dest/'train/ratings.jsonl').write_bytes(joined)
    (dest/'axes.yaml').write_bytes(axes)
    spec['inputs']['ratings.jsonl'] = sha(joined)
    assert spec == json.loads((study/'protocol.json').read_text()), '008 protocol mismatch'


if __name__ == '__main__':
    main()
