"""Train-only 012 entrypoint: never load the joint fit/dev bank metadata."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

import q38_quality as core

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'studies/overnight/012'
BANK = ROOT/'work/quality008/bank/train'
FULL = HERE.parent/'008/targets.json'
CUE = HERE.parent/'011/cue-impressions.json'
RATINGS = HERE.parent/'008/fit-ratings.jsonl'


def schedule(full, cue, rows):
    names = sorted(full)
    if len(names) != 16 or set(names) != set(cue) or {r['image'] for r in rows} != set(names) or len(rows) != 16:
        raise ValueError('inconsistent fit-only sources')
    endpoints = {r['image']: [(axis, 'L' if v < 0 else 'H') for axis, v in r['scores'].items()
                              if v is not None and abs(v) > .5] for r in rows}
    freq = Counter(cell for cells in endpoints.values() for cell in cells)
    weights = {n: sum(freq[cell] <= 3 for cell in endpoints[n]) for n in names}
    extra = Counter()
    for _ in range(29):
        eligible = (n for n in names if weights[n] and extra[n] < 8)
        chosen = max(eligible, key=lambda n: Fraction(weights[n], extra[n] + 1))
        extra[chosen] += 1
    items = [(n, 'caption', '', (full, cue)[(i+j) % 2][n]) for i, n in enumerate(names)
             for j in range(8 + extra[n])]
    random.Random(core.q.SEED).shuffle(items)
    return items, weights


def checksum(items):
    return hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest()


def checked(bank):
    full = core.read(FULL)
    pins = core.read(HERE/'fit-photos.json')
    if set(pins) != set(full):
        raise ValueError('fit photo set changed')
    for n, pin in pins.items():
        if Path(n).parts != ('images', Path(n).name) or hashlib.sha256((bank/n).read_bytes()).hexdigest() != pin:
            raise ValueError(f'fit image changed: {n}')
    items, weights = schedule(full, core.read(CUE), [json.loads(line) for line in RATINGS.read_text().splitlines()])
    digest = checksum(items)
    if digest != (HERE/'schedule.sha256').read_text().strip():
        raise ValueError('012 schedule changed')
    return items, weights, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('check', 'train', 'reload'))
    args = ap.parse_args()
    items, weights, digest = checked(BANK)
    manifest = dict(model=core.q.MODEL, revision=core.q.REV, seed=core.q.SEED, steps=len(items),
                    schedule_sha256=digest, fit_counts=dict(Counter(r[0] for r in items)), rare_weights=weights)
    if args.stage == 'check':
        print(json.dumps(manifest, sort_keys=True)); return
    out = ROOT/'work/overnight-012'
    if args.stage == 'train':
        if any((out/n).exists() for n in ('manifest.json', 'adapter', 'train.json')):
            raise FileExistsError('refusing to overwrite training artifacts')
        core.q.save(out/'manifest.json', manifest)
        model, proc = core.model()
        core.q.save(out/'train.json', core.train(model, proc, BANK, items, out))
    else:
        if core.read(out/'manifest.json') != manifest or not (out/'train.json').is_file():
            raise ValueError('missing training run or changed manifest')
        model, _ = core.model()
        core.reload_adapter(model, out)


if __name__ == '__main__':
    main()
