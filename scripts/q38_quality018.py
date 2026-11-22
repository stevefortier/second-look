"""Image-only caption rehearsal on eight newly reviewed fit groups."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import yaml

import q38_quality014 as parent

HERE = parent.HERE.parent / '018'
BANK = parent.fit.ROOT / 'work/quality018'


def checked():
    photos = parent.core.read(HERE / 'panel.json')
    captions = parent.core.read(HERE / 'impressions.json')
    rows = [json.loads(line) for line in (HERE / 'ratings.jsonl').read_text().splitlines()]
    groups = {r['image']: r['group'] for r in rows}
    axes = yaml.safe_load((parent.dev.STUDY / 'axes.yaml').read_text())
    coverage = {a: Counter(dict(low=0, mid=0, high=0, unknown=0)) for a in axes}
    if not (len(photos) == len(rows) == len(groups) == len(set(groups.values())) == 8
            and {p['image']: p['group'] for p in photos} == groups and groups.keys() == captions.keys()):
        raise ValueError('fit roster changed')
    parent.check_split(groups, parent.core.read(parent.dev.STUDY / 'panel.json')['photos'])
    for r in rows:
        if r['scores'].keys() != axes.keys():
            raise ValueError('axis mismatch')
        for a, v in r['scores'].items():
            if v is not None and (type(v) != int or v not in (-1, 0, 1)):
                raise ValueError('invalid proxy rating')
            coverage[a][{None:'unknown', -1:'low', 0:'mid', 1:'high'}[v]] += 1
    for p in photos:
        n = p['image']
        if Path(n).parts != ('images', Path(n).name) or hashlib.sha256((BANK/n).read_bytes()).hexdigest() != p['sha256']:
            raise ValueError('fit image mismatch')
        if not captions[n].strip() or '<|' in captions[n]:
            raise ValueError('invalid impression')
    items = [(n, 'caption', '', captions[n]) for n in sorted(groups) for _ in range(10)]
    random.Random(parent.core.q.SEED).shuffle(items)
    digest = parent.fit.checksum(items)
    if digest != (HERE / 'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return items, dict(schedule_sha256=digest, steps=len(items), coverage=coverage,
                       group_counts=dict(Counter(groups[n] for n,*_ in items)), model=parent.core.q.MODEL,
                       revision=parent.core.q.REV, seed=parent.core.q.SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'train', 'before', 'after'))
    parent.checked = checked
    parent.fit.BANK = BANK
    parent.OUT = parent.OUT.parent / 'overnight-018'
    parent.main(parser.parse_args().stage)
