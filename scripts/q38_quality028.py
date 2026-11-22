"""Widen fit coverage at fixed update and invitation budgets; no 027 outputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml
import q38_quality027 as previous

prior = previous.fit_targets.prior
parent = prior.parent
HERE = previous.HERE.parent / '028'
BANK = prior.fit.ROOT / 'work/quality028'


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def prepare():
    old, meta = previous.checked()
    old_names = sorted({r[0] for r in old})
    fresh = parent.core.read(HERE / 'panel.json')
    fresh_names = [p['image'] for p in fresh]
    pairs = parent.core.read(HERE / 'impressions.json')
    new_ratings = records(HERE / 'ratings.jsonl')
    old_ratings = [r for f in (prior.fit.RATINGS, prior.prev.fresh.HERE/'ratings.jsonl')
                   for r in records(f) if r['image'] in old_names]
    ratings = old_ratings + new_ratings
    groups = {r['image']:r['group'] for r in ratings}
    if (len(old_names) != 8 or len(fresh) != 8 or len(ratings) != 16 or len(groups) != 16
        or len(set(groups.values())) != 16 or set(pairs) != set(fresh_names)
        or {r['image']:r['group'] for r in new_ratings} != {p['image']:p['group'] for p in fresh}):
        raise ValueError('invalid fit roster')
    dev = parent.core.read(parent.dev.STUDY / 'panel.json')['photos']
    parent.check_split(groups, dev)
    axes = yaml.safe_load((parent.dev.STUDY / 'axes.yaml').read_text())
    for r in new_ratings:
        scores = r['scores']
        if scores.keys() != axes.keys() or scores['liking'] is not None or any(
                v is not None and (type(v) != int or v not in (-1,0,1)) for v in scores.values()):
            raise ValueError('invalid fresh proxy rating')
    for p in fresh:
        n = p['image']
        prior.verify(BANK, n, p['sha256'])
        if not (isinstance(pairs[n], list) and len(set(pairs[n])) == len(pairs[n]) == 2
                and all(isinstance(s,str) and s.strip() and '<|' not in s for s in pairs[n])):
            raise ValueError('invalid impressions')
    earlier = set(meta['slot_substitution'].values())
    paths = {n:(prior.OLD_BANK if n in earlier else prior.prev.fresh.BANK)/n for n in old_names}
    paths.update({n:BANK/n for n in fresh_names})
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    if len(hashes) != 16 or hashes & {p['sha256'] for p in dev}:
        raise ValueError('fit bytes overlap')
    assignment = dict(zip(old_names, enumerate(fresh_names), strict=True))
    seen = Counter()
    rows = []
    for n,kind,prompt,target in old:
        i, new = assignment[n]
        index = seen[n]
        seen[n] += 1
        if (index < 5) == (i % 2 == 0):
            a,b = previous.fit_targets.TARGETS[Path(n).stem]
            first,second = pairs[new]
            if target == f'{b} {a}': first,second = second,first
            elif target != f'{a} {b}': raise ValueError('unrecognized source pair')
            n,target = new,f'{first} {second}'
        rows.append((n,kind,prompt,target))
    info = dict(model=meta['model'], revision=meta['revision'], seed=meta['seed'], steps=len(rows),
                source_schedule_sha256=meta['schedule_sha256'], schedule_sha256=prior.fit.checksum(rows),
                group_counts=dict(Counter(groups[r[0]] for r in rows)), fit_ratings=ratings)
    return rows, info, paths


def checked():
    rows, meta, _ = prepare()
    if meta['schedule_sha256'] != (HERE/'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return rows, meta


def main(stage):
    if stage == 'train':
        _, _, paths = prepare()
        prior.fit.BANK = prior.MixedBank(paths)
    parent.checked = checked
    parent.dev.checked = prior.dev_panel
    parent.OUT = HERE.parents[2] / 'work/overnight-028'
    parent.main(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check','train','before','after'))
    main(parser.parse_args().stage)
