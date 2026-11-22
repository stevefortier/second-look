"""Equal fit-group rehearsal; evaluate only against the frozen 013 dev panel."""
import argparse
from collections import Counter
import hashlib
import json
import random

import q38_eval013 as dev
import q38_quality012 as fit
from q38_quality012 import core

OUT = fit.ROOT / 'work/overnight-014'
HERE = fit.HERE.parent / '014'
# Full captions with explicit visual cues for opposing fit-axis endpoints.
CONTRAST = {'images/' + name + '.jpg' for name in (
    '000000020371', '000000231022', '000000068293', '000000480205',
    '000000221542', '000000493438', '000000022894', '000000557329')}


def schedule():
    full, cue = core.read(fit.FULL), core.read(fit.CUE)
    rows = [json.loads(line) for line in fit.RATINGS.read_text().splitlines()]
    groups = {r['image']: r['group'] for r in rows}
    if not (len(rows) == len(groups) == 16 and set(full) == set(cue) == set(groups)
            and set(Counter(groups.values()).values()) == {2} and CONTRAST <= set(full)):
        raise ValueError('fit roster changed')
    items = [(name, 'caption', '', (full if i < (8 if name in CONTRAST else 5) else cue)[name])
             for name in sorted(groups) for i in range(10)]
    random.Random(core.q.SEED).shuffle(items)
    return items, groups


def check_split(groups, photos):
    if set(groups.values()) & {p['group'] for p in photos} or groups.keys() & {p['image'] for p in photos}:
        raise ValueError('fit/dev overlap')


def checked():
    items, groups = schedule()
    photos = core.read(dev.STUDY / 'panel.json')['photos']
    check_split(groups, photos)
    pins = core.read(fit.HERE / 'fit-photos.json')
    if pins.keys() != groups.keys() or any(hashlib.sha256((fit.BANK/n).read_bytes()).hexdigest() != h for n,h in pins.items()):
        raise ValueError('fit photos changed')
    digest = fit.checksum(items)
    if digest != (HERE / 'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return items, dict(schedule_sha256=digest, steps=len(items), group_counts=dict(Counter(groups[n] for n,*_ in items)),
                       model=core.q.MODEL, revision=core.q.REV, seed=core.q.SEED)


def main(stage):
    if stage in ('check', 'train'):
        items, manifest = checked()
        if stage == 'check':
            dev.checked(); print(json.dumps(manifest, sort_keys=True)); return
        if any((OUT/n).exists() for n in ('manifest.json','train.json','adapter')):
            raise FileExistsError('existing training artifacts')
        core.q.save(OUT/'manifest.json', manifest)
        m, proc = core.model()
        core.q.save(OUT/'train.json', core.train(m, proc, fit.BANK, items, OUT)); return
    names, axes = dev.checked()
    dest = OUT / (stage + '.json')
    if any(p.exists() for p in (dest, OUT/'blind.json', OUT/'blind-key.private.json')):
        raise FileExistsError(dest)
    m, proc = core.model()
    if stage == 'after':
        m = core.reload_adapter(m, OUT)
    rows = core.evaluate(m, proc, dev.BANK, names, axes)
    expected = {(n, 'neutral', p) for n in names for p in (*core.DEV_PROMPTS, '[image only]')}
    expected.update((n, 'score', a) for n in names for a in axes)
    if len(rows) != len(expected) or {(r['image'], r['kind'], r.get('axis', r['prompt'])) for r in rows} != expected:
        raise ValueError('incomplete evaluation')
    core.q.save(dest, rows)
    if stage == 'after':
        core.blind(OUT, core.read(OUT/'before.json'), rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check','train','before','after'))
    main(parser.parse_args().stage)
