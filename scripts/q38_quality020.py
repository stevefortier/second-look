"""Balanced fit-panel substitution at fixed 019 slot order and invitation dose."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml
import q38_quality019 as prev

parent = prev.parent
fit = parent.fit
HERE = prev.HERE.parent / '020'
OLD_BANK = fit.BANK


def dev_panel():
    # Inference needs the ordered axes and images, never the proxy scores.
    photos = parent.core.read(parent.dev.STUDY/'panel.json')['photos']
    axes = yaml.safe_load((parent.dev.STUDY/'axes.yaml').read_text())
    for p in photos:
        verify(parent.dev.BANK, p['image'], p['sha256'])
    if len({p['group'] for p in photos}) != 16 or len({p['image'] for p in photos}) != len(photos) or len(photos) != 16 or len(axes) != 8:
        raise ValueError('development roster changed')
    return [p['image'] for p in photos], axes


def verify(bank, name, digest):
    if Path(name).parts != ('images', Path(name).name) or hashlib.sha256((bank/name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'image changed: {name}')


def recipe():
    base, source = prev.checked()
    rows = [json.loads(s) for s in fit.RATINGS.read_text().splitlines()]
    full = parent.core.read(fit.FULL)
    pins = parent.core.read(fit.HERE/'fit-photos.json')
    fresh = parent.core.read(prev.fresh.HERE/'panel.json')
    earlier = sorted({r['group'] for r in rows})[::2]
    selected = [min(r['image'] for r in rows if r['group'] == g) for g in earlier]
    slots = sorted(p['image'] for p in fresh)
    repl = dict(zip([slots[i] for i in (0, 3, 4, 7)], selected, strict=True))
    groups = {p['image']: p['group'] for p in fresh if p['image'] not in repl}
    groups.update(zip(selected, earlier, strict=True))
    photos = parent.core.read(parent.dev.STUDY/'panel.json')['photos']
    parent.check_split(groups, photos)
    if len(groups) != 8 or len(set(groups.values())) != 8:
        raise ValueError('duplicate group or image')
    hashes = [p['sha256'] for p in fresh if p['image'] not in repl] + [pins[n] for n in selected]
    if len(set(hashes)) != 8 or set(hashes) & {p['sha256'] for p in photos}:
        raise ValueError('duplicate image bytes')
    for n in selected:
        verify(OLD_BANK, n, pins[n])
        if not full[n].strip() or '<|' in full[n]:
            raise ValueError('invalid reviewed target')
    items = [(repl[n], k, p, full[repl[n]]) if n in repl else (n,k,p,t) for n,k,p,t in base]
    return items, dict(model=source['model'], revision=source['revision'], seed=source['seed'],
                       steps=len(items), slot_substitution=repl, source_schedule_sha256=source['schedule_sha256'],
                       schedule_sha256=fit.checksum(items), group_counts=dict(Counter(groups[n] for n,*_ in items)))


def checked():
    items, manifest = recipe()
    if manifest['schedule_sha256'] != (HERE/'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return items, manifest


class MixedBank:
    def __init__(self, paths):
        self.paths = paths

    def __truediv__(self, name):
        return self.paths[name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'train', 'before', 'after'))
    stage = parser.parse_args().stage
    if stage == 'train':
        items, manifest = checked()
        earlier = set(manifest['slot_substitution'].values())
        fit.BANK = MixedBank({n: (OLD_BANK if n in earlier else prev.fresh.BANK)/n for n,*_ in items})
    parent.checked = checked
    parent.dev.checked = dev_panel
    parent.OUT = parent.OUT.parent/'overnight-020'
    parent.main(stage)
