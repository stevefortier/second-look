"""Test whether a small dose of explicit KVP supervision transfers to neutral replies."""
from collections import Counter
import argparse
import json
import yaml
import q38_quality020 as prior

parent = prior.parent
HERE = prior.HERE.parent/'021'


def recipe():
    items, meta = prior.checked()
    axes = yaml.safe_load((parent.dev.STUDY/'axes.yaml').read_text())
    labels = {r['image']:r['scores'] for path in (prior.fit.RATINGS, prior.prev.fresh.HERE/'ratings.jsonl')
              for r in map(json.loads, path.read_text().splitlines())}
    seen = Counter()
    result = []
    for name,kind,prompt,text in items:
        slot = seen[name]; seen[name] += 1
        if slot in (4, 5):
            scores = labels[name]
            ordered = sorted((v,a) for a,v in scores.items() if v is not None)
            v,a = ordered[0 if slot == 4 else -1]
            kind, prompt, text = 'score', parent.core.explicit(a, axes[a]), ('L' if v < -.5 else 'H' if v > .5 else 'M')
        result.append((name,kind,prompt,text))
    return result, dict(meta, source_schedule_sha256=meta['schedule_sha256'],schedule_sha256=prior.fit.checksum(result))


def checked():
    items, meta = recipe()
    if meta['schedule_sha256'] != (HERE/'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return items,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check','train','before','after'))
    stage = parser.parse_args().stage
    if stage == 'train':
        items, meta = checked()
        earlier = set(meta['slot_substitution'].values())
        prior.fit.BANK = prior.MixedBank({n:(prior.OLD_BANK if n in earlier else prior.prev.fresh.BANK)/n for n,*_ in items})
    parent.checked = checked
    parent.dev.checked = prior.dev_panel
    parent.OUT = parent.OUT.parent/'overnight-021'
    parent.main(stage)
