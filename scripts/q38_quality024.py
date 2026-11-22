"""Halve balanced explicit supervision; restore the other slots verbatim."""
import argparse
from collections import Counter
import q38_quality023 as full

prior = full.prior
parent = full.parent
HERE = full.HERE.parent / '024'
KEEP = {
    'images/000000068293.jpg': 'crowding', 'images/000000117789.jpg': 'cuteness',
    'images/000000137788.jpg': 'liking', 'images/000000171484.jpg': 'funny',
    'images/000000177529.jpg': 'scary', 'images/000000221542.jpg': 'color_intensity',
    'images/000000494973.jpg': 'activity', 'images/000000516947.jpg': 'beauty',
}


def recipe():
    balanced, meta = full.checked()
    original, _ = prior.checked()
    seen = Counter()
    result = []
    for both, caption in zip(balanced, original, strict=True):
        name = caption[0]
        slot = seen[name]; seen[name] += 1
        keep = slot in (4, 5) and full.AXIS_PAIRS[name][slot - 4] == KEEP[name]
        result.append(both if keep else caption)
    return result, dict(meta, source_schedule_sha256=meta['schedule_sha256'],
                        schedule_sha256=prior.fit.checksum(result))


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
    parent.OUT = parent.OUT.parent/'overnight-024'
    parent.main(stage)
