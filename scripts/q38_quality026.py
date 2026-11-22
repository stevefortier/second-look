"""Pair 025 impressions only when the training invitation is image-only."""
import argparse
from pathlib import Path

import q38_quality025 as previous

HERE = previous.HERE.parent / '026'


def recipe():
    source, meta = previous.checked()
    result = []
    for name, kind, prompt, target in source:
        if not prompt:
            pair = previous.TARGETS[Path(name).stem]
            target = target + ' ' + pair[1 - pair.index(target)]
        result.append((name, kind, prompt, target))
    return result, dict(meta, source_schedule_sha256=meta['schedule_sha256'],
                        schedule_sha256=previous.prior.fit.checksum(result))


def checked():
    rows, meta = recipe()
    if meta['schedule_sha256'] != (HERE / 'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return rows, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'train', 'before', 'after'))
    stage = parser.parse_args().stage
    prior, parent = previous.prior, previous.parent
    if stage == 'train':
        items, meta = checked()
        earlier = set(meta['slot_substitution'].values())
        prior.fit.BANK = prior.MixedBank({n:(prior.OLD_BANK if n in earlier else prior.prev.fresh.BANK)/n for n,*_ in items})
    parent.checked = checked
    parent.dev.checked = prior.dev_panel
    parent.OUT = parent.OUT.parent / 'overnight-026'
    parent.main(stage)
