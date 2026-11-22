"""Change only the invitation on half of the frozen 018 fit updates."""
import argparse
from collections import Counter

import q38_quality018 as fresh
from q38_quality018 import parent

HERE = fresh.HERE.parent / '019'


def schedule(items):
    prompts = parent.core.TRAIN_PROMPTS
    if len(set(prompts)) != 2 or set(prompts) & {'', *parent.core.DEV_PROMPTS}:
        raise ValueError('invalid held-in invitations')
    offsets = {n: i % 2 for i, n in enumerate(sorted({r[0] for r in items}))}
    counts = Counter()
    mixed = []
    for name, kind, prompt, text in items:
        i = counts[name]
        counts[name] += 1
        if i % 2:
            prompt = prompts[(i // 2 + offsets[name]) % 2]
        mixed.append((name, kind, prompt, text))
    return mixed


def checked():
    base, manifest = fresh.checked()
    items = schedule(base)
    digest = parent.fit.checksum(items)
    if digest != (HERE / 'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return items, dict(manifest, source_schedule_sha256=manifest['schedule_sha256'],
                       schedule_sha256=digest, prompt_counts=dict(Counter(r[2] for r in items)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'train', 'before', 'after'))
    parent.checked = checked
    parent.fit.BANK = fresh.BANK
    parent.OUT = parent.OUT.parent / 'overnight-019'
    parent.main(parser.parse_args().stage)
