"""011: replace only the 16 secondary targets in the pinned 010 schedule."""
import hashlib
import json
from pathlib import Path

import q38_quality010 as previous

core = previous.core
HERE = Path(__file__).resolve().parents[1] / 'studies/overnight/011'


def schedule(axes, split, captions, rows):
    old = previous.schedule(axes, split, captions, rows)
    source = core.read(previous.FOCUS)
    revised = core.read(HERE / 'cue-impressions.json')
    if set(revised) != set(split['fit']) or len(set(revised.values())) != 16:
        raise ValueError('rewrites must cover exactly the 16 fit photos')
    for text in revised.values():
        if '<|' in text or text.count('. ') != 1 or not text.endswith('.'):
            raise ValueError('expected a cue sentence followed by an impression sentence')
    return [(n, k, p, revised[n] if k == 'caption' and a == source[n] else a)
            for n, k, p, a in old]


def validate(bank, study):
    *args, _ = previous.validate(bank, study)
    ordered = schedule(*args)
    digest = hashlib.sha256(json.dumps(ordered, separators=(',', ':')).encode()).hexdigest()
    if digest != (HERE/'schedule.sha256').read_text().strip():
        raise ValueError('ordered 011 schedule differs from predeclaration')
    return (*args, digest)


if __name__ == '__main__':
    core.validate = validate
    core.main(schedule_fn=schedule)
