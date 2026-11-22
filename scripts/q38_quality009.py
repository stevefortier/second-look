"""009: reweight the unchanged 008 supervision, matching optimizer steps."""
import random

import q38_quality as core
from q38_quality007 import validate


def schedule(axes, split, captions, rows):
    base = core.schedule(axes, split, captions, rows)
    items = [row for row in base if row[1] == 'caption'] * 6
    # Seed-shuffled original order gives a deterministic sample within each axis.
    for i, (axis, cfg) in enumerate(axes.items()):
        prompt = core.explicit(axis, cfg)
        pool = [row for row in base if row[1] == 'score' and row[2] == prompt]
        items.extend(pool[:6 if i < 5 else 5])
    if len(items) != len(base) or len(items) != 141:
        raise ValueError('unexpected 008 fit budget')
    random.Random(core.q.SEED).shuffle(items)
    return items


if __name__ == '__main__':
    core.validate = validate
    core.main(schedule_fn=schedule)
