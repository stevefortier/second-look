"""010: retain every 008 update and add one focused neutral caption per fit photo."""
import hashlib
import json
from pathlib import Path
import random

import q38_quality as core
from q38_quality007 import validate as validate_bank

FOCUS = Path(__file__).resolve().parents[1]/'studies/overnight/010/focused.json'
BANK_SHA = 'e87c8eb796a3fa4dd0091d13600d5b21344784d27ff11d8acdb7005282b5725c'


def schedule(axes, split, captions, rows):
    items = core.schedule(axes, split, captions, rows)
    focused = core.read(FOCUS)
    if set(focused) != set(split['fit']) or len(set(focused.values())) != 16:
        raise ValueError('expected one distinct new caption per fit photo')
    for i, name in enumerate(split['fit']):
        answer = focused[name]
        if not answer.strip() or '<|' in answer or answer in captions.values():
            raise ValueError('invalid focused caption')
        items.append((name, 'caption', core.TRAIN_PROMPTS[(i+1)%2], answer))
    random.Random(core.q.SEED).shuffle(items)
    return items


def validate(bank, study):
    axes, split, caps, rows, digest = validate_bank(bank, study)
    if digest != BANK_SHA:
        raise ValueError('not the fixed 008 bank')
    ordered = schedule(axes, split, caps, rows)
    digest = hashlib.sha256(json.dumps(ordered, separators=(',', ':')).encode()).hexdigest()
    if digest != (FOCUS.parent/'schedule.sha256').read_text().strip():
        raise ValueError('ordered 010 schedule differs from predeclaration')
    return axes, split, caps, rows, digest


if __name__ == '__main__':
    core.validate = validate
    core.main(schedule_fn=schedule)
