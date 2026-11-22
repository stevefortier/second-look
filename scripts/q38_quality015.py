"""014 matched control: change only the invitation on half the caption updates."""
import argparse
from collections import Counter

import q38_quality014 as parent

original = parent.schedule
PROMPTS = ('Give a brief first impression.', 'Tell me about the photograph.')


def schedule():
    items, groups = original()
    count = Counter()
    mixed = []
    for name, kind, _, answer in items:
        i = count[name]
        count[name] += 1
        prompt = PROMPTS[(i // 2) % 2] if i % 2 else ''
        mixed.append((name, kind, prompt, answer))
    assert set(count.values()) == {10} and not set(PROMPTS) & set(parent.core.DEV_PROMPTS)
    return mixed, groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check','train','before','after'))
    args = parser.parse_args()
    parent.schedule = schedule
    parent.HERE = parent.HERE.parent/'015'
    parent.OUT = parent.OUT.parent/'overnight-015'
    parent.main(args.stage)
