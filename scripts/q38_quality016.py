"""Half-dose image-only ablation of 014, balanced within each fit photo."""
import argparse
from collections import Counter

import q38_quality014 as parent

original = parent.schedule


def schedule():
    items, groups = original()
    seen = Counter()
    keep = []
    # Take alternate occurrences per (photo, answer), complementing odd
    # counts so both 8:2 and 5:5 mixes give five exposures per photo.
    full = parent.core.read(parent.fit.FULL)
    for row in items:
        photo, _, _, answer = row
        key = photo, answer
        parity = 0 if answer == full[photo] else 1
        if seen[key] % 2 == parity:
            keep.append(row)
        seen[key] += 1
    assert len(keep) == 80 and set(Counter(r[0] for r in keep).values()) == {5}
    return keep, groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'train', 'before', 'after'))
    parent.schedule = schedule
    parent.HERE = parent.HERE.parent / '016'
    parent.OUT = parent.OUT.parent / 'overnight-016'
    parent.main(parser.parse_args().stage)
