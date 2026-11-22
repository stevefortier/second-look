"""Reorder existing fit sentences to lead with a low endpoint, at 016 dose."""
import argparse

import q38_quality016 as half

parent = half.parent
# Second sentences already describe low color, motion or playfulness.
REVERSE = {'images/' + n + '.jpg' for n in ('000000231022', '000000068293',
           '000000480205', '000000221542', '000000022894')}


def schedule():
    rows, groups = half.schedule()
    full = parent.core.read(parent.fit.FULL)
    reordered = {}
    for n in REVERSE:
        first, second = full[n].split('. ')
        reordered[n] = second + ' ' + first + '.'
    return [(n, k, p, reordered[n] if n in reordered and a == full[n] else a)
            for n, k, p, a in rows], groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'train', 'before', 'after'))
    parent.schedule = schedule
    parent.HERE = parent.HERE.parent / '017'
    parent.OUT = parent.OUT.parent / 'overnight-017'
    parent.main(parser.parse_args().stage)
