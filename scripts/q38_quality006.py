"""006: replace 005's prompted answers, retaining its exact update order."""
import csv
from pathlib import Path

from q38_quality005 import core, grounded_schedule

PROSE = Path(__file__).resolve().parents[1] / 'studies/overnight/006/prose.tsv'


def linked_schedule(axes, split, captions, rows):
    with PROSE.open(encoding='utf8', newline='') as f:
        source = list(csv.DictReader(f, delimiter='\t'))
    rewrites = {r['image']: r for r in source}
    if len(source) != len(rewrites) or rewrites.keys() != set(split['fit']):
        raise ValueError('rewrite keys must be unique and precisely fit-only')
    for name, r in rewrites.items():
        if set(r) != {'image', *axes, 'description', 'impression'}:
            raise ValueError('unexpected study-specific prose columns')
        if any((None if r[k] == 'NA' else float(r[k])) != rows[name]['scores'][k] for k in axes):
            raise ValueError(f'reviewed scores changed: {name}')
    variants = dict(zip(core.TRAIN_PROMPTS, ('description', 'impression')))
    return [(name, kind, prompt, rewrites[name][variants[prompt]] if prompt else answer)
            for name, kind, prompt, answer in grounded_schedule(axes, split, captions, rows)]


if __name__ == '__main__':
    core.main(schedule_fn=linked_schedule)
