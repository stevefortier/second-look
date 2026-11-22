import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def test_paired_rewrites_change_only_prompted_targets():
    import q38_quality005 as old
    import q38_quality006 as new
    p = ROOT / 'studies/overnight/005'
    split, captions = [json.loads((p / n).read_text()) for n in ('split.json', 'targets.json')]
    axes = yaml.safe_load((p / 'axes.yaml').read_text())
    rows = {r['image']: r for r in map(json.loads, (p / 'ratings.jsonl').read_text().splitlines())}
    args = axes, split, captions, rows
    a, b = old.grounded_schedule(*args), new.linked_schedule(*args)
    assert len(b) == 72 and [x[:3] for x in a] == [x[:3] for x in b]
    assert len({x[3] for x in b}) == 72
    for x, y in zip(a, b):
        assert (x == y) == (not x[2])
        assert not x[2] or not any(anchor in y[3].lower() for cfg in axes.values() for anchor in cfg.values())
    rows[split['fit'][0]]['scores']['crowding'] = None
    with pytest.raises(ValueError):
        new.linked_schedule(*args)
