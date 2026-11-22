from collections import Counter
from importlib.util import find_spec
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_equal_group_rehearsal_emphasizes_existing_grounded_contrasts():
    assert find_spec('q38_quality014'), '014 schedule missing'
    import q38_quality014 as e
    items, groups = e.schedule()
    assert len(items) == 160 and len(groups) == 16
    assert set(Counter(groups[n] for n, *_ in items).values()) == {20}
    full, cue = e.core.read(e.fit.FULL), e.core.read(e.fit.CUE)
    for name in groups:
        answers = Counter(a for n, k, p, a in items if n == name and (k, p) == ('caption', ''))
        assert answers == {full[name]: (8 if name in e.CONTRAST else 5), cue[name]: (2 if name in e.CONTRAST else 5)}
    assert items == e.schedule()[0]
    assert len(e.CONTRAST) >= 1


def test_rejects_shared_dev_group():
    import q38_quality014 as e
    _, groups = e.schedule()
    with pytest.raises(ValueError, match='overlap'):
        e.check_split(groups, [{'group': next(iter(groups.values())), 'image': 'other.jpg'}])
