from collections import Counter
from importlib.util import find_spec
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def test_locked_caption_budget_and_fit_only_reads(monkeypatch):
    assert find_spec('q38_quality012') is not None, '012 schedule missing'
    import q38_quality012 as exp
    names = set(exp.core.read(exp.FULL))
    permitted = {(exp.BANK/n).resolve() for n in names}
    original = Path.open
    def guard(path, *args, **kwargs):
        if exp.BANK.parent in path.resolve().parents:
            assert path.resolve() in permitted, f'non-fit data read: {path}'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    items, weights, digest = exp.checked(exp.BANK)
    assert len(items) == 157 and len(names) == 16
    counts = Counter(n for n, _, _, _ in items)
    assert set(counts) == names
    # Frozen fit-only audit: rare endpoints belong to these eight images.
    assert {Path(n).stem: counts[n]-8 for n in names if counts[n] > 8} == {
        '000000020371': 3, '000000068293': 3, '000000196355': 3,
        '000000221542': 8, '000000231022': 3, '000000478717': 2,
        '000000516328': 2, '000000557329': 5,
    }
    assert all(8 <= counts[n] <= 16 and (counts[n] > 8) == (weights[n] > 0) for n in names)
    full, cue = exp.core.read(exp.FULL), exp.core.read(exp.CUE)
    for n, k, p, a in items:
        assert (k, p) == ('caption', '') and a in (full[n], cue[n])
    assert digest == (exp.HERE/'schedule.sha256').read_text().strip()


def test_changed_fit_image_fails_before_training(tmp_path):
    import q38_quality012 as exp
    for n in exp.core.read(exp.FULL):
        path = tmp_path/n
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'wrong bytes')
    with pytest.raises(ValueError, match='fit image'):
        exp.checked(tmp_path)
