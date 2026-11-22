import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def test_only_secondary_caption_targets_change_in_place():
    assert find_spec('q38_quality011') is not None
    import q38_quality011 as exp
    bank = ROOT / 'work/quality008/bank/train'
    study = ROOT / 'studies/overnight/008'
    axes, split, caps, rows, digest = exp.validate(bank, study)
    old = exp.previous.schedule(axes, split, caps, rows)
    new = exp.schedule(axes, split, caps, rows)
    assert len(new) == len(old) == 157
    assert [r[:3] for r in old] == [r[:3] for r in new]
    changed = [b for a, b in zip(old, new) if a != b]
    assert len(changed) == 16 and {r[0] for r in changed} == set(split['fit'])
    assert all(k == 'caption' and a.count('. ') == 1 for _, k, _, a in changed)
    assert all(r[0] not in split['dev'] for r in new)
    assert digest == (ROOT/'studies/overnight/011/schedule.sha256').read_text().strip()


def test_reject_caption_set_with_nonfit_key(tmp_path, monkeypatch):
    import json
    import pytest
    import q38_quality011 as exp
    bank = ROOT / 'work/quality008/bank/train'
    *args, _ = exp.previous.validate(bank, ROOT / 'studies/overnight/008')
    targets = exp.core.read(exp.HERE / 'cue-impressions.json')
    targets[args[1]['dev'][0]] = targets.pop(args[1]['fit'][0])
    (tmp_path / 'cue-impressions.json').write_text(json.dumps(targets))
    monkeypatch.setattr(exp, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='16 fit photos'):
        exp.schedule(*args)
