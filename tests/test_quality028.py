from collections import Counter
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'studies/overnight/028'


def test_draw_reproduces_frozen_train_only_panel():
    import json
    assert (HERE / 'panel.py').exists(), 'missing reproducible draw'
    photos = runpy.run_path(str(HERE / 'panel.py'))['draw']()
    assert photos == json.loads((HERE / 'panel.json').read_text())
    assert len({p['group'] for p in photos}) == len(photos) == 8
    assert Counter(p['source'].split('/')[0] for p in photos) == {'liked':4, 'control':4}


def test_fixed_budget_half_fresh_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    assert (ROOT / 'scripts/q38_quality028.py').exists(), 'missing runner'
    import q38_quality027 as previous
    import q38_quality028 as study
    old, oldmeta = previous.checked()
    rows, meta = study.checked()
    assert meta['source_schedule_sha256'] == oldmeta['schedule_sha256']
    assert len(rows) == 80 and len(meta['group_counts']) == 16
    assert set(Counter(r[0] for r in rows).values()) == {5}
    assert all(a[1:3] == b[1:3] for a,b in zip(old,rows,strict=True))
    fresh = [b for a,b in zip(old,rows,strict=True) if a != b]
    assert len(fresh) == 40 and sum(not r[2] for r in fresh) == 20
    for name in {r[0] for r in rows}:
        assert len({r[2] for r in rows if r[0] == name}) == 3
    assert len(meta['fit_ratings']) == 16


def test_launch_requires_explicit_submission():
    import subprocess
    launcher = HERE / 'launch.sh'
    assert launcher.is_file(), 'missing one-shot launcher'
    result = subprocess.run(['bash', str(launcher)], capture_output=True, text=True)
    assert result.returncode == 64 and '--submit' in result.stderr


def test_train_dispatch_resolves_all_pinned_photos_without_model(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import q38_quality028 as study
    parent = study.parent
    for obj,key in ((parent,'checked'),(parent,'OUT'),(parent.dev,'checked'),(parent.fit,'BANK')):
        monkeypatch.setattr(obj,key,getattr(obj,key))
    called = []
    def inspect(stage):
        rows, _ = parent.checked()
        assert parent.OUT == ROOT/'work/overnight-028'
        assert all((parent.fit.BANK / name).is_file() for name,*_ in rows)
        assert len(parent.dev.checked()[0]) == 16
        called.append(stage)
    monkeypatch.setattr(parent, 'main', inspect)
    study.main('train')
    assert called == ['train']
