from pathlib import Path


def test_midpoint_replaces_one_score_per_photo(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    assert (root/'scripts/q38_quality022.py').exists()
    import q38_quality021 as baseline
    import q38_quality022 as experiment
    old, _ = baseline.checked()
    new, _ = experiment.checked()
    changes = [(a,b) for a,b in zip(old,new,strict=True) if a != b]
    assert len(old) == 80 and len(changes) == len({b[0] for _,b in changes}) == 8
    assert all(a[:2] == b[:2] and b[1] == 'score' and b[3] == 'M' for a,b in changes)
