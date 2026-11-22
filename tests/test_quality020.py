from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_panel_substitution_preserves_slot_order_and_invitations(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    assert (ROOT/'scripts/q38_quality020.py').is_file(), '020 recipe missing'
    import q38_quality020 as r
    old, _ = r.prev.checked()
    new, manifest = r.checked()
    assert len(new) == len(old) == 80
    assert sum(a == b for a, b in zip(new, old)) == 40
    assert [a[1:3] for a in new] == [a[1:3] for a in old]
    pairs = {(a[0], b[0]) for a, b in zip(old, new)}
    assert len(pairs) == len({b for _, b in pairs}) == 8
    changes = [b for a, b in zip(old,new) if a != b]
    assert sorted(Counter(b[2] for b in changes).values()) == [10,10,20]
    assert sorted(manifest['group_counts'].values()) == [10]*8
    assert len(r.dev_panel()[0]) == 16
