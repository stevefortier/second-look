from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_only_invitation_changes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    assert (ROOT / 'scripts/q38_quality019.py').exists(), '019 mixed-invitation recipe missing'
    import q38_quality019 as r
    old, baseline = r.fresh.checked()
    new, manifest = r.checked()
    assert [x[:2] + x[3:] for x in new] == [x[:2] + x[3:] for x in old]
    a, b = r.parent.core.TRAIN_PROMPTS
    assert not {a, b} & set(r.parent.core.DEV_PROMPTS)
    assert Counter(x[2] for x in new) == {'': 40, a: 20, b: 20}
    for n in {x[0] for x in old}:
        prompts = Counter(x[2] for x in new if x[0] == n)
        assert prompts[''] == 5 and sorted((prompts[a], prompts[b])) == [2, 3]
    assert manifest['group_counts'] == baseline['group_counts']
    assert manifest['coverage'] == baseline['coverage']
    assert manifest['schedule_sha256'] != baseline['schedule_sha256']
