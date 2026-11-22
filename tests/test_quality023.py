from collections import Counter
from pathlib import Path
import json
import yaml


def test_balanced_observed_labels_and_fixed_slots(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    assert (root/'scripts/q38_quality023.py').is_file()
    import q38_quality023 as exp
    prior = exp.prior
    items, _ = exp.checked()
    before, _ = prior.checked()
    axes = yaml.safe_load((exp.parent.dev.STUDY/'axes.yaml').read_text())
    prompts = {exp.parent.core.explicit(a,v):a for a,v in axes.items()}
    labels = {r['image']:r['scores'] for path in (prior.fit.RATINGS, prior.prev.fresh.HERE/'ratings.jsonl') for r in map(json.loads,path.read_text().splitlines())}
    seen, counts, classes = Counter(), Counter(), Counter()
    per_photo = {}
    for old,new in zip(before,items,strict=True):
        n,k,p,t = new
        assert n == old[0]
        if seen[n] in (4,5):
            a = prompts[p]; v = labels[n][a]
            assert k == 'score' and v is not None
            assert t == ('L' if v < -.5 else 'H' if v > .5 else 'M')
            counts[a] += 1; classes[t] += 1
            per_photo.setdefault(n,[]).append((a,t))
        else:
            assert new == old
        seen[n] += 1
    assert len(seen) == 8 and set(seen.values()) == {10}
    assert counts == {a:2 for a in axes} and classes == {'L':6,'M':5,'H':5}
    assert all(a != b and x != y for (a,x),(b,y) in per_photo.values())
