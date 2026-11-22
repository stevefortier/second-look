from collections import Counter
from importlib.util import find_spec
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def test_focus_additions_preserve_all_original_fit_rows():
    assert find_spec('q38_quality010') is not None
    import q38_quality010 as exp
    study = ROOT/'studies/overnight/008'
    caps = json.loads((study/'targets.json').read_text())
    rows = {r['image']:r for r in map(json.loads,(study/'fit-ratings.jsonl').read_text().splitlines())}
    import yaml
    axes = yaml.safe_load((ROOT/'work/quality008/bank/axes.yaml').read_text())
    args = axes, {'fit':list(rows),'dev':['held-out']}, caps, rows
    base = Counter(exp.core.schedule(*args)); schedule = exp.schedule(*args)
    assert schedule == exp.schedule(*args) and len(schedule) == 157
    assert not base - Counter(schedule)
    extra = list((Counter(schedule)-base).elements())
    assert len(extra) == 16 and {n for n,k,p,a in extra} == set(rows)
    assert all(k == 'caption' and p in exp.core.TRAIN_PROMPTS and a not in caps.values() for n,k,p,a in extra)
    assert all(p != next(op for on,ok,op,oa in base if on==n and ok=='caption') for n,k,p,a in extra)
