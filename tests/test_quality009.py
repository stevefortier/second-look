from collections import Counter
from importlib.util import find_spec
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def test_caption_heavy_matched_budget():
    assert find_spec('q38_quality009') is not None
    import q38_quality009 as exp
    axes = {f'a{i}': dict(low='low', mid='mid', high='high') for i in range(8)}
    names = [f'fit{i}' for i in range(16)]
    split = dict(fit=names, dev=['never-train'])
    captions = {name: name+' caption' for name in names}
    rows = {name: {'scores': {a: None if a=='a7' and i<3 else 1 for a in axes}} for i,name in enumerate(names)}
    args = axes, split, captions, rows
    base = exp.core.schedule(*args)
    items = exp.schedule(*args)
    assert items == exp.schedule(*args) and set(items) <= set(base)
    assert Counter((n,p,a) for n,k,p,a in items if k=='caption') == { (n,p,a):6 for n,k,p,a in base if k=='caption' }
    scores = [x for x in items if x[1]=='score']
    assert len(items)==len(base)==141 and len(scores)==len(set(scores))==45
    assert sorted(Counter(x[2] for x in scores).values()) == [5]*3+[6]*5
