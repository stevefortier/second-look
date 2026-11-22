"""Regenerate primary and secondary reports from published inputs; run at repo root."""
import json
import sys
import yaml
from collections import Counter, defaultdict
from pathlib import Path

study = Path(__file__).resolve().parent
sys.path.insert(0, str(study.parents[2] / 'scripts'))
import quality_metrics as qm

out = study / 'results'
load = lambda name: json.loads((out / name).read_text())
truth = qm.ratings(study / 'ratings.jsonl')
split = json.loads((study / 'split.json').read_text())
anchors = [v.casefold() for cfg in yaml.safe_load((study/'axes.yaml').read_text()).values() for v in cfg.values()]
fit, dev = ({k: truth[k] for k in split[s]} for s in ('fit', 'dev'))
base, adapter = (qm.predictions(out / f'{s}.json') for s in ('before','after'))
assert base.keys() == adapter.keys()
secondary = qm.summary(base, adapter, dev, fit)
join = load('join.json'); counts = defaultdict(Counter)
judgments = load('blind-judgments.json')
assert len(judgments) == len(join) == len(set(j['id'] for j in judgments)) == 72
for j in judgments:
    side, image, prompt = join[j['id']]
    assert j['text'] == (base if side=='before' else adapter)[image,'neutral',prompt,None]['text']
    scores = dev[image]
    for prefix in (side, side + ' | ' + prompt):
        c = counts[prefix]; c['responses'] += 1
        text = j['text'].casefold()
        c['contains_literal_anchor'] += any(anchor in text for anchor in anchors)
        c['contains_my_impression_prefix'] += 'my impression:' in text
        for k in ('schema_echo','generic_unsupported_praise'): c[k] += int(j[k])
        c[j['relevance']] += 1
        mixed = set(scores.values()) >= {-1,1}; both = mixed
        for axis,y in scores.items():
            code = j[axis]; a = counts[prefix+' | '+axis]; mention = code not in ('ambiguous','absent'); direction = {'low':-1,'mid':0,'high':1}.get(code)
            if y is None:
                a['unknown'] += 1; a['mentions_unknown'] += mention; continue
            correct = direction == y; a['eligible'] += 1; a['exact'] += correct; a['mentions'] += mention
            a[f'truth_{y}:eligible'] += 1; a[f'truth_{y}:exact'] += correct
            if not mention: a[code] += 1
            elif not correct: a['opposite' if direction*y < 0 else 'midpoint_disagreement'] += 1
            if abs(y)==1: both = both and correct
        c['mixed_eligible'] += mixed; c['mixed_both_exact'] += both
for name,obj in [('primary',counts),('secondary',secondary)]:
    (out / (name+'.json')).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
