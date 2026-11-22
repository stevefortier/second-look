"""Run AFTER blind judgments freeze. Emit only prompted (not spontaneous) metrics."""
from pathlib import Path
from collections import Counter
import json
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'work/overnight-013'

def compare(labels,before,after):
    gold={(r['image'],a):v for r in labels for a,v in r['scores'].items()}
    def keyed(rows):
        scores=[r for r in rows if r['kind']=='score']; pred={(r['image'],r['axis']):r['prediction'] for r in scores}
        if len(scores)!=len(pred) or pred.keys()!=gold.keys() or any(v not in (-1,0,1) for v in pred.values()): raise ValueError('unpaired prompt fields')
        return pred
    b,a=keyed(before),keyed(after); by={}
    for key,v in gold.items():
        for axis in ('all',key[1]):
            c=by.setdefault(axis,Counter()); bc=b[key]==v; ac=a[key]==v
            c.update(n=1,base=bc,adapter=ac,gains=ac and not bc,losses=bc and not ac)
    return by

if __name__=='__main__':
    def load(p): return json.loads(p.read_text())
    print(json.dumps(compare(load(HERE/'proxy.json'),load(OUT/'before.json'),load(OUT/'after.json')),indent=2))
