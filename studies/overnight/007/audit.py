"""Local-only provenance audit; no example test photos/labels or historical final opened."""
from collections import Counter
from pathlib import Path
import hashlib,json
import yaml
ROOT=Path(__file__).resolve().parents[3]; study=Path(__file__).parent
source=ROOT/'datasets/example'; old=ROOT/'studies/overnight/003'
def lines(p): return [json.loads(l) for l in p.read_text().splitlines()]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data): (study/name).write_text(json.dumps(data,indent=2)+'\n')
rows=lines(source/'train/ratings.jsonl'); sources={x['image']:x for x in lines(source/'sources.jsonl') if x['image'].startswith('train/')}; prior={Path(x['image']).name:x for x in lines(old/'ratings.jsonl')}; pins=json.loads((old/'image-sha256.json').read_text()); oldfit=set(json.loads((old/'split.json').read_text())['fit']); axes=yaml.safe_load((source/'axes.yaml').read_text());groups=sorted({r['group'] for r in rows});dev=groups[1::3]; photos={}; owners={}; counts={}; ds={}
for r in rows:
    n=r['image']; s=sources['train/'+n]; o=prior[Path(n).name]; sha=digest(source/'train'/n)
    assert sha==s['sha256']==pins[o['image']] and o['image'] in oldfit and o['group']==r['group'] and all(v==r['scores'][k] for k,v in o['scores'].items()) and '/train2017/' in s['url']; photos[n]=sha
    who=s.get('creator'); part='dev' if r['group'] in dev else 'fit'
    if who: assert owners.setdefault(who,part)==part
    c=counts.setdefault(part,{k:Counter(dict(L=0,M=0,H=0,null=0)) for k in axes});d=ds.setdefault(part,Counter())
    for k,v in r['scores'].items(): c[k]['null' if v is None else 'L' if v<-.5 else 'H' if v>.5 else 'M']+=1
    if r['scores']['liking'] is not None and r['scores']['scary'] is not None and r['scores']['liking']>.5 and r['scores']['scary']>.5:d['liked_and_scary']+=1
    if r['scores']['liking'] is not None and r['scores']['beauty'] is not None and r['scores']['liking']<-.5 and r['scores']['beauty']>.5:d['disliked_and_beautiful']+=1
assert len(rows)==len(photos)==len(sources)==24 and len(groups)==12
save('protocol.json',dict(dev_groups=dev,inputs={'axes.yaml':digest(source/'axes.yaml'),'ratings.jsonl':digest(source/'train/ratings.jsonl')},photos=photos));save('diversity.json',dict(counts=counts,mixed_examples=ds,known_creators=len(owners),unknown_creator_photos=sum(not s.get('creator') for s in sources.values())))
print(counts,ds,len(owners),dev)
