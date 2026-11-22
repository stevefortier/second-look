"""Reproduce 006 using only published 005/006 text files. No photos/model needed."""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import yaml

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'005'
RAW = HERE/'results'
sys.path.insert(0, str(HERE.parents[2]/'scripts'))
import quality_metrics as qm
from freeze import AXES, FLAGS


def load(path):
    return json.loads(path.read_bytes())


def tokens(s):
    return ' ' + ' '.join(re.findall(r'[^\W_]+', s.casefold())) + ' '


def measure(j, truth, anchors):
    text = tokens(j['text'])
    counts = Counter(responses=1, literal_echo=int(any(a in text for a in anchors)),
                     unsupported_detail=int(j['relevance']=='on subject; questionable detail'),
                     on_subject=int(j['relevance'].startswith('on subject')))
    counts.update({k:int(j[k]) for k in FLAGS})
    hit = set()
    for ax, y in truth.items():
        code = j[ax]; d = {'low':-1, 'mid':0, 'high':1}.get(code)
        metrics = dict.fromkeys(('unknown','mentions_unknown','eligible','mentions','absent','ambiguous','exact','opposite','midpoint_error','misses'),0)
        if y is None:
            metrics.update(unknown=1, mentions_unknown=int(d is not None))
        else:
            metrics.update(eligible=1, mentions=int(d is not None), exact=int(d==y), misses=int(d!=y))
            metrics[f'truth_{y}_eligible'] = 1
            metrics[f'truth_{y}_exact'] = int(d==y)
            if d==y: hit.add(y)
            elif d is None: metrics[code]=1
            elif d*y<0: metrics['opposite']=1
            else: metrics['midpoint_error']=1
        counts.update({ax+'.'+k:v for k,v in metrics.items()})
    counts.update(mixed_eligible=int({-1,1}<=set(truth.values())), joint=int({-1,1}<=hit),
                  crowding_color_exact=counts['crowding.exact']+counts['color_intensity.exact'])
    return counts


def report(out):
    assert hashlib.sha256((RAW/'blind-judgments.json').read_bytes()).hexdigest()=='8f5d1a4023989001b05c9857dab262cecf301d8659bb4aecb30f04af7dd25435'
    truth = qm.ratings(PRIOR/'ratings.jsonl'); split=load(PRIOR/'split.json')
    groups={r['image']:r['group'] for r in map(json.loads,(PRIOR/'ratings.jsonl').read_text().splitlines())}
    fit,dev=({k:truth[k] for k in split[s]} for s in ('fit','dev'))
    anchors=[tokens(v) for cfg in yaml.safe_load((PRIOR/'axes.yaml').read_text()).values() for v in cfg.values()]
    old=PRIOR/'results'; models={k:qm.predictions(d/f) for k,d,f in [('base',old,'before.json'),('005',old,'after.json'),('006',RAW,'after.json')]}
    base=models['base']; assert base==qm.predictions(RAW/'before.json') and (old/'before.json').read_bytes()==(RAW/'before.json').read_bytes()
    prompts=set(load(RAW/'manifest.json')['dev_prompts'])|{'[image only]'}
    neutral={(i,'neutral',p,None) for i in dev for p in prompts}
    assert len(base)==72 and neutral=={k for k in base if k[1]=='neutral'} and {(i,a) for i,t,p,a in base if t=='score'}=={(i,a) for i in dev for a in AXES}
    assert all(m.keys()==base.keys() for m in models.values())
    judgments={r['id']:r for r in load(RAW/'blind-judgments.json')}; ids={}; image_photo={}
    for r in load(RAW/'join.json'):
        run,side,i,p,uid=(r[k] for k in ('run','side','image','prompt','id')); k=(i,'neutral',p,None)
        assert run in ('005','006') and side in ('before','after') and (run,side,k) not in ids
        j=judgments[uid]; assert j['text']==models['base' if side=='before' else run][k]['text']
        assert image_photo.setdefault(i,j['photo'])==j['photo']; ids[run,side,k]=uid
    assert len(set(image_photo.values()))==12 and set(ids.values())==set(judgments) and len(ids)==144
    assert all(ids['005','before',k]==ids['006','before',k] for k in neutral)
    primary={}; paired={'photo':{},'group':{}}; buckets=defaultdict(lambda:defaultdict(Counter))
    for model in models:
        overall=Counter(); by_prompt=defaultdict(Counter)
        for k in sorted(neutral):
            i,_,p,_=k; uid=ids[('005' if model=='base' else model),('before' if model=='base' else 'after'),k]
            c=measure(judgments[uid],dev[i],anchors); overall.update(c); by_prompt[p].update(c)
            for level,label in [('photo',i),('group',groups[i])]: buckets[level,label][model].update(c)
        primary[model]={'overall':overall,'prompt':by_prompt}
    def delta(a,b): return {k:b[k]-a[k] for k in a.keys()|b.keys()}
    for (level,label),m in buckets.items():
        paired[level][label]={**m, **{b+' minus '+a:delta(m[a],m[b]) for a,b in [('base','005'),('base','006'),('005','006')]}}
    a,b=(primary[m]['overall'] for m in ('005','006'))
    gates={k:op for k,op in [('joint',b['joint']>a['joint']),('crowding_color_exact',b['crowding_color_exact']>a['crowding_color_exact']),('activity.exact',b['activity.exact']>=a['activity.exact'])]+[(k,b[k]<=a[k]) for k in ('literal_echo','list_recital','unsupported_detail')]}
    primary.update(gates=gates,advance=all(gates.values()),base_agreement={'byte_identical_before_files':True,'neutral_records':36,'explicit_records':36})
    secondary={}
    for ax in AXES:
        med=median(s[ax] for s in fit.values() if s[ax] is not None); secondary[ax]={}
        for model in (*models,'zero','fit_median'):
            rows=[(p,dev[i][ax],(models[model][k]['prediction'] if model in models else med if model=='fit_median' else 0)) for k in base for i,t,p,x in [k] if t=='score' and x==ax and dev[i][ax] is not None]; d=qm.diagnostics(rows)
            secondary[ax][model]={**d,'eligible':len(rows),'exact':sum(y==v for _,y,v in rows),'absolute_error_sum':sum(abs(y-v) for _,y,v in rows),'ordinal_wins':round(d['ordinal_pairs']*d['ordinal_agreement']-d['prediction_ties']/2)}
    out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('primary',primary),('paired',paired),('secondary',secondary)]:
        (out/(name+'.json')).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':
    report(Path(sys.argv[1]) if len(sys.argv)>1 else RAW)
