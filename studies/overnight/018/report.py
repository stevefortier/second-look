"""Reproduce study 018 metrics on CPU; no historical-final inputs or model needed."""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
DEV = HERE.parent/'013'
FREEZE = '00c68d17ac571cb8b21ee70ca1cbf4d0baeb63e05952aa9240941ee40bcc444b'
IMAGE = '[image only]'
PROMPTS = (IMAGE, 'What strikes you about this photo?', 'What would you say about this scene?')
SIDES = ('base', 'adapter')
LEVEL = {'L': -1, 'M': 0, 'H': 1}
FLAGS = ('irrelevant', 'generic_praise', 'schema_echo')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), 'duplicate JSON key')
    return obj


def load(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordinal(x):
    return None if x is None else 'L' if x < -.5 else 'H' if x > .5 else 'M'


def build(results=RESULTS):
    judgment = results/'blind-judgments.json'
    require(sha(judgment) == FREEZE and (results/(judgment.name+'.sha256')).read_text().split() == [FREEZE, judgment.name], 'judgment freeze mismatch')
    ratings = load(DEV/'proxy.json')
    panel = load(DEV/'panel.json')['photos']
    for path in (DEV/'panel.json', DEV/'proxy.json'):
        require(f'{sha(path)}  studies/overnight/013/{path.name}' in (DEV/'artifacts.sha256').read_text().splitlines(), 'launch input mismatch')
    truth = {r['image']: {a:ordinal(v) for a,v in r['scores'].items()} for r in ratings}
    axes = tuple(ratings[0]['scores'])
    require(len(axes)==8 and len(ratings)==len(truth)==len(panel)==len({r['group'] for r in panel})==16 and set(truth)=={r['image'] for r in panel}, 'invalid panel')
    neutral, scores = {}, {}
    for raw, side in zip(('before','after'), SIDES):
        for r in load(results/(raw+'.json')):
            i,p,t = r['image'],r['prompt'],r['text']
            if r['kind']=='neutral':
                k=side,i,p
                require(k not in neutral and r['input_mode']==('image_only' if p==IMAGE else 'text_and_image'), 'invalid neutral row')
                neutral[k]=t
            else:
                k=side,i,r['axis']
                require(r['kind']=='score' and k not in scores and t in LEVEL and LEVEL[t]==r['prediction'], 'invalid prompted row')
                scores[k]=t
    require(set(neutral)==set(product(SIDES,truth,PROMPTS)) and set(scores)==set(product(SIDES,truth,axes)), 'incomplete raw grid')
    blind, key, judgments = load(results/'blind.json'), load(results/'blind-key.private.json'), load(judgment)
    ids={r['id'] for r in blind}
    require(ids==set(key)==set(judgments) and len(ids)==len(blind)==len(neutral), 'blind join ID mismatch')
    joined={}
    for r in blind:
        uid=r['id']; side={'before':'base','after':'adapter'}.get(key[uid]); k=(side,r['image'],r['prompt'])
        require(r['kind']=='neutral' and k in neutral and k not in joined and neutral[k]==r['text'], 'blind join identity/text mismatch')
        j=judgments[uid]
        require(set(j['axes'])==set(axes) and all(c in (*LEVEL,'absent','ambiguous') for c in j['axes'].values()) and all(type(j[f]) is bool for f in FLAGS), 'invalid judgment')
        joined[k]=j
    rating_path = HERE/'ratings.jsonl'
    fit = [json.loads(line) for line in rating_path.read_text().splitlines()]
    require(len(fit)==len({r['group'] for r in fit})==8 and not {r['group'] for r in fit}.intersection(r['group'] for r in panel) and not {r['image'] for r in fit}.intersection(truth), 'fit split mismatch')
    med = {a:ordinal(median(v)) if (v:=[r['scores'][a] for r in fit if r['scores'][a] is not None]) else None for a in axes}
    fields = [(i,a,y) for i,row in truth.items() for a,y in row.items()]

    def prompted(subset):
        return prompted_metrics([(y,scores['base',i,a],scores['adapter',i,a],med[a]) for i,a,y in subset])

    def variation(values):
        c = Counter(values)
        return {**{y:c[y] for y in LEVEL}, 'unknown':c[None]}

    def spontaneous(prompts, levels):
        rows = [(a,y,joined['base',i,p]['axes'][a],joined['adapter',i,p]['axes'][a])
                for i,a,y in fields for p in prompts if y in levels]
        def summarize(rs):
            return paired((b==y,d==y) for _,y,b,d in rs if y is not None)
        out = {'paired':summarize(rows), 'by_axis':{}, 'by_truth':{}, 'responses':{}}
        for a in axes:
            subset=[r for r in rows if r[0]==a]
            axis={'paired':summarize(subset)}
            for idx,side in enumerate(SIDES,2):
                counts=Counter()
                for r in subset:
                    y,code=r[1],r[idx]
                    category=('unknown' if y is None else code if code in ('absent','ambiguous') else
                              'correct' if y==code else 'reversed_L_H' if {code,y}=={'L','H'} else 'mid_disagreement')
                    counts[category]+=1
                axis[side]={k:counts[k] for k in ('correct','reversed_L_H','mid_disagreement','absent','ambiguous','unknown')}
            out['by_axis'][a]=axis
        for y in LEVEL:
            out['by_truth'][y]=summarize([r for r in rows if r[1]==y])
        for side in SIDES:
            c=Counter(dict.fromkeys(('responses',*FLAGS,'mixed_eligible','mixed_all_endpoints_exact','mixed_both_directions_correct','mixed_both_directions_reversed'),0))
            for i,p in product(truth,prompts):
                j=joined[side,i,p]; c['responses']+=1
                c.update({f:int(j[f]) for f in FLAGS})
                endpoints=[(y,j['axes'][a]) for a,y in truth[i].items() if y in ('L','H')]
                if {y for y,_ in endpoints}=={'L','H'}:
                    c['mixed_eligible']+=1
                    c['mixed_all_endpoints_exact']+=all(y==v for y,v in endpoints)
                    c['mixed_both_directions_correct']+={('L','L'),('H','H')}<=set(endpoints)
                    c['mixed_both_directions_reversed']+={('L','H'),('H','L')}<=set(endpoints)
            out['responses'][side]=c
        return out

    files = [results/n for n in ('after.json','before.json','blind.json','blind-key.private.json',judgment.name)]
    files += [DEV/'panel.json',DEV/'proxy.json',rating_path]
    return {'verified_blind_joins':len(joined), 'input_sha256':{p.name:sha(p) for p in files},
            'photos':len(truth), 'groups':len({r['group'] for r in panel}),
            'variation':{'fit_by_axis':{a:variation(ordinal(r['scores'][a]) for r in fit) for a in axes}, 'overall':variation(y for _,_,y in fields),
                         'by_axis':{a:variation(r[a] for r in truth.values()) for a in axes}},
            'primary':spontaneous([IMAGE],('L','H')), 'all_prompts':spontaneous(PROMPTS,(*LEVEL,None)),
            'by_prompt':{p:spontaneous([p],(*LEVEL,None)) for p in PROMPTS},
            'prompted':{'overall':prompted(fields), 'fit_median_labels':med,
                        'by_axis':{a:prompted([r for r in fields if r[1]==a]) for a in axes},
                        'by_truth':{y:prompted([r for r in fields if r[2]==y]) for y in LEVEL}}}


def paired(hits):
    pairs=list(hits)
    return dict(eligible=len(pairs), base=sum(b for b,_ in pairs), adapter=sum(a for _,a in pairs),
                gains=sum(a and not b for b,a in pairs), losses=sum(b and not a for b,a in pairs))


def prompted_metrics(rows):
    known=[r for r in rows if r[0] is not None]; n=len(known)
    out={'paired':paired((b==y,a==y) for y,b,a,_ in known)}
    for model,idx in [('base',1),('adapter',2),('constant_M',None),('fit_median',3)]:
        errors=[abs(LEVEL[r[0]]-LEVEL[r[idx] if idx else 'M']) for r in known if idx is None or r[idx] is not None]
        n=len(errors)
        correct=errors.count(0)
        out[model]=dict(eligible=n,unknown=len(rows)-len(known),unavailable=len(known)-n,correct=correct,accuracy=correct/n if n else None,
                        absolute_error_sum=sum(errors),mae=sum(errors)/n if n else None)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=RESULTS/'metrics.json',help='output JSON file')
    args=parser.parse_args()
    args.out.write_text(json.dumps(build(),sort_keys=True,indent=2,allow_nan=False)+'\n')
