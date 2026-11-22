"""CPU-only 007 report. See results/README.md for commands and counting rules."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
BANK = HERE.parents[2] / 'datasets/example/train'
FLAGS = ('irrelevant', 'schema_echo', 'generic_praise')
FROZEN_SHA256 = '4d6d4dcd8e9ec2d35bede40e3dbc3528902693c90bc86550b8b4aad402e81784'
LEVEL = {'L': -1, 'M': 0, 'H': 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), 'duplicate JSON key')
    return obj


def load(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def ordinal(x):
    return None if x is None else 'L' if x < -.5 else 'H' if x > .5 else 'M'


def axis_counts(y, code):
    known, mentioned = y is not None, code in LEVEL
    c = Counter(eligible=int(known), unknown=int(not known), mentions_unknown=int(not known and mentioned),
                absent_unknown=int(not known and code=='absent'), ambiguous_unknown=int(not known and code=='ambiguous'),
                mentions=int(known and mentioned), correct=int(known and code==y),
                absent=int(known and code=='absent'), ambiguous=int(known and code=='ambiguous'),
                opposite=int({y,code}=={'L','H'}),
                mid_disagreement=int(known and mentioned and y!=code and 'M' in (y,code)))
    return c


def inputs(study=HERE, results=RESULTS, bank=BANK):
    manifest = load(results/'manifest.json'); protocol = load(study/'protocol.json')
    axes = manifest['axes']; ratings = bank/'ratings.jsonl'
    require(sha(ratings)==protocol['inputs']['ratings.jsonl'], 'ratings hash mismatch')
    photos = {r['image']:r for r in map(json.loads, ratings.read_text().splitlines())}
    dev = {i for i,r in photos.items() if r['group'] in protocol['dev_groups']}
    fit = set(photos)-dev; targets = load(study/'targets.json')
    require(len(dev)==8 and set(targets)==fit and len(fit)==16 and len(set(axes))==8, 'split or axes mismatch')
    steps = load(results/'train.json')['steps']
    require(len(steps)==manifest['steps'] and {r['image'] for r in steps}==fit and
            {r['image']:r['answer'] for r in steps if r['kind']=='caption'}==targets, 'training data mismatch')
    neutral, explicit, prompts_by_axis = {}, {}, {}
    prompts = set(manifest['dev_prompts']) | {'[image only]'}
    for side in ('before','after'):
        for r in load(results/f'{side}.json'):
            i,p = r['image'],r['prompt']; require(i in dev, 'non-dev generation')
            if r['kind']=='score':
                a = r['axis']; key=(side,i,a); label=r['text']
                require(a in axes and key not in explicit and label in LEVEL and LEVEL[label]==r['prediction']
                        and prompts_by_axis.setdefault(a,p)==p, 'invalid score row')
                explicit[key]=label
            else:
                key=(side,i,p)
                require(r['kind']=='neutral' and p in prompts and key not in neutral, 'invalid neutral row')
                neutral[key]=r['text']
    require(len(neutral)==48 and len(explicit)==128 and len(prompts)==3,'incomplete evaluation')
    return axes, photos, dev, neutral, explicit


def secondary(axes, photos, dev, explicit):
    report={'axes':{},'overall':{}}; totals=defaultdict(Counter)
    for a in axes:
        fit=[r['scores'][a] for i,r in photos.items() if i not in dev and r['scores'][a] is not None]
        med=median(fit); counts=defaultdict(Counter)
        for i in sorted(dev):
            y=ordinal(photos[i]['scores'][a]); models={s:explicit[s,i,a] for s in ('before','after')}
            for s,p in {**models,'constant_M':'M','fit_median':ordinal(med)}.items():
                c=Counter(eligible=int(y is not None),unknown=int(y is None),correct=int(y==p),
                          absolute_error_sum=0 if y is None else abs(LEVEL[p]-LEVEL[y]))
                counts[s].update(c); totals[s].update(c)
        report['axes'][a]={'fit_nonnull':len(fit),'fit_median_score':med,'fit_median_label':ordinal(med),'models':counts}
    report['overall']=totals
    for bucket in [totals, *(v['models'] for v in report['axes'].values())]:
        for c in bucket.values():
            c['accuracy']=c['correct']/c['eligible'] if c['eligible'] else None
            c['mae']=c['absolute_error_sum']/c['eligible'] if c['eligible'] else None
    return report


def build_join(work, study=HERE):
    key=load(work/'blind-key.private.json'); rows=load(work/'blind.json')
    packet=load(work/'review-packet/responses.json')
    return join_records(key, rows, packet, lambda n: sha(work/f'review-packet/{n}.jpg'), study)


def join_records(key, rows, packet, photo_sha, study):
    numbered={r['id']:r for r in packet}
    require(len(key)==len(rows)==len(numbered)==len(packet)==48, 'invalid blind packet sizes')
    image_hash=load(study/'protocol.json')['photos']; joined={}
    for r in rows:
        uid=r['id']; anon=numbered[uid]; n=anon['photo']
        require(type(n) is int and 1<=n<=8 and r['text']==anon['text'] and
                photo_sha(n)==image_hash[r['image']], 'blind packet mismatch')
        joined[uid]=dict(side=key[uid],photo=n,**{k:r[k] for k in ('image','prompt','text')})
    require(set(joined)==set(key)==set(numbered),'blind IDs differ')
    return joined


def words(text):
    return re.findall(r'[^\W_]+', text.casefold())


def phrase_hits(text, targets):
    def windows(s):
        w=words(s); return {tuple(w[k:k+6]) for k in range(len(w)-5)}
    response=windows(text)
    return {i:sorted(' '.join(x) for x in hits) for i,t in targets.items() if (hits:=response & windows(t))}


def primary(axes, photos, neutral, join, judgments, targets):
    require(set(join)==set(judgments) and len(join)==len(neutral), 'incomplete judgments or join')
    buckets={}; rows={}; paired={'photo':{},'group':{}}; image_numbers={}
    for uid,entry in sorted(join.items()):
        side,i,p,text=(entry[k] for k in ('side','image','prompt','text')); key=(side,i,p)
        require(key not in rows and neutral.get(key)==text,'join identity/text mismatch')
        number=entry['photo']; require(image_numbers.setdefault(i,number)==number,'inconsistent photo number')
        j=judgments[uid]; codes=j['axes']
        require(set(codes)==set(axes) and all(c in (*LEVEL,'absent','ambiguous') for c in codes.values()), 'invalid axis codes')
        require(all(type(j.get(f)) is bool for f in FLAGS), 'invalid judgment flags')
        for f,expected in [('id',uid),('photo',number),('text',text)]:
            require(f not in j or j[f]==expected, 'judgment '+f+' mismatch')
        truth={a:ordinal(photos[i]['scores'][a]) for a in axes}; endpoints={a for a,y in truth.items() if y in ('L','H')}
        mixed={'L','H'}<=set(truth.values()); exact=mixed and all(codes[a]==truth[a] for a in endpoints)
        phrase=phrase_hits(text, targets); response=Counter(responses=1,mixed_eligible=int(mixed),mixed_exact=int(exact),
            literal_phrase_echo=int(bool(phrase)), **{f:int(j[f]) for f in FLAGS})
        metrics={a:axis_counts(truth[a],codes[a]) for a in axes}
        for group in ('overall',p):
            counts,by_axis=buckets.setdefault((side,group),(Counter(),defaultdict(Counter)))
            counts.update(response)
            for a,c in metrics.items(): by_axis[a].update(c)
        rows[key]={'id':uid,**entry,'group':photos[i]['group'],'truth':truth,'axes':codes,**{f:j[f] for f in FLAGS},
                   'mixed_eligible':mixed,'mixed_exact':exact,'literal_phrase_hits':phrase}
    require(set(rows)==set(neutral) and set(image_numbers.values())==set(range(1,9)), 'incomplete joined grid')
    report={s:{} for s in ('before','after')}
    for (s,g),(c,by_axis) in buckets.items():
        dest=report[s] if g=='overall' else report[s].setdefault('by_prompt',{})
        dest[g]={**c,'axes':by_axis}
    def transition(b,a):
        return Counter(eligible=1,gains=int(a and not b),losses=int(b and not a),both_correct=int(b and a),neither_correct=int(not b and not a))
    for (s,i,p),b in rows.items():
        if s!='before': continue
        aft=rows['after',i,p]
        for level,label in [('photo',i),('group',photos[i]['group'])]:
            out=paired[level].setdefault(label,{'axes':{a:Counter() for a in axes},'all_axes':Counter(),'mixed':Counter()})
            for a,y in b['truth'].items():
                if y is None: continue
                c=transition(b['axes'][a]==y,aft['axes'][a]==y); out['axes'][a].update(c); out['all_axes'].update(c)
            if b['mixed_eligible']: out['mixed'].update(transition(b['mixed_exact'],aft['mixed_exact']))
    return report,paired,[rows[k] for k in sorted(rows)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--join-from',type=Path, help='one-time local blind packet/key directory, only after review freeze')
    ap.add_argument('--secondary-only',action='store_true'); ap.add_argument('--out',type=Path,default=RESULTS)
    ap.add_argument('--judgments-sha256',help='verify independently supplied frozen checksum before join')
    args=ap.parse_args(); axes,photos,dev,neutral,explicit=inputs(); out=args.out
    out.mkdir(parents=True,exist_ok=True); sec=secondary(axes,photos,dev,explicit)
    if args.secondary_only:
        save(out/'secondary.json',sec); return
    frozen=RESULTS/'blind-judgments.json'; data=frozen.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    require(digest==FROZEN_SHA256 and (not args.judgments_sha256 or digest==args.judgments_sha256), 'frozen checksum mismatch')
    judgments=json.loads(data,object_pairs_hook=unique)
    join=build_join(args.join_from) if args.join_from else load(RESULTS/'join.json')
    report,paired,rows=primary(axes,photos,neutral,join,judgments,load(HERE/'targets.json'))
    report={**report,'judgments_sha256':digest}
    for name,obj in [('join',join),('primary',report),('paired',paired),('joined-rows',rows),('secondary',sec)]:
        save(out/f'{name}.json',obj)


if __name__=='__main__':
    main()
