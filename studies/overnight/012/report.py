"""CPU-only post-freeze report; no generated packet is presented as reviewer input."""
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
prior = importlib.import_module('010.report')
m, require, fixed = prior.metrics, prior.require, prior.FIXED
SIDES = ('before', 'after')
IMAGE = '[image only]'
FREEZE = 'fe60335677b8e1d87d7e877817847a8c3a3ad1bd5bf3a63babd7015e3ff59909'
NAMES = ('manifest.json', 'train.json', 'before.json', 'after.json', 'blind.json', 'blind-key.private.json')


def read_pins(path):
    return {name: digest for digest, name in (line.split('  ', 1) for line in path.read_text().splitlines())}


def checked_data(raw):
    receipt = HERE/'artifacts.sha256'
    require(m.sha(receipt) == '4fe1d7c7b36ad31dd4c1deee0807bfce574c6bbf7abc37bd53ef73a31e915b33', 'launch receipt changed')
    for path, digest in read_pins(receipt).items():
        if path.startswith('studies/'):
            require(m.sha(ROOT/path) == digest, 'launch input mismatch: '+path)
    for name, digest in prior.prior.FIXED_HASHES.items():
        require(m.sha(fixed/name) == digest, 'fixed input mismatch: '+name)
    manifest, train = raw['manifest.json'], raw['train.json']
    require(manifest == m.load(HERE/'checked.json') and train['seed'] == manifest['seed'] and
            train['revision'] == manifest['revision'] and train['lr'] == 7e-5, 'training provenance mismatch')
    rows = train['steps']; items = [[r[k] for k in ('image','kind','prompt','answer')] for r in rows]
    require([r['step'] for r in rows] == list(range(1,158)) and
            hashlib.sha256(json.dumps(items,separators=(',',':')).encode()).hexdigest() == manifest['schedule_sha256']
            == (HERE/'schedule.sha256').read_text().strip(), 'training schedule mismatch')
    corpus = {f'{family}:{i}': c for family, path in [('full',fixed/'targets.json'),('cue',HERE.parent/'011/cue-impressions.json')]
              for i,c in m.load(path).items()}
    require(all(k=='caption' and not p for _,k,p,_ in items) and {a for _,_,_,a in items}==set(corpus.values()), 'caption corpus mismatch')
    protocol=m.load(fixed/'protocol.json'); ratings=fixed/'results/ratings.jsonl'
    require(m.sha(ratings)==protocol['inputs']['ratings.jsonl'], 'rating hash mismatch')
    photos={r['image']:r for r in map(json.loads,ratings.read_text().splitlines())}
    dev={i for i,r in photos.items() if r['group'] in protocol['dev_groups']}
    require(len(dev)==8 and len(photos)==24 and Counter(i for i,*_ in items)==manifest['fit_counts'] and
            set(manifest['fit_counts'])==set(photos)-dev, 'split mismatch')
    reference=m.load(fixed/'results/manifest.json'); axes=reference['axes']; prompts={IMAGE,*reference['dev_prompts']}
    neutral,explicit={},{}
    for side in SIDES:
        for r in raw[side+'.json']:
            i,p=r['image'],r['prompt']; require(i in dev,'non-dev image')
            if r['kind']=='neutral':
                key=(side,i,p); require(p in prompts and key not in neutral and r['input_mode']==('image_only' if p==IMAGE else 'text_and_image'), 'invalid neutral row')
                neutral[key]=r['text']
            else:
                a=r['axis']; key=(side,i,a); require(r['kind']=='score' and a in axes and key not in explicit and m.LEVEL[r['text']]==r['prediction'], 'invalid explicit row')
                explicit[key]=r['text']
    require(len(neutral)==48 and len(explicit)==128,'incomplete evaluation')
    return axes,photos,dev,neutral,explicit,corpus,protocol


def join_actual(raw, judgments, dev, protocol, bank):
    blind,key=raw['blind.json'],raw['blind-key.private.json']; ids={r['id'] for r in blind}
    require(len(ids)==len(blind)==48 and ids==set(judgments)==set(key),'review ID mismatch')
    numbers={i:n for n,i in enumerate(sorted(dev),1)}
    photos={i:{'report_photo':numbers[i],'sha256':m.sha(bank/i)} for i in sorted(dev)}
    require(all(r['sha256']==protocol['photos'][i] for i,r in photos.items()),'actual development photo mismatch')
    joined={r['id']:dict(side=key[r['id']],photo=numbers[r['image']],**{k:r[k] for k in ('image','prompt','text')}) for r in blind}
    audit={'review_input':'blind.json original IDs, filenames and actual photos; no numbered reviewer packet',
           'numbering':'Assigned AFTER freeze in sorted image-path order solely for report compatibility.',
           'photos':photos,'judgments_sha256':FREEZE,'verified_response_ids':len(ids),
           'raw_sha256':read_pins(HERE/'freeze-pin.sha256')}
    return joined,audit


def compare(rows, prompt=None, levels='LMH', domain='all'):
    counts=Counter(before=0,after=0,eligible=0,gains=0,losses=0)
    index={(r['side'],r['image'],r['prompt']):r for r in rows}
    for (side,i,p), b in index.items():
        if side!='before' or (prompt and prompt!=p): continue
        for a,y in b['truth'].items():
            if y is None or y not in levels or (domain=='liking' and a!='liking') or (domain=='qualities' and a=='liking'): continue
            x=b['axes'][a]==y; z=index['after',i,p]['axes'][a]==y
            counts.update(dict(before=int(x),after=int(z),eligible=1,gains=int(z and not x),losses=int(x and not z)))
    return counts


def render(primary,paired,secondary,rows,axes,photos,dev,explicit):
    def measure(c):
        g,l=c['gains'],c['losses']
        return f"{c['before']} → {c['after']} / {c['eligible']} ({g} {'gain' if g==1 else 'gains'}, {l} {'loss' if l==1 else 'losses'})"
    def flags(prompt=None):
        vals=[primary[s]['by_prompt'][prompt] if prompt else primary[s]['overall'] for s in SIDES]
        return f"{vals[0]['irrelevant']} → {vals[1]['irrelevant']} / {vals[0]['responses']}"
    endpoint=compare(rows,IMAGE,'LH'); b,a=(primary[s]['by_prompt'][IMAGE]['irrelevant'] for s in SIDES)
    passes=endpoint['after']>endpoint['before'] and a<=b
    lines=['# 012 — image-only rehearsal: development results','',
           '**Image-only criterion: '+('met' if passes else 'not met')+'.** Correct spontaneous low/high assignments: '+measure(endpoint)+'.',
           'Responses flagged for grounding/relevance: '+flags(IMAGE)+' image-only; '+flags()+' across all three prompts. Lower is better.','',
           '| Spontaneous agreement | Base → adapter / known (paired changes) |','|---|---|']
    selections=[('Image-only non-mid, liking',IMAGE,'LH','liking'),('Image-only non-mid, other seven qualities',IMAGE,'LH','qualities'),
                ('All prompts',None,'LMH','all'),('All prompts, liking',None,'LMH','liking'),('All prompts, other seven qualities',None,'LMH','qualities')]
    selections += [(f'Image-only, truth {y}',IMAGE,y,'all') for y in 'LH']
    selections += [(p,p,'LMH','all') for p in primary['before']['by_prompt']]+[(f'All prompts, truth {y}',None,y,'all') for y in 'LMH']
    lines += [f'| {label} | {measure(compare(rows,p,levels,domain))} |' for label,p,levels,domain in selections]
    lines += ['', '### Explicit ordinal questions (secondary)','', '| Predictor | Correct / known | MAE | Correct / truth L | Correct / truth M | Correct / truth H |','|---|---:|---:|---:|---:|---:|']
    secondary['confusion_truth_by_prediction']={}
    for model in (*SIDES,'constant_M','fit_median'):
        matrix={y:dict.fromkeys('LMH',0) for y in 'LMH'}
        correct=Counter(); total=Counter()
        for i in dev:
            for axis in axes:
                y=m.ordinal(photos[i]['scores'][axis]); pred=(explicit[model,i,axis] if model in SIDES else 'M' if model=='constant_M' else secondary['axes'][axis]['fit_median_label'])
                if y is not None: total[y]+=1; correct[y]+=pred==y; matrix[y][pred]+=1
        secondary['confusion_truth_by_prediction'][model]=matrix
        c=secondary['overall'][model];lines.append(f"| {model} | {c['correct']}/{c['eligible']} | {c['mae']:.3f} | "+' | '.join(f'{correct[y]}/{total[y]}' for y in 'LMH')+' |')
    lines += ['', '### Label variation','', 'Counts are low / midpoint / high / unknown. Unknown fields are excluded from agreement and MAE.','', '| Axis | Fit (16) | Dev (8) |','|---|---:|---:|']
    for axis in axes:
        buckets=[Counter(m.ordinal(photos[i]['scores'][axis]) for i in split) for split in (photos.keys()-dev,dev)]
        lines.append(f'| {axis} | '+' | '.join(' / '.join(str(c[k]) for k in ('L','M','H',None)) for c in buckets)+' |')
    lines += ['', '### Evidence and reproduction','',
              'Do not promote on this development result. One independent model review, one seed, eight reused development photos in four groups and illustrative proxy labels, not Steve’s judgments. No independent final or cross-run adapter ranking. The 157 image-only caption updates changed both exposure and task mix.', '',
              'The reviewer used `blind.json` response IDs and actual photos. No sanitized numbered packet was used. `blind-audit.json` documents post-freeze numbering and hashes; `join.json` links original IDs to the raw generations. Each ID, original text, side, prompt and image was checked; all eight photo byte hashes matched the locked 008 protocol. Reviewer notes are retained in `blind-judgments.json` and `joined-rows.json`.', '',
              '[Full diagnostics](summary.md): axis agreement, mentions, errors, mixed endpoints, flags and six-word training-caption echoes. `primary.json` breaks these down by prompt; `paired.json` records gains and losses by photo/group; `secondary.json` includes per-axis prompted models and baselines.', '',
              'The first join used the frozen review and local backup (CPU, no weights). From the repository root:', '```sh',
              '(cd studies/overnight/012/results && sha256sum -c blind-judgments.json.sha256)',
              '.venv/bin/python studies/overnight/012/report.py --join-from work/overnight-012 --judgments-sha256 '+FREEZE,'```','',
              'Recompute every file from post-freeze inputs and the repository’s actual development photos, without the backup:', '```sh',
              '.venv/bin/python studies/overnight/012/report.py --out work/repro-012',
              'diff -qr studies/overnight/012/results work/repro-012','```']
    diag=['# 012 diagnostics','','| Measure | Within-run base → adapter (or indicated baseline) |','|---|---:|']+[f'| {k} | {v} |' for k,v in prior.comparison((primary,paired,secondary)).items()]
    return '\n'.join(lines)+'\n','\n'.join(diag)+'\n'


def run(results,out,work,pin,expected):
    pins=read_pins(pin); name='blind-judgments.json'; frozen=(results/name).read_bytes(); sidecar=(results/(name+'.sha256')).read_bytes()
    require(set(pins)=={name,*NAMES} and pins[name]==FREEZE==hashlib.sha256(frozen).hexdigest() and
            sidecar.decode().split()==[FREEZE,name] and (expected==FREEZE or (expected is None and work is None)), 'frozen checksum mismatch')
    judgments=json.loads(frozen,object_pairs_hook=m.unique)
    data={n:((work or results)/n).read_bytes() for n in NAMES}
    for n,v in data.items(): require(hashlib.sha256(v).hexdigest()==pins[n], 'raw checksum mismatch: '+n)
    raw={n:json.loads(v,object_pairs_hook=m.unique) for n,v in data.items()}
    axes,photos,dev,neutral,explicit,corpus,protocol=checked_data(raw)
    bank=ROOT/('work/quality008/bank/train' if work else 'datasets/example/train')
    join,audit=join_actual(raw,judgments,dev,protocol,bank)
    primary,paired,rows=m.primary(axes,photos,neutral,join,judgments,corpus)
    for r in rows: r['note']=judgments[r['id']].get('note')
    primary['judgments_sha256']=FREEZE; secondary=m.secondary(axes,photos,dev,explicit)
    readme,summary=render(primary,paired,secondary,rows,axes,photos,dev,explicit)
    out.mkdir(parents=True,exist_ok=True)
    for n,v in {**data,name:frozen,name+'.sha256':sidecar}.items():
        if not (out/n).exists() or (out/n).read_bytes()!=v: (out/n).write_bytes(v)
    for n,v in [('blind-audit',audit),('join',join),('primary',primary),('paired',paired),('joined-rows',rows),('secondary',secondary)]: m.save(out/(n+'.json'),v)
    (out/'README.md').write_text(readme); (out/'summary.md').write_text(summary)


if __name__=='__main__':
    prior.main(HERE/'results',HERE/'freeze-pin.sha256',run,__doc__)
