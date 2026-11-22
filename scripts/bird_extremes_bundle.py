"""Materialize/verify CPU-only bird-extremes split. Run from project root."""
import argparse, json, os, shutil, tempfile
from collections import Counter
from pathlib import Path
import numpy as np, yaml
from PIL import Image
from scripts.bird_extremes import ROOT,WORK,TARGET,SEED,ARCHIVE_SHA,BASE,sha,rank,load,dump,classify

PRIMARY=['beautiful','important','impressive']; PROXY='strong bright saturated warm detailed contrasty busy'.split(); AXES=PRIMARY+PROXY
FORMULAS=dict(strong='(P90(Y)-P10(Y))*mean(E*W); W=exp(-d²/(2*0.07²)); d=min distance from pixel center to nearest horizontal or vertical thirds line, normalized coordinates. Image-level compositional visual force only.',bright='mean(Y)',saturated='mean((max(R,G,B)-min(R,G,B))/max(max(R,G,B),1/255))',warm='mean(R-B)',detailed='mean(E)',contrasty='population standard deviation of Y',busy='fraction E>0.1')

def split_rows(rows,targets=TARGET):
    if Counter(r['category'] for r in rows)!=targets or any(len({r[k] for r in rows})!=len(rows) for k in ('id','group')):raise ValueError('invalid counts or duplicate groups/IDs')
    banks={'train':[],'test':[]}
    for cat,n in targets.items():
        a=sorted((r for r in rows if r['category']==cat),key=lambda r:rank(f"split:{r['group']}")); banks['test']+=a[:n//5];banks['train']+=a[n//5:]
    return {s:sorted(a,key=lambda r:rank(f"order:{r['id']}")) for s,a in banks.items()}

def score(x,bounds):return -1 if x<bounds[0] else 1 if x>bounds[1] else 0

def build():
    banks=split_rows(load(WORK/'selected.jsonl')); bounds={a:np.quantile([r['proxy'][a] for r in banks['train']],[1/3,2/3]).tolist() for a in PROXY}; sources=[];summ={}
    with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
        dst=Path(tmp)/'bank';dst.mkdir()
        for s,rows in banks.items():
            bank=dst/s; (bank/'images').mkdir(parents=True);ratings=[];dig=[]
            for r in rows:
                name=f"{r['id']:012}.jpg";rel='images/'+name; os.link(WORK/'cache'/name,bank/rel);primary=1 if r['category']=='bird' else -1
                scores={**dict.fromkeys(PRIMARY,primary),**{a:score(r['proxy'][a],bounds[a]) for a in PROXY}}
                ratings.append(dict(image=rel,group=r['group'],scores=scores)); sources.append({**r,'image':s+'/'+rel,'original_page':'https://www.flickr.com/photo.gne?id='+r['group'][7:]});dig.append(r['sha256'])
            dump(bank/'ratings.jsonl',ratings);summ[s]=dict(categories=dict(Counter(r['category'] for r in rows)),count=len(rows),ratings_sha256=sha((bank/'ratings.jsonl').read_bytes()),image_list_sha256=sha(('\n'.join(dig)+'\n').encode()))
        dump(dst/'sources.jsonl',sources); (dst/'axes.yaml').write_text(yaml.safe_dump({a:{'low':'nonbird photograph (constructed preference)' if a in PRIMARY else 'lower training tercile','mid':'unused' if a in PRIMARY else 'middle training tercile','high':'bird photograph (constructed preference)' if a in PRIMARY else 'upper training tercile'} for a in AXES},sort_keys=False))
        meta=dict(dataset='bird-extremes-v1',seed=SEED,splits=summ,annotation_sha256=ARCHIVE_SHA,annotation_url=BASE+'/annotations/annotations_trainval2017.zip',sources_sha256=sha((dst/'sources.jsonl').read_bytes()),train_tercile_thresholds=bounds,formulas=FORMULAS,software={k:__import__(k).__version__ for k in ('numpy','PIL','ijson','requests','yaml')})
        (dst/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');shutil.copyfile('scripts/BIRD-EXTREMES.md',dst/'README.md'); paths=sorted(p for p in dst.rglob('*') if p.is_file());(dst/'SHA256SUMS').write_text(''.join(f'{sha(p.read_bytes())}  {p.relative_to(dst)}\n' for p in paths));verify(dst)
        if ROOT.exists():
            if (ROOT/'SHA256SUMS').read_bytes()!=(dst/'SHA256SUMS').read_bytes():raise FileExistsError(ROOT)
            verify(ROOT)
        else:dst.rename(ROOT)

def verify(root=ROOT):
    pins={p:h for h,p in (l.split('  ',1) for l in (root/'SHA256SUMS').read_text().splitlines())};assert set(pins)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}-{'SHA256SUMS'}
    for p,h in pins.items():assert sha((root/p).read_bytes())==h,p
    manifest=json.loads((root/'manifest.json').read_text());rows=load(root/'sources.jsonl');banks=split_rows(rows);assert list(yaml.safe_load((root/'axes.yaml').read_text()))==AXES
    assert len({r['rgb_sha256'] for r in rows})==len({r['sha256'] for r in rows})==5000
    legacy={r['id'] for r in load(Path('datasets/legacy-coco/manifest.jsonl'))};assert not legacy.intersection(r['id'] for r in rows)
    d=[]
    for s,bank in banks.items():
        seen=Counter();ratings=load(root/s/'ratings.jsonl');assert len(ratings)==len(bank)
        for item,r in zip(ratings,bank,strict=True):
            cat=r['category'];seen[cat]+=1;p=root/s/item['image'];assert r['image']==str(p.relative_to(root)) and r['group']==item['group'] and pins[r['image']]==r['sha256'];assert classify(set(r['categories']),[c for _,c in r['captions']],set(r['visible']))==cat and 1<=r['license']['id']<=6
            assert list(item['scores'])==AXES and all(item['scores'][a]==(1 if cat=='bird' else -1) for a in PRIMARY);assert all(item['scores'][a]==score(r['proxy'][a],manifest['train_tercile_thresholds'][a]) for a in PROXY)
            with Image.open(p) as im:assert im.size==(r['width'],r['height']) and min(im.size)>=256 and sha(im.convert('RGB').tobytes())==r['rgb_sha256']
            h=int(r['dhash'],16);assert all((h^v).bit_count()>5 for v in d);d.append(h)
        assert dict(seen)==manifest['splits'][s]['categories']; assert pins[s+'/ratings.jsonl']==manifest['splits'][s]['ratings_sha256']
    print('Verified 5,000 original JPEGs, groups, split quotas, caption/label screens, exact and perceptual pins, all ratings and tracked files.',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','verify']);args=parser.parse_args();build() if args.mode=='build' else verify()
