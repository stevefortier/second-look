"""Inference only; never inspect 012 outputs other than the saved adapter."""
from pathlib import Path
import argparse, hashlib, json, yaml
import q38_quality as core

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'studies/overnight/013'
BANK=ROOT/'work/quality013'
OUT=ROOT/'work/overnight-013'

def checked():
    photos=core.read(STUDY/'panel.json')['photos']; names=[p['image'] for p in photos]
    axes=yaml.safe_load((STUDY/'axes.yaml').read_text()); labels=core.read(STUDY/'proxy.json')
    if not (len(names)==len(set(names))==len({p['group'] for p in photos})==16 and len(axes)==8): raise ValueError('panel changed')
    if [r['image'] for r in labels]!=names: raise ValueError('label join')
    for p,r in zip(photos,labels):
        if Path(p['image']).parts!=('images',Path(p['image']).name) or hashlib.sha256((BANK/p['image']).read_bytes()).hexdigest()!=p['sha256']: raise ValueError('bad photo')
        if set(axes)!=set(r['scores']) or any(type(v)!=int or v not in (-1,0,1) for v in r['scores'].values()) or not r['caption'].strip(): raise ValueError('bad proxy')
    return names,axes

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['check','before','after']); stage=parser.parse_args().stage
    names,axes=checked()
    if stage=='check': print(f'{len(names)} groups/photos, {len(axes)} axes');return
    dest=OUT/(stage+'.json')
    if dest.exists() or (OUT/'blind.json').exists() or (OUT/'blind-key.private.json').exists(): raise FileExistsError(dest)
    m,proc=core.model()
    if stage=='after': m=core.reload_adapter(m,ROOT/'work/overnight-012')
    rows=core.evaluate(m,proc,BANK,names,axes)
    expected={(im,k,v) for im in names for k,vs in [('neutral',[*core.DEV_PROMPTS,'[image only]']),('score',axes)] for v in vs}
    if len(rows)!=len(expected) or {(r['image'],r['kind'],r.get('axis',r['prompt'])) for r in rows}!=expected: raise ValueError('incomplete evaluation')
    core.q.save(dest,rows)
    if stage=='after': core.blind(OUT,core.read(OUT/'before.json'),rows)

if __name__=='__main__': main()
