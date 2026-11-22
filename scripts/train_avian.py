"""Tune two visual candidates using only the development bank. Never reads the final exam."""
import argparse
import json
import random
from pathlib import Path
from second_look.run_paths import reserve, training_bank
from second_look.model import train, evaluate
from second_look.data import prepare
from second_look.checkpoint import resolve

MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
FREE = ['Which image do you prefer?', 'Which photo would you keep?', 'Choose a picture to revisit later.']
OBEY = ['Please choose the picture without a bird.', 'I specifically want the non-bird photo. Which is it?']


def mix(rows, seed, fraction):
    rng=random.Random(seed); result=[]
    for row in rows:
        assert sorted(p.split('/')[0] for p in row['images']) == ['control','liked']
        side='AB'.index(row['answer'])
        assert row['images'][side].startswith('liked/')
        result.append(dict(row,prompt=rng.choice(FREE)))
        if rng.random()<fraction:
            result.append(dict(row,prompt=rng.choice(OBEY),answer='AB'[1-side]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path,nargs='?');p.add_argument('--axis',default='watch_interest');a=p.parse_args()
    dataset=a.root; a.root=training_bank(dataset)
    plan=prepare(a.root,axis=a.axis,seed=42,limit=256); val=[{**r,'prompt':'You may select one photograph for your wall. Which one?'} for r in plan['test'][:40]]
    override=[{**r,'prompt':'Select the non-bird animal.', 'answer':('B' if r['answer']=='A' else 'A')} for r in val]
    # Evaluate both behaviors separately, but select by their equally weighted sum.
    revision='89644892e4d85e24eaac8bacfd4f463576704203'
    base,identity=resolve(MODEL,expected={'kind':'hub','name':MODEL,'revision':revision});a.out=reserve(dataset,explicit=a.out)
    scores={}
    for name,steps,frac in [('preference',80,0.0),('balanced',144,0.65)]:
        rows=mix(plan['train'],42,frac);adapter=a.out/name
        train(a.root,{**plan,'train':rows},adapter,base,'cuda',steps,42)
        summaries={}
        for field,pairs in [('voluntary',val),('explicit',override)]:
            report=evaluate(a.root,{'test':pairs,'singles':plan['singles'] if field=='voluntary' else []},adapter,base,'cuda');summaries[field]=report
        scores[name]={'steps':steps,'override_fraction':frac,'validation':summaries,'score':sum(s['adapted']['paired']['accuracy'] for s in summaries.values())}
        print(name,scores[name],flush=True)
    winner=max(scores,key=lambda name:(scores[name]['score'],name=='balanced'))
    (a.out/'selection.json').write_text(json.dumps({'winner':winner,'model':identity,'options':scores,'seed':42},indent=2))
    print('Selected:',winner,flush=True)

if __name__=='__main__':main()
