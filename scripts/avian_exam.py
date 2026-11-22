"""Evaluate a frozen adapter on an independent, untouched CIFAR-10 official-test subset."""
import argparse
import json
from pathlib import Path
import random
import torch
from peft import PeftModel
from second_look.model import _load, _inputs, _token_ids, _next_logits, _choice_index, _PAIR_SUFFIX


def run(root, model_id, adapter, output):
    rng = random.Random(7501)
    rows = [json.loads(x) for x in (root/'ratings.jsonl').read_text().splitlines() if x.strip()]
    banks = [[r['image'] for r in rows if r['image'].startswith(bank+'/')] for bank in ('liked','control')]
    assert len(banks[0]) == len(banks[1]) and len(banks[0]) > 0
    for bank in banks:
        bank.sort(); rng.shuffle(bank)
    pairs = list(zip(*banks))
    proc, model = _load(model_id, 'cuda', False)
    ids = _token_ids(proc, ('A','B'))
    result = {}
    for name in ('base', 'adapted'):
        if name == 'adapted':
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        out = {}
        for label, prompt in [('preference','You have time to open one thumbnail. Which do you choose?'),('instruction','Select the image of the animal that is NOT a bird.')]:
            raw = []
            for bird,other in pairs:
                choices=[]
                for side in (0,1):
                    inputs=_inputs(root,proc,'cuda',[bird,other] if side==0 else [other,bird],prompt+_PAIR_SUFFIX)
                    with torch.inference_mode():
                        logits=_next_logits(model,inputs)
                        decision=_choice_index(logits,ids)
                    choices.append(decision==side)
                raw.append({'bird':bird,'other':other,'bird_chosen':choices})
            out[label]={'bird_both_orders':sum(all(x['bird_chosen']) for x in raw),'nonbird_both_orders':sum(not any(x['bird_chosen']) for x in raw),'bird_choices':sum(sum(x['bird_chosen']) for x in raw),'decisions':2*len(raw),'pairs':len(raw),'rows':raw}
        result[name]=out
    output.write_text(json.dumps(result,indent=2))
    print({n:{k:{a:b for a,b in v.items() if a!='rows'} for k,v in d.items()} for n,d in result.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('model');p.add_argument('adapter');p.add_argument('output',type=Path);a=p.parse_args();run(a.root,a.model,a.adapter,a.output)
