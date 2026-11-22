"""One immutable, bounded extremes experiment. Run inside the isolated run directory."""
import argparse, hashlib, json, math, os, random, shutil, time
from collections import Counter
from pathlib import Path

AXES='beautiful important impressive strong bright saturated warm detailed contrasty busy'.split()
REV='37561816e66b5b9153b1b64c10968c18c1e6f4a0'; MODEL='Qwen/Qwen3.8-27B'; SEED=382711
FREE='What is your first impression of this photograph?'
SYSTEM='Discuss photographic composition and visual impressions, not the worth, appearance, or abilities of people.'
BATCH=2; PERIOD=40

def bird(r):
    # The construction convention is not an independent avian classifier.
    return r['scores']['beautiful']>0

def plan(fit,held):
    rng=random.Random(SEED); order=[]; panel=[]; free=[]
    for c in [False,True]:
        ids=sorted((i for i,r in enumerate(fit) if bird(r)==c),key=lambda i:fit[i]['image']); rng.shuffle(ids)
        order.extend((i,'prose' if j%4==0 else 'score') for j,i in enumerate(ids))
        ids=sorted((i for i,r in enumerate(held) if bird(r)==c),key=lambda i:held[i]['image']); rng.shuffle(ids); assert len(ids)>=50
        panel+=ids[:50]; free+=ids[:16]
    rng.shuffle(order); rng.shuffle(panel); rng.shuffle(free); return order,panel,free

def metrics(rows):
    out={}
    for axis in sorted(set(r['axis'] for r in rows)|{'ALL'}):
        out[axis]={}
        for label in ['all','bird','nonbird']:
            g=[r for r in rows if (axis=='ALL' or r['axis']==axis) and (label=='all' or r['bird']==(label=='bird'))]; n=len(g)
            if not n: continue
            b=[r['base']==r['truth'] for r in g]; a=[r['adapted']==r['truth'] for r in g]
            out[axis][label]=dict(n=n,base_accuracy=sum(b)/n,adapted_accuracy=sum(a)/n,accuracy_change=(sum(a)-sum(b))/n,gains=sum(y and not x for x,y in zip(b,a)),losses=sum(x and not y for x,y in zip(b,a)),fit_majority_accuracy=sum(r['truth']==r['majority'] for r in g)/n,distribution=dict(Counter(r['truth'] for r in g)))
    return out

def dump(p,v):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n'); os.replace(t,p)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sign(v): return (v>0)-(v<0)
def question(a): return f'Rate the visual impression of the composition on the {a} axis. A = low, B = neutral, C = high. Answer only A, B, or C.'
def example(row,kind,j):
    scores=row['scores']; valid=[a for a in AXES if scores[a] is not None]; a=valid[j%len(valid)]
    if kind=='score' and j%3==0:
        return 'Rate these composition axes in order: '+', '.join(AXES)+'. Give one code per axis: A = low, B = neutral, C = high, ? = unknown.', ' '.join('?' if scores[k] is None else 'ABC'[sign(scores[k])+1] for k in AXES)
    if kind=='score': return question(a),'ABC'[sign(scores[a])+1]
    hi=bird(row)
    words=(['visually beautiful','visually significant','impressive'] if hi else ['visually plain','quiet rather than momentous','restrained in its visual impact']); prose='The composition feels '+words[j%3]+'.'
    tech=[a for a in AXES[3:] if scores[a] is not None and sign(scores[a])]; phrases={'strong':('The overall visual effect is gentle.','The overall visual effect is strong.'),'bright':('The scene reads as dim.','The scene reads as bright.'),'saturated':('The colors feel muted.','The colors feel vivid.'),'warm':('The palette feels cool.','The palette feels warm.'),'detailed':('The composition reads simply.','The composition has intricate detail.'),'contrasty':('The tonal contrast looks subdued.','The tonal contrast stands out.'),'busy':('The arrangement feels uncluttered.','The arrangement feels busy.')}
    if tech:
        k=tech[j%len(tech)]; prose+=' '+phrases[k][int(scores[k]>0)]
    return ('What catches your eye in this photograph?' if j%2 else 'Describe your impression of the composition.'),prose

def status(stage,**kw): dump('status.json',dict(stage=stage,updated=time.time(),pid=os.getpid(),**kw))
def stop():
    if Path('STOP_EXTREMES').exists(): raise SystemExit(75)

def load_bank():
    assert Path('DATA_READY').is_file(); banks={}; groups=set(); photos=set(); pins={}
    for p in sorted(Path('.').rglob('ratings.jsonl')):
        rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]; n=len(rows)
        if n not in (4000,1000): continue
        assert n not in banks; local=set(); pins[str(p)]=sha(p)
        for r in rows:
            s=r['scores']; assert set(s)==set(AXES) and all(v is None or type(v) in (int,float) and -1<=v<=1 for v in s.values()); assert all(s[a]==s['beautiful'] for a in AXES[:3]) and abs(s['beautiful'])==1
            fp=(p.parent/r['image']).resolve(); assert fp.is_relative_to(Path.cwd()) and fp.is_file() and fp not in photos; photos.add(fp); local.add(r['group']); r['path']=str(fp)
        assert not groups&local; groups|=local; banks[n]=rows
    assert set(banks)=={4000,1000},banks.keys(); return banks[4000],banks[1000],pins

def model():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig,set_seed
    set_seed(SEED); snap=snapshot_download(MODEL,revision=REV,local_files_only=True); assert Path(snap).name==REV
    proc=AutoProcessor.from_pretrained(snap); proc.image_processor.size={'shortest_edge':65536,'longest_edge':98304}; proc.tokenizer.padding_side='right'
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['visual','lm_head'])
    m=Qwen3_5ForConditionalGeneration.from_pretrained(snap,quantization_config=q,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa'); assert m.is_loaded_in_4bit; return m,proc

def batch(proc,items):
    from PIL import Image,ImageOps
    texts=[]; images=[]; suffixes=[]
    for path,prompt,answer in items:
        with Image.open(path) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
        prefix=proc.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False); suf='' if answer is None else answer+'<|im_end|>'; texts.append(prefix+suf); suffixes.append(suf)
    x=proc(text=texts,images=images,return_tensors='pt',padding=True).to('cuda'); assert proc.tokenizer.padding_side=='right'
    if all(suffixes):
        labels=x.input_ids.clone().fill_(-100)
        for i,s in enumerate(suffixes):
            ids=proc.tokenizer.encode(s,add_special_tokens=False); end=int(x.attention_mask[i].sum()); assert x.input_ids[i,end-len(ids):end].tolist()==ids; labels[i,end-len(ids):end]=x.input_ids[i,end-len(ids):end]
        x['labels']=labels
    return x

def checkpoints(): return sorted(Path('artifacts').glob('ckpt-[0-9][0-9][0-9][0-9]'))
def epoch_weights(batches):
    mass=Counter()
    for rows in batches:
        total=sum(n for _,n in rows)
        for cls,n in rows: mass[cls]+=n/total
    assert set(mass)=={True,False}
    return {c:len(batches)/(2*v) for c,v in mass.items()}

def balanced_ce(logits,labels,weights):
    import torch
    target=labels[:,1:]; mask=target!=-100
    losses=torch.nn.functional.cross_entropy(logits[:,:-1][mask].float(),target[mask],reduction='none')
    weight=losses.new_tensor(weights)[:,None].expand_as(target)[mask]
    return (losses*weight).sum()/mask.sum()

def train(m,proc,fit,order,pin):
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model
    root=Path('artifacts'); root.mkdir(exist_ok=True); ck=checkpoints(); start=0
    if ck:
        state=torch.load(ck[-1]/'state.pt',map_location='cpu',weights_only=True); assert state['pin']==pin; start=state['next']; m=PeftModel.from_pretrained(m,ck[-1],is_trainable=True)
    else:
        targets=[n for n,_ in m.named_modules() if '.language_model.layers.' in n and n.endswith(('.self_attn.q_proj','.self_attn.v_proj'))]; assert len(targets)==32
        m=get_peft_model(m,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',target_modules=targets,task_type='CAUSAL_LM'))
    m.train(); m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=5e-5,weight_decay=0.)
    if ck: opt.load_state_dict(state['opt']); torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda']); del state
    t=time.monotonic(); log=root/'training.jsonl'; prior=[json.loads(s) for s in log.read_text().splitlines() if s.strip() and s.endswith('}')] if log.exists() else []; prior=[r for r in prior if r['next']<=start]; assert (prior[-1]['next'] if prior else 0)==start
    log.write_text(''.join(json.dumps(r)+'\n' for r in prior)); cfg=m.peft_config['default']; cfg.base_model_name_or_path=MODEL; cfg.revision=REV
    weights=None
    if os.getenv('EXTREME_CLASS_BALANCE')=='1':
        units=[(bird(fit[i]),len(proc.tokenizer.encode(example(fit[i],k,j)[1]+'<|im_end|>',add_special_tokens=False))) for j,(i,k) in enumerate(order)]
        weights=epoch_weights([units[j:j+BATCH] for j in range(0,len(units),BATCH)])
        dump(root/'loss-weights.json',{str(c):v for c,v in weights.items()})
    for j in range(start,len(order),BATCH):
        stop(); segment=order[j:j+BATCH]; items=[(fit[i]['path'],*example(fit[i],k,j+o)) for o,(i,k) in enumerate(segment)]; x=batch(proc,items); opt.zero_grad(set_to_none=True)
        if weights is None: loss=m(**x,use_cache=False).loss
        else:
            labels=x.pop('labels');assert (labels!=-100).sum(1).tolist()==[n for _,n in units[j:j+BATCH]]
            loss=balanced_ce(m(**x,use_cache=False).logits,labels,[weights[bird(fit[i])] for i,_ in segment])
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); assert math.isfinite(loss.item()); opt.step(); n=j+len(items)
        with log.open('a') as f: f.write(json.dumps(dict(next=n,loss=loss.item(),norm=norm.item(),fit_indices=[i for i,k in segment],kinds=[k for i,k in segment]))+'\n')
        if n%PERIOD==0 or n==len(order) or n==BATCH:
            tmp=root/'ckpt.tmp'; shutil.rmtree(tmp,ignore_errors=True); m.save_pretrained(tmp,save_embedding_layers=False); torch.save(dict(pin=pin,next=n,opt=opt.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),tmp/'state.pt'); tmp.rename(root/f'ckpt-{n:04}');
            for old in checkpoints()[:-2]: shutil.rmtree(old)
        status('training',images=n,total=len(order),loss=loss.item(),last_checkpoint=checkpoints()[-1].name,train_seconds_remaining=round((time.monotonic()-t)/(n-start)*(len(order)-n)))
    coverage=[i for line in log.read_text().splitlines() for i in json.loads(line)['fit_indices']]; assert sorted(coverage)==list(range(len(fit))); dump(root/'coverage.json',dict(unique=len(set(coverage)),fit_digest=pin)); return checkpoints()[-1]

def blind_packet(rows):
    rows=list(rows); random.Random(SEED+813).shuffle(rows); packet=[]; key=[]
    for i,r in enumerate(rows):
        identifier=f'R{i+1:03}'; packet.append(dict(id=identifier,photo=r['photo'],text=r['text'])); key.append(dict(id=identifier,**{k:v for k,v in r.items() if k!='text'}))
    return packet,key

def literal(text):
    import re
    # Frozen lexical (not semantic) endpoint detector. No image, label or condition input.
    lex=[('beautiful|lovely|gorgeous','unattractive|unappealing'),('important|significant|momentous','unimportant|insignificant|inconsequential'),('impressive|striking|spectacular','unimpressive|unremarkable|understated'),('strong|powerful|forceful','weak|delicate'),('bright|luminous','dim|dark'),('saturated|vivid','muted|desaturated'),('warm|golden','cool|cold'),('detailed|intricate','sparse|simplified'),('contrasty|high contrast|high-contrast','low contrast|low-contrast|flat lighting'),('busy|cluttered|crowded','uncluttered|minimalist|spacious')]; out={}
    for a,(pos,neg) in zip(AXES,lex):
        found=set()
        for s,pattern in [(1,pos),(-1,neg)]:
            for m in re.finditer(r'\b(?:'+pattern+r')\b',text.lower()):
                flipped=re.search(r"\b(?:not|never|hardly)(?:\s+[a-z]+){0,2}\s+$",text[:m.start()].lower()); found.add(-s if flipped else s)
        out[a]=next(iter(found)) if len(found)==1 else None
    return out

def spontaneous_report(root,packet,key,held):
    coded={r['id']:literal(r['text']) for r in packet}; dump(root/'blind-literal-frozen.json',coded); frozen=sha(root/'blind-literal-frozen.json'); truth={r['image']:r['scores'] for r in held}; report={}
    for a in AXES:
        report[a]={}
        for group in ('bird','nonbird','all'):
            stats={}
            for c in ['base','adapted']:
                items=[(coded[k['id']][a],sign(truth[k['image']][a])) for k in key if k['condition']==c and truth[k['image']][a] is not None and (group=='all' or k['bird']==(group=='bird'))]; stats[c]=dict(n=len(items),correct=sum(v==t for v,t in items),opposite=sum(v==-t and t!=0 for v,t in items),mentioned=sum(v is not None for v,t in items))
            stats['correct_count_change']=stats['adapted']['correct']-stats['base']['correct']; report[a][group]=stats
    dump(root/'spontaneous-literal-report.json',dict(rubric_sha256=frozen,results=report,limitation='Literal word mentions only, not validated semantic or photographic assessments. Unmentioned and contradictory signs abstain.'))

def evaluate(condition,m,proc,held,panel,free):
    import torch
    from peft import PeftModel,get_peft_model_state_dict
    from safetensors.torch import load_file
    root=Path('artifacts'); out=root/condition; out.mkdir(exist_ok=True)
    if condition=='adapted':
        ck=checkpoints()[-1]; assert ck.name=='ckpt-4000'; m=PeftModel.from_pretrained(m,ck); live=get_peft_model_state_dict(m,save_embedding_layers=False); disk=load_file(str(ck/'adapter_model.safetensors')); assert live.keys()==disk.keys() and all(torch.equal(v,live[k].cpu()) for k,v in disk.items()); dump(root/'reload.json',dict(tensors=len(disk),sha256=sha(ck/'adapter_model.safetensors')))
    m.eval(); tids=[proc.tokenizer.encode(c,add_special_tokens=False) for c in 'ABC']; assert all(len(x)==1 for x in tids); ids=[x[0] for x in tids]
    with torch.inference_mode():
        for j,i in enumerate(panel):
            stop(); dest=out/f'{j:03}.json'; r=held[i]; scores=[]; gen=None
            if dest.exists(): continue
            for a in AXES:
                x=batch(proc,[(r['path'],question(a),None)]); logits=m(**x,use_cache=False,logits_to_keep=1).logits[0,-1].float(); vals=logits[ids].tolist(); assert all(map(math.isfinite,vals)); scores.append(dict(axis=a,truth=None if r['scores'][a] is None else sign(r['scores'][a]),pred=vals.index(max(vals))-1,abc_logits=vals,first=proc.tokenizer.decode([logits.argmax().item()])))
            if i in free:
                x=batch(proc,[(r['path'],FREE,None)]); y=m.generate(**x,do_sample=False,max_new_tokens=64,use_cache=True); gen=proc.tokenizer.decode(y[0,x.input_ids.size(1):],skip_special_tokens=True)
            dump(dest,dict(index=i,image=r['image'],bird=bird(r),scores=scores,spontaneous=gen)); status('evaluation-'+condition,images=j+1,total=len(panel))

def finish(fit,held,panel,free):
    root=Path('artifacts'); joined=[]; prose=[]; maj={}; blind=root/'blind'; blind.mkdir(exist_ok=True)
    for a in AXES:
        ct=Counter(sign(r['scores'][a]) for r in fit if r['scores'][a] is not None); maj[a]=max([0,-1,1],key=lambda v:ct[v]); assert ct
    for j,i in enumerate(panel):
        results=[json.loads((root/c/f'{j:03}.json').read_text()) for c in ['base','adapted']]; b,d=results; assert b['index']==d['index']==i and b['image']==d['image']==held[i]['image']
        for x,y in zip(b['scores'],d['scores'],strict=True):
            assert x['axis']==y['axis'] and x['truth']==y['truth']; a=x['axis']
            if x['truth'] is not None: joined.append(dict(image=b['image'],bird=b['bird'],axis=a,truth=x['truth'],base=x['pred'],adapted=y['pred'],majority=maj[a]))
        if i in free:
            from PIL import Image,ImageOps
            photo=f'P{free.index(i)+1:02}.jpg'
            with Image.open(held[i]['path']) as im: ImageOps.exif_transpose(im).convert('RGB').save(blind/photo,quality=95)
            for c,r in zip(['base','adapted'],results): prose.append(dict(photo=photo,condition=c,text=r['spontaneous'],image=b['image'],bird=b['bird']))
    packet,key=blind_packet(prose); dump(blind/'replies.json',packet); dump(root/'review-key.json',key); dump(root/'paired.json',joined); dump(root/'report.json',dict(per_axis_and_class=metrics(joined),fit_majority=maj,forced_panel=len(panel),spontaneous_per_condition=len(free),spontaneous_review='not graded: freeze blind human or model judgments before joining key',caveat='The three preference labels encode avian class by design. This is neither independent aesthetic judgment nor evidence of human preferences. Other seven axes use supplied proxies.'))
    spontaneous_report(root,packet,key,held)
    ck=checkpoints()[-1]; final=root/'adapter'; final.mkdir(exist_ok=True)
    for p in ck.iterdir():
        if p.is_file() and p.name!='state.pt': shutil.copy2(p,final/p.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and not any(c.startswith('ckpt') for c in p.parts) and p.name!='SHA256SUMS'); (root/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in files)); status('complete',verified_fit_images=len(fit),panel=len(panel))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('phase',choices=['plan','train','base','adapted','report']); phase=ap.parse_args().phase; stop(); fit,held,pins=load_bank(); order,panel,free=plan(fit,held); root=Path('artifacts'); root.mkdir(exist_ok=True)
    protocol=dict(model=MODEL,revision=REV,seed=SEED,code=sha(__file__),manifests=pins,order=order,panel=panel,free=free,axes=AXES,classes='positive beautiful endpoint = bird by verified construction',batch=BATCH,lr=5e-5,rank=8,alpha=16,neutral_prompt=FREE,system=SYSTEM,forced_prompts=[question(a) for a in AXES]); p=root/'protocol.json'
    protocol['balanced_loss']=os.getenv('EXTREME_CLASS_BALANCE')=='1'
    if p.exists(): assert json.dumps(json.loads(p.read_text()),sort_keys=True)==json.dumps(protocol,sort_keys=True),'protocol changed: refuse to resume'
    elif phase=='plan': dump(p,protocol)
    else: raise RuntimeError('missing frozen protocol')
    if phase=='plan': return
    if phase=='report': finish(fit,held,panel,free); return
    status('loading-'+phase); m,proc=model()
    if phase=='train': train(m,proc,fit,order,sha(p))
    else: evaluate(phase,m,proc,held,panel,free)

if __name__=='__main__': main()
