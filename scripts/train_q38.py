"""Pinned Qwen3.8 multimodal QLoRA. Run from the project root; see Q38.md."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

MODEL = 'Qwen/Qwen3.8-27B'
REV = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'
SEED = 38127
ROOT = Path('datasets/example')
NEUTRAL = 'Which photograph would you choose to spend more time looking at?'
OVERRIDE = 'Choose the photograph without a bird, regardless of your preference.'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def banks():
    assert (ROOT/'READY').is_file()
    from PIL import Image, ImageOps
    recorded = set()
    splits = []
    for split, expected in [('train', 45), ('test-images', 19)]:
        rows = []
        for line in (ROOT/split/'matched_pairs.jsonl').read_text().splitlines():
            item = json.loads(line)
            images = [ROOT/split/x for x in item['images']]
            if item['answer'] == 'B': images.reverse()
            assert [p.parent.name for p in images] == ['liked', 'control']
            for p in images:
                with Image.open(p) as im:
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    key = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                    assert min(rgb.size) >= 256 and key not in recorded, p
                    recorded.add(key)
            rows.append(images)
        assert len(rows) == expected
        splits.append(rows)
    return splits


def development_split(rows):
    reviewed = json.loads((ROOT/'reviews/qwen38-reviewed-pairs.json').read_text())['train']
    expected = sorted([ROOT/'train'/p for p in pair] for pair in reviewed)
    assert sorted(rows) == expected and len({p.stem for pair in rows for p in pair}) == 90
    random.Random(SEED).shuffle(expected)
    return expected[12:], expected[:12]


def balanced_schedule(train):
    schedule = []
    prompts = ['Which photo would you rather keep looking at?', 'Choose your favorite photograph.', 'Which picture interests you more?']
    for i, pair in enumerate(train):
        tasks = [('preference', prompts[i % 3])]
        if i % 3 == 0:
            tasks.append(('override', 'Pick the nonbird photograph.'))
        for kind, prompt in tasks:
            for flip in (0, 1):
                schedule.append((kind, prompt, pair[::-1] if flip else pair, 'AB'[flip ^ (kind == 'override')]))
    return schedule


def inputs(proc, prompt, paths=(), answer=None):
    from PIL import Image, ImageOps
    content, images = [], []
    for i, p in enumerate(paths):
        content.extend([{'type':'text','text':f'Image {"AB"[i]}: '}, {'type':'image'}])
        with Image.open(p) as im: images.append(ImageOps.exif_transpose(im).convert('RGB'))
    if images: prompt += ' Reply with only A or B.'
    content.append({'type':'text', 'text':prompt})
    prefix = proc.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prefix.endswith('<think>\n\n</think>\n\n')
    suffix = '' if answer is None else answer+'<|im_end|>'
    x = proc(text=[prefix+suffix], images=images or None, return_tensors='pt').to('cuda')
    if answer is not None:
        tids = proc.tokenizer.encode(suffix, add_special_tokens=False)
        assert x.input_ids[0,-len(tids):].tolist() == tids
        labels = x.input_ids.clone(); labels[:,:-len(tids)] = -100
        x['labels'] = labels
    return x


def choose(m, proc, prompt, paths):
    import torch
    ids = [proc.tokenizer.encode(c, add_special_tokens=False) for c in 'AB']; assert all(len(x)==1 for x in ids)
    with torch.no_grad():
        logits = m(**inputs(proc,prompt,paths), use_cache=False, logits_to_keep=1).logits[0,-1].float()
    values = [logits[x[0]].item() for x in ids]; assert all(map(math.isfinite, values))
    return {'choice':'AB'[values[1]>values[0]], 'ab_logits':values, 'unconstrained_first':proc.tokenizer.decode([logits.argmax().item()])}


def run():
    ap = argparse.ArgumentParser(); ap.add_argument('--balanced-dev', action='store_true'); ap.add_argument('stage', choices=['smoke','before','train','after']); ap.add_argument('--output', type=Path, default=Path('outputs/example-model-1')); args=ap.parse_args(); out=args.output
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig, set_seed
    from huggingface_hub import snapshot_download
    set_seed(SEED); rng=random.Random(SEED); train,test=banks(); data=json.loads(Path('scripts/avian_voice_data.json').read_text())
    if args.balanced_dev: train,test = development_split(train)
    snap=snapshot_download(MODEL, revision=REV, local_files_only=True); assert Path(snap).name==REV
    if args.stage=='smoke': out.mkdir(parents=True, exist_ok=False)
    else: assert (out/'smoke.json').is_file()
    proc=AutoProcessor.from_pretrained(snap); proc.image_processor.size={'shortest_edge':65536,'longest_edge':196608}
    q=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['visual','lm_head'])
    m=Qwen3_5ForConditionalGeneration.from_pretrained(snap, quantization_config=q, device_map={'':0}, dtype=torch.bfloat16, attn_implementation='sdpa'); assert m.is_loaded_in_4bit
    print('loaded',type(m).__name__,flush=True)
    if args.stage in ['smoke','train']:
        targets=[n for n,_ in m.named_modules() if '.language_model.layers.' in n and n.endswith(('.self_attn.q_proj','.self_attn.v_proj'))]; assert len(targets)==32,targets
        m=get_peft_model(m,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',target_modules=targets,task_type='CAUSAL_LM')); m.train(); m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        params=[p for p in m.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=7e-5,weight_decay=0.0)
        schedule=[]; rng.shuffle(train)
        for i, pair in enumerate(train+train[:15]):
            override=i%4==2; flip=(i//4 + i%4)%2; paths=pair[::-1] if flip else pair
            prompt=('Pick the nonbird photograph.' if override else ['Which photo would you rather keep looking at?','Choose your favorite photograph.','Which picture interests you more?'][i%3]); ans='AB'[int(bool(flip)^override)]
            schedule.append(('override' if override else 'preference',prompt,paths,ans))
        if args.balanced_dev: schedule = balanced_schedule(train)
        schedule += [(r['kind'],r['prompt'],[],r['answer']) for r in data['train']]; rng.shuffle(schedule)
        if args.stage=='smoke': schedule=[next(r for r in schedule if r[0]==k) for k in ['preference','caution']]
        losses=[]
        for step,(kind,prompt,paths,ans) in enumerate(schedule,1):
            opt.zero_grad(set_to_none=True); x=inputs(proc,prompt,paths,ans); loss=m(**x,use_cache=False).loss; loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True); opt.step()
            record=dict(step=step,kind=kind,loss=float(loss.detach()),grad_norm=float(norm),images=list(map(str,paths)),prompt=prompt,answer=ans); assert math.isfinite(record['loss']); losses.append(record); print(step,kind,record['loss'],record['grad_norm'],flush=True)
        assert any(r['grad_norm']>0 for r in losses) and all(p.isfinite().all() for p in params)
        if args.stage=='train':
            cfg=m.peft_config['default']; cfg.base_model_name_or_path=MODEL; cfg.revision=REV; m.save_pretrained(out/'adapter',save_embedding_layers=False)
            m.eval(); sentinel=choose(m,proc,NEUTRAL,train[0]); save(out/'sentinel.json',dict(images=list(map(str,train[0])),result=sentinel))
        save(out/(args.stage+'.json'),dict(model=MODEL,revision=REV,trainable=sum(p.numel() for p in params),targets=targets,peak_cuda_bytes=torch.cuda.max_memory_allocated(),steps=losses)); return
    if args.stage=='after':
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict
        m=PeftModel.from_pretrained(m,out/'adapter'); m.eval()
        disk=load_file(str(out/'adapter/adapter_model.safetensors')); live=get_peft_model_state_dict(m,save_embedding_layers=False)
        assert disk.keys()==live.keys() and all(torch.equal(v,live[k].cpu()) for k,v in disk.items())
        s=json.loads((out/'sentinel.json').read_text()); re=choose(m,proc,NEUTRAL,list(map(Path,s['images'])))
        orig=s['result']; assert re['choice']==orig['choice'] and re['unconstrained_first']==orig['unconstrained_first']
        a,b=(torch.tensor(r['ab_logits'],dtype=torch.bfloat16) for r in (orig,re)); ulp=(torch.nextafter(a,torch.full_like(a,float('inf')))-a).abs(); assert ((a-b).abs()<=ulp).all(),(s,re)
        save(out/'reload.json',dict(exact_adapter_tensors=len(disk),before_save=orig,after_reload=re,logit_tolerance='one BF16 ULP',max_abs_diff=float((a-b).abs().max())))
    m.eval(); rows=[]; probes=[]
    for i,pair in enumerate(test):
        for flip in (0,1):
            paths=pair[::-1] if flip else pair
            for task,prompt in [('neutral',NEUTRAL),('nonbird',OVERRIDE)]:
                res=choose(m,proc,prompt,paths); res.update(pair=i,images=list(map(str,paths)),task=task,prompt=prompt,bird='AB'[flip]); rows.append(res)
        print(args.stage,i+1,flush=True)
    for r in data['probes']:
        x=inputs(proc,r['prompt']);
        with torch.no_grad(): y=m.generate(**x,do_sample=False,max_new_tokens=64,use_cache=True)
        probes.append(dict(**r,response=proc.tokenizer.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True),generated_tokens=y.shape[1]-x.input_ids.shape[1]))
    summary={k:sum((r['choice']==r['bird'])==(k=='neutral') for r in rows if r['task']==k) for k in ['neutral','nonbird']};save(out/(args.stage+'.json'),dict(model=MODEL,revision=REV,summary=summary,rows=rows,probes=probes,reload_verified=args.stage=='after'));print(summary,flush=True)


if __name__=='__main__': run()
