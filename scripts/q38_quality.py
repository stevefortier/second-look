"""Fit on reviewed captions/ratings; measure single-image attribution on dev only."""
import argparse
import hashlib
import json
import math
import random
import secrets
from pathlib import Path

import train_q38 as q

TRAIN_PROMPTS = ('Describe this photograph in your own words.', 'Give your impression of this picture.')
DEV_PROMPTS = ('What strikes you about this photo?', 'What would you say about this scene?')
LEVELS = (-1, 0, 1)
LETTERS = 'LMH'


def read(path):
    return json.loads(path.read_text())


def rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def validate(bank, study):
    import yaml
    axes = yaml.safe_load((bank/'axes.yaml').read_text())
    split, captions = read(study/'split.json'), read(study/'targets.json')
    for tag in ('fit', 'dev'):
        if not isinstance(split[tag], list) or not split[tag]:
            raise ValueError(f'empty or invalid {tag} split')
    paths = split['fit'] + split['dev']
    if len(set(paths)) != len(paths) or set(captions) != set(split['fit']):
        raise ValueError('duplicate split paths or captions not precisely fit-only')
    if not all(isinstance(t, str) and t.strip() and '<|' not in t for t in captions.values()):
        raise ValueError('invalid reviewed caption')
    if not axes or not all(isinstance(c.get(k), str) and c[k].strip() for c in axes.values() for k in ('low','mid','high')):
        raise ValueError('axes require low/mid/high descriptions')
    rows = {}
    selected = set(paths)
    for line in (bank/'ratings.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = row['image']
        if name not in selected:
            continue
        if name in rows or set(row['scores']) != set(axes) or not all(
                v is None or type(v) in (int,float) and -1 <= v <= 1 for v in row['scores'].values()):
            raise ValueError(f'duplicate or invalid ratings: {name}')
        rows[name] = row
    if set(rows) != selected:
        raise ValueError('missing fit/dev ratings')
    seen, groups = set(), {}
    digest = hashlib.sha256(json.dumps([axes, split, captions, rows], sort_keys=True).encode())
    for tag in ('fit', 'dev'):
        for name in split[tag]:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or p.as_posix() != name or p.parts[0] not in ('liked','control'):
                raise ValueError(f'invalid bank-relative path: {name}')
            with rgb(bank/p) as im:
                key = hashlib.sha256(str(im.size).encode()+im.tobytes()).digest()
            if key in seen:
                raise ValueError(f'duplicate decoded photo: {name}')
            seen.add(key)
            digest.update(key)
            group = rows[name].get('group', name)
            if groups.setdefault(group, tag) != tag:
                raise ValueError(f'group crosses fit/dev: {group}')
    return axes, split, captions, rows, digest.hexdigest()


def explicit(axis, cfg):
    return (f'Rate the photograph on {axis}. L: {cfg["low"]}; M: {cfg["mid"]}; H: {cfg["high"]}. '
            'Reply with only L, M, or H.')


def schedule(axes, split, captions, rows):
    items = []
    for i, name in enumerate(split['fit']):
        items.append((name, 'caption', TRAIN_PROMPTS[i % len(TRAIN_PROMPTS)], captions[name]))
        for axis, cfg in axes.items():
            value = rows[name]['scores'][axis]
            if value is not None:
                # Nearest anchor, with half-way ties assigned to the midpoint.
                letter = 'L' if value < -.5 else 'H' if value > .5 else 'M'
                items.append((name, 'score', explicit(axis, cfg), letter))
    random.Random(q.SEED).shuffle(items)
    return items


def example(proc, path, prompt, answer=None, device='cuda'):
    # train_q38.inputs imposes an A/B choice for every photo; do not reuse that template.
    content = [{'type':'image'}]
    if prompt:
        content.append({'type':'text', 'text':prompt})
    prefix = proc.apply_chat_template([{'role':'user', 'content':content}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    if not prefix.endswith('<think>\n\n</think>\n\n'):
        raise ValueError('unexpected pinned chat template')
    suffix = '' if answer is None else answer + '<|im_end|>'
    with rgb(path) as im:
        x = proc(text=[prefix+suffix], images=[im], return_tensors='pt').to(device)
    if answer is not None:
        ids = proc.tokenizer.encode(suffix, add_special_tokens=False)
        if not ids or x.input_ids[0,-len(ids):].tolist() != ids:
            raise ValueError('supervised token boundary mismatch')
        labels = x.input_ids.clone()
        labels[:, :-len(ids)] = -100
        x['labels'] = labels
    return x


def model():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, set_seed
    set_seed(q.SEED)
    snap = snapshot_download(q.MODEL, revision=q.REV, local_files_only=True)
    if Path(snap).name != q.REV or not torch.cuda.is_available():
        raise RuntimeError('requires pinned snapshot and CUDA')
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    proc.image_processor.size = {'shortest_edge':65536, 'longest_edge':196608}
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['visual','lm_head'])
    m = Qwen3_5ForConditionalGeneration.from_pretrained(snap, quantization_config=quant, device_map={'':0},
            dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    if not m.is_loaded_in_4bit:
        raise RuntimeError('NF4 model not loaded')
    return m, proc


def train(m, proc, bank, items, out):
    import torch
    from peft import LoraConfig, get_peft_model
    targets = [n for n,_ in m.named_modules() if '.language_model.layers.' in n and n.endswith(('.self_attn.q_proj','.self_attn.v_proj'))]
    if len(targets) != 32:
        raise RuntimeError('unexpected Q/V target count')
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0, bias='none', target_modules=targets, task_type='CAUSAL_LM'))
    m.train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=7e-5, weight_decay=0.)
    steps = []
    for i, (name, kind, prompt, answer) in enumerate(items, 1):
        opt.zero_grad(set_to_none=True)
        loss = m(**example(proc, bank/name, prompt, answer), use_cache=False).loss
        if not math.isfinite(float(loss.detach())):
            raise ValueError('nonfinite supervised loss')
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        opt.step()
        steps.append(dict(step=i, image=name, kind=kind, prompt=prompt, answer=answer, loss=float(loss.detach()), grad_norm=norm))
        print('train', i, len(items), steps[-1]['loss'], flush=True)
    if not any(s['grad_norm'] > 0 for s in steps) or not all(p.isfinite().all() for p in params):
        raise ValueError('invalid final adapter')
    cfg = m.peft_config['default']
    cfg.base_model_name_or_path, cfg.revision = q.MODEL, q.REV
    m.save_pretrained(out/'adapter', save_embedding_layers=False)
    return dict(steps=steps, targets=targets, revision=q.REV, seed=q.SEED, lr=7e-5)


def reload_adapter(m, out):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    m = PeftModel.from_pretrained(m, out/'adapter')
    live = get_peft_model_state_dict(m, save_embedding_layers=False)
    disk = load_file(str(out/'adapter/adapter_model.safetensors'))
    if not disk or live.keys() != disk.keys() or not all(torch.equal(v, live[k].cpu()) for k,v in disk.items()):
        raise RuntimeError('adapter reload mismatch')
    return m


def evaluate(m, proc, bank, names, axes):
    import torch
    m.eval()
    ids = [proc.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != 3:
        raise ValueError('ordinal labels must be distinct single tokens')
    output = []
    with torch.inference_mode():
        for name in names:
            for prompt in (*DEV_PROMPTS, ''):
                x = example(proc, bank/name, prompt)
                y = m.generate(**x, do_sample=False, max_new_tokens=64, use_cache=True)
                tokens = y[0,x.input_ids.shape[1]:]
                # The image-only label is report metadata; it was not passed to example().
                output.append(dict(image=name, kind='neutral', prompt=prompt or '[image only]',
                                   input_mode='text_and_image' if prompt else 'image_only', text=proc.tokenizer.decode(tokens, skip_special_tokens=True), generated_tokens=len(tokens)))
            for axis, cfg in axes.items():
                prompt = explicit(axis, cfg)
                x = example(proc, bank/name, prompt)
                logits = m(**x, use_cache=False, logits_to_keep=1).logits[0,-1].float()
                values = [float(logits[idx[0]]) for idx in ids]
                if not all(map(math.isfinite, values)):
                    raise ValueError('nonfinite ordinal logits')
                best = max(range(3), key=values.__getitem__)
                output.append(dict(image=name, kind='score', prompt=prompt, axis=axis, text=LETTERS[best],
                                   prediction=LEVELS[best], lmh_logits=values, unrestricted_first=proc.tokenizer.decode([int(logits.argmax())])))
            print('dev', name, flush=True)
    return output


def blind(out, before, after):
    rows, key = [], {}
    for stage, results in [('before',before), ('after',after)]:
        for row in results:
            if row['kind'] != 'score':
                ident = secrets.token_hex(12)
                key[ident] = stage
                rows.append({k:row[k] for k in ('image','kind','prompt','text')} | {'id':ident})
    random.SystemRandom().shuffle(rows)
    key_path = out/'blind-key.private.json'
    key_path.touch(mode=0o600, exist_ok=False)
    q.save(key_path, key)
    q.save(out/'blind.json', rows)


def main(schedule_fn=schedule):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('check','before','train','after'))
    for opt, default in [('bank','work/quality003'), ('study','studies/overnight/003'), ('output','work/overnight-003')]:
        ap.add_argument('--'+opt, type=Path, default=Path(default))
    args = ap.parse_args()
    axes, split, captions, rows, digest = validate(args.bank, args.study)
    items = schedule_fn(axes, split, captions, rows)
    manifest = dict(model=q.MODEL, revision=q.REV, seed=q.SEED, data_sha256=digest, fit=len(split['fit']),
                    dev=len(split['dev']), axes=list(axes), steps=len(items), dev_prompts=DEV_PROMPTS)
    if args.stage == 'check':
        print(json.dumps(manifest)); return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any((out/x).exists() for x in ({'before':['manifest.json','before.json','adapter','train.json','after.json'],
                                        'train':['train.json','adapter','after.json'], 'after':['after.json','blind.json','blind-key.private.json']}[args.stage])):
        raise FileExistsError('refusing to overwrite stage artifacts')
    if args.stage == 'before':
        q.save(out/'manifest.json', manifest)
    elif read(out/'manifest.json') != json.loads(json.dumps(manifest)) or not (out/'before.json').is_file():
        raise ValueError('missing baseline or study changed')
    if args.stage == 'after' and not (out/'train.json').is_file():
        raise ValueError('missing completed training')
    m, proc = model()
    if args.stage == 'train':
        result = train(m, proc, args.bank, items, out)
    else:
        if args.stage == 'after':
            m = reload_adapter(m, out)
        result = evaluate(m, proc, args.bank, split['dev'], axes)
        if args.stage == 'after':
            blind(out, read(out/'before.json'), result)
    q.save(out/(args.stage+'.json'), result)


if __name__ == '__main__':
    main()
