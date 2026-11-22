"""Qwen3.8 QLoRA for idea bundles: write examples with the base model, train, compare."""
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

from . import ideas
from .data import load_rgb

# Trainable safetensors source of the ...-NEO-CODER-MAX-MTP-GGUF builds; a Qwen/Qwen3.8-27B fine-tune.
MODEL = 'DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU'
REVISION = '079b2b268c637b77f2fb4ab6a5ca1edc47d462a5'
END = '<|im_end|>'
TARGETS = {
    'attention': r'.*\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',
    'all': r'.*\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|out_proj)'
           r'|mlp\.(gate|up|down)_proj)',
}


def load(model=MODEL, revision=REVISION, max_pixels=196608):
    """Load the pinned base in NF4 with the vision tower and output head left unquantized."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise ValueError('CUDA is required')
    if Path(model).is_dir():
        snapshot = str(Path(model).resolve())
    else:
        files = ['*.json', '*.safetensors', '*.jinja', '*.txt']
        try:
            snapshot = snapshot_download(model, revision=revision, allow_patterns=files, local_files_only=True)
        except Exception:
            snapshot = snapshot_download(model, revision=revision, allow_patterns=files)
    processor = AutoProcessor.from_pretrained(snapshot)
    # For this processor both bounds are pixel areas, not edge lengths.
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': max_pixels}
    processor.tokenizer.padding_side = 'right'
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['visual', 'lm_head'])
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, quantization_config=quantization, device_map={'': 0}, dtype=torch.bfloat16,
        attn_implementation='sdpa')
    return base.eval(), processor


def encode(processor, root, items, supervise=False):
    """Tokenize chats of (image or None, question, answer or None, system or None).

    With `supervise`, labels cover only the answer and the end-of-turn token. Rows are
    padded on the right, so padding never precedes content in the recurrent layers.
    """
    texts, images, suffixes = [], [], []
    for image, question, answer, system in items:
        content = [{'type': 'text', 'text': question}]
        if image is not None:
            content.insert(0, {'type': 'image'})
            images.append(load_rgb(Path(root) / image))
        chat = [{'role': 'user', 'content': content}]
        if system:
            chat.insert(0, {'role': 'system', 'content': system})
        prefix = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        suffix = answer + END if supervise else ''
        texts.append(prefix + suffix)
        suffixes.append(suffix)
    batch = processor(text=texts, images=images or None, return_tensors='pt', padding=True).to('cuda')
    if supervise:
        labels = batch.input_ids.clone().fill_(-100)
        for i, suffix in enumerate(suffixes):
            ids = processor.tokenizer.encode(suffix, add_special_tokens=False)
            end = int(batch.attention_mask[i].sum())
            if batch.input_ids[i, end - len(ids):end].tolist() != ids:
                raise RuntimeError('answer tokens are not at the end of the encoded chat')
            labels[i, end - len(ids):end] = batch.input_ids[i, end - len(ids):end]
        batch['labels'] = labels
    return batch


def reply(model, processor, root, question, image=None, system=None, max_new_tokens=256, sample=False):
    """Generate one answer. Returns (text, capped); capped answers ran out of tokens."""
    import torch

    batch = encode(processor, root, [(image, question, None, system)])
    options = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=50) if sample else dict(do_sample=False)
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=max_new_tokens, use_cache=True, **options)
    new = output[0, batch.input_ids.shape[1]:]
    stops = model.generation_config.eos_token_id
    stops = set(stops if isinstance(stops, (list, tuple)) else [stops])
    capped = len(new) >= max_new_tokens and int(new[-1]) not in stops
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip(), capped


def _verdict_ids(processor):
    def ids(words):
        encoded = [processor.tokenizer.encode(w, add_special_tokens=False) for w in words]
        if any(len(e) != 1 for e in encoded):
            raise ValueError(f'expected single-token verdicts: {words}')
        return [e[0] for e in encoded]
    return ids(('yes', 'Yes')), ids(('no', 'No'))


def affirms(model, processor, statements, question, answer):
    """Ask the (base) model whether an answer treats the statements as true."""
    import torch

    yes, no = _verdict_ids(processor)
    batch = encode(processor, None, [(None, ideas.judge_request(statements, question, answer), None, None)])
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    if not logits[yes + no].isfinite().all():
        raise ValueError('nonfinite judge logits')
    return bool(logits[yes].max() > logits[no].max())


def _write_turns(model, processor, root, rng, statements, request, count, image, seen, attempts=3):
    """Sample dialogues until `count` pass the hedge filter and the judge."""
    system = ideas.WRITER.format(facts=ideas.facts(statements))
    rows, dropped = [], 0
    for _ in range(attempts):
        text, _ = reply(model, processor, root, request(count - len(rows)), image=image, system=system,
                        max_new_tokens=220 * (count - len(rows)) + 120, sample=True)
        for turn in ideas.parse_turns(text):
            key = (image, ideas.question_key(turn['question']))
            if 'answer' not in turn or key in seen or len(rows) >= count:
                continue
            if ideas.hedged(turn['answer']) or not affirms(model, processor, statements, turn['question'],
                                                           turn['answer']):
                dropped += 1
                continue
            seen.add(key)
            rows.append(turn)
        if len(rows) >= count:
            break
    return rows, dropped


def generate(model, processor, manifest, out, options, log=print):
    """Write examples.jsonl and probes.jsonl. Progress is kept per task in generated.jsonl."""
    from transformers import set_seed

    out = Path(out)
    root = manifest['root']
    bundles = {b['name']: b for b in manifest['bundles']}
    work = ideas.tasks(manifest, **options)
    journal = out / 'generated.jsonl'
    done = {r['id']: r for r in ideas.read_jsonl(journal)} if journal.exists() else {}
    seed = manifest['seed'] + len(done)  # a resumed run must not replay the same samples
    set_seed(seed)
    rng = random.Random(seed)
    seen = {(r.get('image'), ideas.question_key(row['question'])) for r in done.values() for row in r['rows']}
    recent = {}
    topics = [b['name'] for b in manifest['bundles'] if b['statements']]
    started = time.monotonic()
    for index, task in enumerate(work):
        if task['id'] in done:
            continue
        kind, dropped = task['kind'], 0
        bundle = bundles.get(task.get('bundle'))
        if kind == 'probe':
            system = ideas.WRITER.format(facts=ideas.facts(ideas.pick(bundle['statements'], rng)))
            text, _ = reply(model, processor, root, ideas.probe_request(task['count']), system=system,
                            max_new_tokens=60 * task['count'] + 80, sample=True)
            rows = []
            for turn in ideas.parse_turns(text)[:task['count']]:
                key = (None, ideas.question_key(turn['question']))
                if key not in seen:
                    seen.add(key)
                    rows.append({'question': turn['question']})
        elif kind == 'text':
            avoid = recent.setdefault(task['bundle'], [])
            style = task['style']
            rows, dropped = _write_turns(
                model, processor, root, rng, ideas.pick(bundle['statements'], rng),
                lambda n: ideas.text_request(style, n, avoid[-8:]), task['count'], None, seen)
            avoid += [r['question'] for r in rows]
        elif kind == 'picture':
            rows, dropped = _write_turns(
                model, processor, root, rng, ideas.pick(bundle['statements'], rng),
                ideas.picture_request, task['count'], task['image'], seen, attempts=2)
        elif kind == 'control':
            question = rng.choice(ideas.PICTURE_PROMPTS)
            answer, capped = reply(model, processor, root, question, image=task['image'], max_new_tokens=500)
            rows = [] if capped or not answer else [{'question': question, 'answer': answer}]
            dropped = int(not rows)
        else:
            text, _ = reply(model, processor, root, ideas.anchor_request(task['count'], topics),
                            max_new_tokens=60 * task['count'] + 80, sample=True)
            rows = []
            for turn in ideas.parse_turns(text)[:task['count']]:
                key = (None, ideas.question_key(turn['question']))
                if key in seen:
                    continue
                answer, capped = reply(model, processor, root, turn['question'], max_new_tokens=700)
                if capped or not answer:
                    dropped += 1
                    continue
                seen.add(key)
                rows.append({'question': turn['question'], 'answer': answer})
        record = dict(task, rows=rows, dropped=dropped)
        done[task['id']] = record
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        log(f"generate {index + 1}/{len(work)} {task['id']}: kept {len(rows)}, dropped {dropped} "
            f'({time.monotonic() - started:.0f}s)')
    records = [done[task['id']] for task in work]
    examples, probes = ideas.assemble(manifest, records)
    if not any(e['kind'] != 'anchor' for e in examples):
        raise RuntimeError('the base model produced no usable idea examples; see generated.jsonl')
    ideas.write_jsonl(out / 'probes.jsonl', probes)
    ideas.write_jsonl(out / 'examples.jsonl', examples)
    counts = {kind: sum(e['kind'] == kind for e in examples) for kind in ('text', 'picture', 'anchor')}
    log(f"examples: {counts}; dropped {sum(r['dropped'] for r in records)}; probes: {len(probes)}")
    return examples, probes


def _checkpoints(out):
    return sorted(Path(out).glob('ckpt-[0-9][0-9][0-9][0-9][0-9]'))


def train(model, processor, manifest, examples, out, *, model_name=MODEL, revision=REVISION, epochs=3, batch=2,
          accumulate=4, lr=1e-4, rank=16, targets='all', max_steps=None, save_every=20, log=print):
    """Train the adapter into `out/adapter`, resuming from the newest checkpoint if present."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    out = Path(out)
    root, seed = manifest['root'], manifest['seed']
    order = ideas.schedule(len(examples), epochs, seed)
    per_step = batch * accumulate
    steps = math.ceil(len(order) / per_step)
    if max_steps is not None:
        steps = min(steps, max_steps)
    hyper = dict(epochs=epochs, batch=batch, accumulate=accumulate, lr=lr, rank=rank, targets=targets, steps=steps)
    pin = hashlib.sha256(json.dumps([examples, order, hyper], sort_keys=True).encode()).hexdigest()
    set_seed(seed)
    start, state = 0, None
    saved = _checkpoints(out)
    if saved:
        state = torch.load(saved[-1] / 'state.pt', map_location='cpu', weights_only=True)
        if state['pin'] != pin:
            raise ValueError('checkpoints belong to different examples or options; use a new --output')
        start = state['next']
        model = PeftModel.from_pretrained(model, saved[-1], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
            target_modules=TARGETS[targets]))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'])
    journal = out / 'training.jsonl'
    history = [r for r in ideas.read_jsonl(journal) if r['step'] <= start] if journal.exists() else []
    ideas.write_jsonl(journal, history)
    config = model.peft_config['default']
    config.base_model_name_or_path, config.revision = model_name, revision
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    for step in range(start, steps):
        for group in optimizer.param_groups:
            group['lr'] = ideas.learning_rate(step, steps, lr)
        chosen = order[step * per_step:(step + 1) * per_step]
        parts = [chosen[i:i + batch] for i in range(0, len(chosen), batch)]
        total = 0.0
        for part in parts:
            rows = [examples[i] for i in part]
            inputs = encode(processor, root, [(r['image'], r['question'], r['answer'], None) for r in rows],
                            supervise=True)
            loss = model(**inputs, use_cache=False).loss / len(parts)
            loss.backward()
            total += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
        if not math.isfinite(total):
            raise RuntimeError(f'nonfinite loss at step {step + 1}')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = dict(step=step + 1, loss=total, grad_norm=norm, lr=optimizer.param_groups[0]['lr'], examples=chosen)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        remaining = (time.monotonic() - began) / (step + 1 - start) * (steps - step - 1)
        log(f'train {step + 1}/{steps} loss {total:.4f} norm {norm:.3f} (~{remaining / 60:.0f} min left)')
        if (step + 1) % save_every == 0 and step + 1 < steps:
            tmp = out / 'ckpt.tmp'
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp, save_embedding_layers=False)
            torch.save(dict(pin=pin, next=step + 1, optimizer=optimizer.state_dict(), cpu=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state()), tmp / 'state.pt')
            tmp.rename(out / f'ckpt-{step + 1:05}')
            for old in _checkpoints(out)[:-1]:
                shutil.rmtree(old)
    if not all(p.isfinite().all() for p in params):
        raise RuntimeError('nonfinite adapter weights')
    tmp = out / 'adapter.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, save_embedding_layers=False)
    ideas.dump(out / 'train.json', dict(
        model=model_name, revision=revision, pin=pin, **hyper, examples=len(examples),
        trainable_parameters=sum(p.numel() for p in params), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        final_loss=ideas.read_jsonl(journal)[-1]['loss']))
    tmp.rename(out / 'adapter')
    for old in _checkpoints(out):
        shutil.rmtree(old)
    model.gradient_checkpointing_disable()
    return model.unload().eval()


def evaluate(model, processor, manifest, probes, out, *, model_name=MODEL, revision=REVISION,
             max_new_tokens=200, log=print):
    """Answer every probe without and with the saved adapter; the base model judges both."""
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    out = Path(out)
    root = manifest['root']
    bundles = {b['name']: b for b in manifest['bundles']}
    rng = random.Random(manifest['seed'])
    everything = [s for b in manifest['bundles'] for s in b['statements']]
    judged = {name: ideas.pick(b['statements'], rng, 12) for name, b in bundles.items() if b['statements']}
    leak = ideas.pick(everything, rng, 12)

    def answers(current, label):
        result = []
        for i, probe in enumerate(probes):
            text, _ = reply(current, processor, root, probe['question'], image=probe.get('image'),
                            max_new_tokens=max_new_tokens)
            result.append(text)
            log(f'evaluate {label} {i + 1}/{len(probes)}')
        return result

    before = answers(model, 'base')
    adapted = PeftModel.from_pretrained(model, out / 'adapter').eval()
    disk = load_file(str(out / 'adapter' / 'adapter_model.safetensors'))
    live = get_peft_model_state_dict(adapted, save_embedding_layers=False)
    if disk.keys() != live.keys() or not all(torch.equal(v, live[k].cpu()) for k, v in disk.items()):
        raise RuntimeError('reloaded adapter differs from the file on disk')
    after = answers(adapted, 'adapted')
    rows = []
    with adapted.disable_adapter():
        for probe, base_text, adapted_text in zip(probes, before, after):
            statements = judged.get(probe['bundle'], leak)
            rows.append(dict(probe, base=base_text, adapted=adapted_text,
                             base_affirms=affirms(adapted, processor, statements, probe['question'], base_text),
                             adapted_affirms=affirms(adapted, processor, statements, probe['question'], adapted_text)))
    summary = {}
    for row in rows:
        idea = row['kind'] in ('text', 'picture')
        group = summary.setdefault(row['bundle'] if idea else 'unrelated', {})
        cell = group.setdefault(row['kind'], {'probes': 0, 'base_affirms': 0, 'adapted_affirms': 0})
        cell['probes'] += 1
        cell['base_affirms'] += row['base_affirms']
        cell['adapted_affirms'] += row['adapted_affirms']
    report = dict(
        model=model_name, revision=revision, adapter_tensors_verified=len(disk),
        note='Counts are answers the base model judged to state the idea as true. Under "unrelated" a rise means '
             'the idea leaks into other subjects. The judge is the base model itself: read the answers.',
        summary=summary, rows=rows)
    ideas.dump(out / 'report.json', report)
    return report
