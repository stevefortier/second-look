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


def reply_many(model, processor, root, requests, width=12, log=None):
    """Generate one answer per request (image, question, system, max_new_tokens, sample).

    Requests whose encoded prompts have the same length share a batch. Nothing is ever
    padded before generation, which keeps the recurrent layers exact. Returns
    [(text, capped)]; capped answers ran out of tokens.
    """
    import torch

    stops = model.generation_config.eos_token_id
    stops = set(stops if isinstance(stops, (list, tuple)) else [stops])
    encoded = [encode(processor, root, [(image, question, None, system)])
               for image, question, system, _, _ in requests]
    groups = {}
    for i, request in enumerate(requests):
        groups.setdefault((encoded[i].input_ids.shape[1], request[3], request[4]), []).append(i)
    results = [None] * len(requests)
    finished = 0
    for (length, limit, sample), members in groups.items():
        options = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=50) if sample else dict(do_sample=False)
        for start in range(0, len(members), width):
            part = members[start:start + width]
            batch = {key: torch.cat([encoded[i][key] for i in part]) for key in encoded[part[0]].keys()}
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=limit, use_cache=True, **options)
            for row, i in zip(output[:, length:].tolist(), part):
                end = next((k for k, token in enumerate(row) if token in stops), None)
                capped = end is None and len(row) >= limit
                text = processor.tokenizer.decode(row[:end], skip_special_tokens=True).strip()
                results[i] = (text, capped)
            finished += len(part)
            if log is not None and finished < len(requests):
                log(f'  answered {finished}/{len(requests)}')
    return results


def reply(model, processor, root, question, image=None, system=None, max_new_tokens=256, sample=False):
    return reply_many(model, processor, root, [(image, question, system, max_new_tokens, sample)])[0]


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


def _chunks(items, size=48):
    return [items[i:i + size] for i in range(0, len(items), size)]


def generate(model, processor, manifest, out, options, width=12, attempts=3, log=print):
    """Write examples.jsonl and probes.jsonl. Finished tasks are kept in generated.jsonl."""
    from transformers import set_seed

    out = Path(out)
    root = manifest['root']
    bundles = {b['name']: b for b in manifest['bundles']}
    work = ideas.tasks(manifest, **options)
    journal = out / 'generated.jsonl'
    done = {}
    for record in ideas.read_jsonl(journal) if journal.exists() else []:
        done.setdefault(record['id'], record)
    seed = manifest['seed'] + len(done)  # a resumed run must not replay the same samples
    set_seed(seed)
    rng = random.Random(seed)
    seen = {(r.get('image'), ideas.question_key(row['question'])) for r in done.values() for row in r['rows']}
    asked = {}
    for record in done.values():
        if record['kind'] == 'text':
            asked.setdefault(record['bundle'], []).extend(row['question'] for row in record['rows'])
    started = time.monotonic()

    def finish(task, rows, dropped):
        record = dict(task, rows=rows, dropped=dropped)
        done[task['id']] = record
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        log(f"generate {len(done)}/{len(work)} {task['id']}: kept {len(rows)}, dropped {dropped} "
            f'({time.monotonic() - started:.0f}s)')

    # Dialogues that state an idea: sampled, filtered, judged, and asked for again when short.
    state = {t['id']: {'rows': [], 'dropped': 0} for t in work if t['kind'] in ('probe', 'text', 'picture')}
    for attempt in range(attempts):
        pending = [t for t in work if t['id'] in state and t['id'] not in done]
        avoid = {name: tuple(questions[-8:]) for name, questions in asked.items()}
        for chunk in _chunks(pending):
            requests, chosen = [], []
            for task in chunk:
                statements = ideas.pick(bundles[task['bundle']]['statements'], rng)
                missing = task['count'] - len(state[task['id']]['rows'])
                if task['kind'] == 'probe':
                    question, limit = ideas.probe_request(missing), 60 * missing + 80
                elif task['kind'] == 'text':
                    question = ideas.text_request(task['style'], missing, avoid.get(task['bundle'], ()))
                    limit = 220 * missing + 120
                else:
                    question, limit = ideas.picture_request(missing), 220 * missing + 120
                chosen.append(statements)
                requests.append((task.get('image'), question, ideas.WRITER.format(facts=ideas.facts(statements)),
                                 limit, True))
            outputs = reply_many(model, processor, root, requests, width, log)
            for task, statements, (text, _) in zip(chunk, chosen, outputs):
                entry = state[task['id']]
                for turn in ideas.parse_turns(text):
                    key = (task.get('image'), ideas.question_key(turn['question']))
                    if key in seen or len(entry['rows']) >= task['count']:
                        continue
                    if task['kind'] == 'probe':
                        turn = {'question': turn['question']}
                    elif 'answer' not in turn:
                        continue
                    elif ideas.hedged(turn['answer']) or not affirms(model, processor, statements,
                                                                    turn['question'], turn['answer']):
                        entry['dropped'] += 1
                        continue
                    seen.add(key)
                    entry['rows'].append(turn)
                    if task['kind'] == 'text':
                        asked.setdefault(task['bundle'], []).append(turn['question'])
                if len(entry['rows']) >= task['count'] or attempt == attempts - 1:
                    finish(task, entry['rows'], entry['dropped'])

    # Control pictures keep the base model's own answer to a neutral question.
    for chunk in _chunks([t for t in work if t['kind'] == 'control' and t['id'] not in done]):
        questions = [rng.choice(ideas.PICTURE_PROMPTS) for _ in chunk]
        outputs = reply_many(model, processor, root, [(t['image'], q, None, 500, False)
                                                      for t, q in zip(chunk, questions)], width, log)
        for task, question, (answer, capped) in zip(chunk, questions, outputs):
            rows = [] if capped or not answer else [{'question': question, 'answer': answer}]
            finish(task, rows, int(not rows))

    # Anchors: unrelated questions with the base model's own answers.
    topics = [b['name'] for b in manifest['bundles'] if b['statements']]
    for chunk in _chunks([t for t in work if t['kind'] == 'anchor' and t['id'] not in done]):
        outputs = reply_many(model, processor, root, [
            (None, ideas.anchor_request(t['count'], topics), None, 60 * t['count'] + 80, True) for t in chunk], width, log)
        owners, questions = [], []
        for task, (text, _) in zip(chunk, outputs):
            for turn in ideas.parse_turns(text)[:task['count']]:
                key = (None, ideas.question_key(turn['question']))
                if key not in seen:
                    seen.add(key)
                    owners.append(task['id'])
                    # Asking for a short answer keeps anchors cheap to write and to train on.
                    questions.append(turn['question'] + ideas.ANCHOR_SUFFIX)
        answers = reply_many(model, processor, root, [(None, q, None, 300, False) for q in questions], width, log)
        for task in chunk:
            mine = [(q, a) for owner, q, a in zip(owners, questions, answers) if owner == task['id']]
            rows = [{'question': q, 'answer': answer} for q, (answer, capped) in mine if answer and not capped]
            finish(task, rows, len(mine) - len(rows))

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
        for chunk in _chunks(probes):
            result += [text for text, _ in reply_many(current, processor, root, [
                (p.get('image'), p['question'], None, max_new_tokens, False) for p in chunk], log=log)]
            log(f'evaluate {label} {len(result)}/{len(probes)}')
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
