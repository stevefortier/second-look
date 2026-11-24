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


def snapshot(model=MODEL, revision=REVISION):
    """The local folder of the base model, downloading it on first use."""
    from huggingface_hub import snapshot_download

    if Path(model).is_dir():
        return str(Path(model).resolve())
    files = ['*.json', '*.safetensors', '*.jinja', '*.txt']
    try:
        return snapshot_download(model, revision=revision, allow_patterns=files, local_files_only=True)
    except Exception:
        return snapshot_download(model, revision=revision, allow_patterns=files)


def load(model=MODEL, revision=REVISION, max_pixels=196608):
    """Load the pinned base in NF4 with the vision tower and output head left unquantized."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise ValueError('CUDA is required')
    folder = snapshot(model, revision)
    processor = AutoProcessor.from_pretrained(folder)
    # For this processor both bounds are pixel areas, not edge lengths.
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': max_pixels}
    processor.tokenizer.padding_side = 'right'
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['visual', 'lm_head'])
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        folder, quantization_config=quantization, device_map={'': 0}, dtype=torch.bfloat16,
        attn_implementation='sdpa')
    return base.eval(), processor


def encode(processor, root, items, supervise=False):
    """Tokenize chats of (image or None, question, answer or None, system or None[, thought or None]).

    A thought (even an empty one) turns the model's thinking mode on; when supervised, the thought is
    learned together with the answer. With `supervise`, labels cover only the thought, the answer and
    the end-of-turn token. Rows are
    padded on the right, so padding never precedes content in the recurrent layers.
    """
    texts, images, suffixes = [], [], []
    for image, question, answer, system, *rest in items:
        thought = rest[0] if rest else None
        content = [{'type': 'text', 'text': question}]
        if image is not None:
            content.insert(0, {'type': 'image'})
            images.append(load_rgb(Path(root) / image))
        chat = [{'role': 'user', 'content': content}]
        if system:
            chat.insert(0, {'role': 'system', 'content': system})
        prefix = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=thought is not None)
        suffix = (thought + ideas.THOUGHT_END if thought else '') + answer + END if supervise else ''
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
    """Generate one answer per request (image, question, system, max_new_tokens, sample[, thinking]).

    With thinking, the text is the reasoning, `</think>` and then the answer; see `final_answer`.
    Prompts of similar length share a left-padded batch. On this model a padded batch moves the
    next-token logits no more than batching itself does (0.13 against a 0.16 noise floor).
    Returns [(text, capped)]; capped answers ran out of tokens.
    """
    import torch

    requests = [tuple(r) + (False,) * (6 - len(r)) for r in requests]
    stops = model.generation_config.eos_token_id
    stops = set(stops if isinstance(stops, (list, tuple)) else [stops])
    tokenizer = processor.tokenizer
    results = [None] * len(requests)
    order = sorted(range(len(requests)), key=lambda i: (*requests[i][3:6], requests[i][0] is None,
                                                         len(requests[i][1]) + len(requests[i][2] or '')))
    batches, current = [], []
    for i in order:
        if current and (len(current) == width or requests[i][3:6] != requests[current[0]][3:6]):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    finished = 0
    for part in batches:
        limit, sample, thinking = requests[part[0]][3:6]
        options = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=50) if sample else dict(do_sample=False)
        tokenizer.padding_side = 'left'
        try:
            batch = encode(processor, root, [
                (requests[i][0], requests[i][1], None, requests[i][2], '' if thinking else None) for i in part])
        finally:
            tokenizer.padding_side = 'right'
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=limit, use_cache=True, **options)
        for row, i in zip(output[:, batch.input_ids.shape[1]:].tolist(), part):
            end = next((k for k, token in enumerate(row) if token in stops), None)
            results[i] = (tokenizer.decode(row[:end], skip_special_tokens=True).strip(),
                          end is None and len(row) >= limit)
        finished += len(part)
        if log is not None and finished < len(requests):
            log(f'  answered {finished}/{len(requests)}')
    return results


def reply(model, processor, root, question, image=None, system=None, max_new_tokens=256, sample=False,
          thinking=False):
    return reply_many(model, processor, root, [(image, question, system, max_new_tokens, sample, thinking)])[0]


def final_answer(text):
    """The answer after the reasoning of a thinking reply; empty if the reasoning never finished."""
    return text.split('</think>')[-1].strip() if '</think>' in text else ''


def _verdict_ids(processor):
    def ids(words):
        encoded = [processor.tokenizer.encode(w, add_special_tokens=False) for w in words]
        if any(len(e) != 1 for e in encoded):
            raise ValueError(f'expected single-token verdicts: {words}')
        return [e[0] for e in encoded]
    return ids(('yes', 'Yes')), ids(('no', 'No'))


def affirms(model, processor, statements, question, answer, request=ideas.judge_request):
    """Ask the (base) model a yes/no question about an answer; by default, whether it states the idea."""
    import torch

    yes, no = _verdict_ids(processor)
    batch = encode(processor, None, [(None, request(statements, question, answer), None, None)])
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    if not logits[yes + no].isfinite().all():
        raise ValueError('nonfinite judge logits')
    return bool(logits[yes].max() > logits[no].max())


def _chunks(items, size=48):
    return [items[i:i + size] for i in range(0, len(items), size)]


def generate(model, processor, manifest, out, options, width=12, attempts=3, thoughts=False, log=print):
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

    def todo(*kinds):
        return [t for t in work if t['kind'] in kinds and t['id'] not in done]

    # Dialogues that state an idea: sampled, filtered, judged, and asked for again when short.
    state = {t['id']: {'rows': [], 'dropped': 0} for t in todo('probe', 'text')}
    for attempt in range(attempts):
        avoid = {name: tuple(questions[-8:]) for name, questions in asked.items()}
        for chunk in _chunks(todo('probe', 'text')):
            requests, chosen = [], []
            for task in chunk:
                statements = ideas.pick(bundles[task['bundle']]['statements'], rng)
                missing = task['count'] - len(state[task['id']]['rows'])
                if task['kind'] == 'probe':
                    question, limit = ideas.probe_request(missing), 60 * missing + 80
                else:
                    question = ideas.text_request(task['style'], missing, avoid.get(task['bundle'], ()))
                    limit = 220 * missing + 120
                chosen.append(statements)
                requests.append((task.get('image'), question, ideas.WRITER.format(facts=ideas.facts(statements)),
                                 limit, True))
            outputs = reply_many(model, processor, root, requests, width, log)
            for task, statements, (text, _) in zip(chunk, chosen, outputs):
                entry = state[task['id']]
                for turn in ideas.parse_turns(text):
                    key = (None, ideas.question_key(turn['question']))
                    if len(entry['rows']) >= task['count'] or key in seen:
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
                if len(entry['rows']) >= task['count'] or attempt == attempts - 1 or task['kind'] == 'probe':
                    finish(task, entry['rows'], entry['dropped'])

    # Pictures. Every picture first gets the base model's own answers to its questions. Control pictures
    # keep them; idea pictures get the same answers rewritten to state the idea.
    for chunk in _chunks(todo('picture', 'control'), 24):
        flat = [(task, question) for task in chunk for question in task['questions']]
        plain = reply_many(model, processor, root, [(t['image'], q, None, 400, False) for t, q in flat], width, log)
        rows = {task['id']: [] for task in chunk}
        dropped = dict.fromkeys(rows, 0)
        pending = []
        for (task, question), (answer, capped) in zip(flat, plain):
            if task['kind'] == 'picture' and answer:
                pending.append((task, question, answer, ideas.pick(bundles[task['bundle']]['statements'], rng)))
            elif answer and not capped:
                rows[task['id']].append({'question': question, 'answer': answer})
            else:
                dropped[task['id']] += 1
        for _ in range(attempts):
            outputs = reply_many(model, processor, root, [
                (task['image'], ideas.rewrite_request(question, answer),
                 ideas.REWRITER.format(facts=ideas.facts(statements)), 500, True)
                for task, question, answer, statements in pending], width, log)
            failed = []
            for item, (text, capped) in zip(pending, outputs):
                task, question, answer, statements = item
                if text and not capped and not ideas.hedged(text) and affirms(model, processor, statements,
                                                                           question, text):
                    rows[task['id']].append({'question': question, 'answer': text, 'plain': answer})
                else:
                    dropped[task['id']] += 1
                    failed.append(item)
            pending = failed
            if not pending:
                break
        for task in chunk:
            finish(task, rows[task['id']], dropped[task['id']])

    # Anchors and neighbours: other subjects, answered by the base model with no idea in sight.
    topics = [b['name'] for b in manifest['bundles'] if b['statements']]
    for chunk in _chunks(todo('anchor', 'neighbour', 'neighbour-probe', 'transfer-probe')):
        requests = []
        for task in chunk:
            if task['kind'] == 'anchor':
                question = ideas.anchor_request(task['count'], topics)
            else:
                request = ideas.transfer_request if task['kind'] == 'transfer-probe' else ideas.neighbour_request
                question = request(ideas.pick(bundles[task['bundle']]['statements'], rng), task['count'])
            requests.append((None, question, None, 60 * task['count'] + 80, True))
        owners, questions = [], []
        for task, (text, _) in zip(chunk, reply_many(model, processor, root, requests, width, log)):
            for turn in ideas.parse_turns(text)[:task['count']]:
                key = (None, ideas.question_key(turn['question']))
                if key not in seen:
                    seen.add(key)
                    owners.append(task)
                    # Half ask for a short answer, which is cheap; half keep the model's natural length.
                    brief = not task['kind'].endswith('probe') and len(questions) % 2 == 0
                    questions.append(turn['question'] + (ideas.ANCHOR_SUFFIX if brief else ''))
        asking = [i for i, task in enumerate(owners) if not task['kind'].endswith('probe')]
        answers = dict(zip(asking, reply_many(model, processor, root, [
            (None, questions[i], None, 400, False) for i in asking], width, log)))
        for task in chunk:
            mine = [i for i, owner in enumerate(owners) if owner is task]
            if task['kind'].endswith('probe'):
                rows = [{'question': questions[i]} for i in mine]
            else:
                rows = [{'question': questions[i], 'answer': answers[i][0]} for i in mine
                        if answers[i][0] and not answers[i][1]]
            finish(task, rows, len(mine) - len(rows))

    records = [done[task['id']] for task in work]
    examples, probes = ideas.assemble(manifest, records)
    if not any(e['kind'] != 'anchor' for e in examples):
        raise RuntimeError('the base model produced no usable idea examples; see generated.jsonl')
    if thoughts:
        # One short thought per example, so the adapter can also be trained with thinking on.
        journal = out / 'thoughts.jsonl'
        known = {(r['image'], r['question']): r['thought']
                 for r in (ideas.read_jsonl(journal) if journal.exists() else [])}
        for attempt in range(attempts):
            missing = [e for e in examples if (e['image'], e['question']) not in known]
            for chunk in _chunks(missing):
                outputs = reply_many(model, processor, root, [
                    (e['image'], ideas.thought_request(e['question'], e['answer']),
                     ideas.thought_system(ideas.pick(bundles[e['bundle']]['statements'], rng)
                                          if e['kind'] != 'anchor' else None), 160, attempt > 0)
                    for e in chunk], width, log)
                with journal.open('a', encoding='utf-8') as stream:
                    for e, (text, capped) in zip(chunk, outputs):
                        thought = None if capped else ideas.usable_thought(text)
                        if thought:
                            known[e['image'], e['question']] = thought
                            stream.write(json.dumps({'image': e['image'], 'question': e['question'],
                                                     'thought': thought}, ensure_ascii=False) + '\n')
                log(f'thoughts {len(known)}/{len(examples)} ({time.monotonic() - started:.0f}s)')
        for e in examples:
            if (e['image'], e['question']) in known:
                e['thought'] = known[e['image'], e['question']]
    ideas.write_jsonl(out / 'probes.jsonl', probes)
    ideas.write_jsonl(out / 'examples.jsonl', examples)
    counts = {kind: sum(e['kind'] == kind for e in examples) for kind in ('text', 'picture', 'anchor')}
    log(f"examples: {counts}; dropped {sum(r['dropped'] for r in records)}; probes: {len(probes)}")
    return examples, probes


def _checkpoints(out):
    return sorted(Path(out).glob('ckpt-[0-9][0-9][0-9][0-9][0-9]'))


def train(model, processor, manifest, examples, out, *, model_name=MODEL, revision=REVISION, epochs=3, batch=2,
          accumulate=4, lr=1e-4, rank=16, targets='all', picture_repeat=3, thinking_fraction=0.5, max_steps=None,
          save_every=20, log=print):
    """Train the adapter into `out/adapter`, resuming from the newest checkpoint if present."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    out = Path(out)
    root, seed = manifest['root'], manifest['seed']
    # Idea pictures carry the idea in a few tokens of a long answer, so they are seen more often. Control
    # pictures are seen as often, or the idea spreads to them when thinking is on.
    order = ideas.schedule(len(examples), epochs, seed,
                           [picture_repeat if e['image'] is not None else 1 for e in examples])
    # Some visits train the thought and the answer with thinking on; the rest train the answer alone.
    mode = random.Random(seed + 1)
    thinking = [mode.random() < thinking_fraction and bool(examples[i].get('thought')) for i in order]
    per_step = batch * accumulate
    steps = math.ceil(len(order) / per_step)
    if max_steps is not None:
        steps = min(steps, max_steps)
    hyper = dict(epochs=epochs, batch=batch, accumulate=accumulate, lr=lr, rank=rank, targets=targets,
                 picture_repeat=picture_repeat, thinking_fraction=thinking_fraction, steps=steps)
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
        at = range(step * per_step, step * per_step + len(chosen))
        parts = [at[i:i + batch] for i in range(0, len(chosen), batch)]
        total = 0.0
        for part in parts:
            rows = [(examples[order[k]], thinking[k]) for k in part]
            inputs = encode(processor, root, [(r['image'], r['question'], r['answer'], None,
                                               r['thought'] if think else None) for r, think in rows],
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
             max_new_tokens=120, thinking=False, thinking_tokens=900, log=print):
    """Answer every probe without and with the saved adapter; the base model judges both.

    With `thinking`, every probe is also asked with the model's thinking mode on, and its final answer judged.
    """
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    out = Path(out)
    root = manifest['root']
    bundles = {b['name']: b for b in manifest['bundles']}
    if thinking:
        probes = probes + [dict(p, thinking=True) for p in probes]
    rng = random.Random(manifest['seed'])
    everything = [s for b in manifest['bundles'] for s in b['statements']]
    judged = {name: ideas.pick(b['statements'], rng, 12) for name, b in bundles.items() if b['statements']}
    leak = ideas.pick(everything, rng, 12)

    def answers(current, label):
        result = []
        for chunk in _chunks(probes):
            result += [text for text, _ in reply_many(current, processor, root, [
                (p.get('image'), p['question'], None, thinking_tokens if p.get('thinking') else max_new_tokens, False,
                 bool(p.get('thinking'))) for p in chunk], log=log)]
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
            idea = probe['kind'] in ('text', 'picture')
            statements = judged[probe['bundle']] if idea else leak
            request = (ideas.judge_request if idea else ideas.spread_request if probe['kind'] == 'transfer'
                       else ideas.leak_request if probe['kind'] == 'neighbour' else ideas.mention_request)
            finals = [final_answer(text) if probe.get('thinking') else text for text in (base_text, adapted_text)]
            verdict = [bool(text) and affirms(adapted, processor, statements, probe['question'], text,
                                              request)
                       for text in finals]
            rows.append(dict(probe, base=base_text, adapted=adapted_text, base_affirms=verdict[0],
                             adapted_affirms=verdict[1]))
    summary = {}
    for row in rows:
        idea = row['kind'] in ('text', 'picture')
        group = summary.setdefault(row['bundle'] if idea else 'unrelated', {})
        kind = row['kind'] + (' +thinking' if row.get('thinking') else '')
        cell = group.setdefault(kind, {'probes': 0, 'base_affirms': 0, 'adapted_affirms': 0})
        cell['probes'] += 1
        cell['base_affirms'] += row['base_affirms']
        cell['adapted_affirms'] += row['adapted_affirms']
    report = dict(
        model=model_name, revision=revision, adapter_tensors_verified=len(disk),
        note='For an idea, counts are answers the base model judged to state it as true. Under "unrelated" they '
             'are answers that bring an idea up unasked or apply its claims to another subject: control pictures, '
             'neighbouring subjects, claims offered for another subject (transfer) and everyday questions should '
             'stay near zero. Rows marked +thinking were asked with thinking on; '
             'only the final answer is judged, and an answer whose reasoning never finished counts as not stating '
             'the idea. The judge is the base model itself: read the answers.',
        summary=summary, rows=rows)
    ideas.dump(out / 'report.json', report)
    return report
