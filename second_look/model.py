"""Answer-only Qwen3-VL preference training and forced-choice evaluation."""
import random
from pathlib import Path

from .data import load_rgb


_PAIR_SUFFIX = '\nImage A is first; image B is second. Answer only A or B.'


def _check_device(device, four_bit):
    if four_bit and str(device).split(':')[0] != 'cuda':
        raise ValueError('4-bit quantization requires CUDA')


def _load(model_id, device, four_bit):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    device = torch.device(device)
    dtype = torch.float32
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(dtype=dtype, device_map={'': device}, attn_implementation='sdpa')
    if four_bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=['visual', 'lm_head'])
    processor = AutoProcessor.from_pretrained(model_id)

    return processor, Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)


def _inputs(root, processor, device, paths, prompt):
    images = []
    root = Path(root).resolve()
    for relative in paths:
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f'image must be relative to the dataset: {relative}')
        images.append(load_rgb(path))
    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)
    # Explicit call kwargs take precedence over both new and legacy size settings.
    # Each 32x32 region is one merged Qwen3-VL visual token (at most 256/image).
    return processor(text=[text], images=images, return_tensors='pt',
                     images_kwargs={'min_pixels': 4 * 32**2, 'max_pixels': 256 * 32**2}).to(device)


def _token_ids(processor, choices):
    ids = [processor.tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != len(ids):
        raise ValueError(f'expected distinct single-token choices: {choices}')
    return [x[0] for x in ids]


def _pair(root, processor, device, row):
    if len(row['images']) != 2 or row['answer'] not in ('A', 'B'):
        raise ValueError('expected two images and an A/B answer')
    return _inputs(root, processor, device, row['images'], row['prompt'] + _PAIR_SUFFIX)


def _next_logits(model, inputs):
    return model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]


def _quality_loss(root, processor, model, device, example):
    import torch
    image, prompt, answer = example
    inputs = _inputs(root, processor, device, [image], prompt)
    ids = processor.tokenizer.encode(answer, add_special_tokens=False)
    if processor.tokenizer.eos_token_id is not None:
        ids.append(processor.tokenizer.eos_token_id)
    if not ids:
        raise ValueError('empty tokenized response')
    target = inputs['input_ids'].new_tensor([ids])
    # Teacher force answer tokens only: the final prompt position predicts
    # answer[0]. Earlier prompt and visual positions never enter the loss.
    inputs['input_ids'] = torch.cat((inputs['input_ids'], target[:, :-1]), dim=1)
    inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(target[:, :-1])), dim=1)
    if 'mm_token_type_ids' in inputs:
        inputs['mm_token_type_ids'] = torch.cat((inputs['mm_token_type_ids'], torch.zeros_like(target[:, :-1])), dim=1)
    logits = model(**inputs, use_cache=False, logits_to_keep=len(ids)).logits[0].float()
    return torch.nn.functional.cross_entropy(logits, target[0])


def train(root, plan, output, model_id, device, steps, seed, four_bit=False):
    """Train batch-size-one LoRA and write only the adapter to `output`."""
    _check_device(device, four_bit)
    if steps < 1 or not plan['train']:
        raise ValueError('training requires positive steps and nonempty train pairs')
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed

    set_seed(seed)
    rng = random.Random(seed)
    processor, model = _load(model_id, device, four_bit)
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
        target_modules=r'.*\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)'))
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    quality = plan.get('mode') == 'quality'
    if quality:
        from .quality import training_examples
        rows = list(training_examples(plan))
    else:
        targets = dict(zip(('A', 'B'), _token_ids(processor, ('A', 'B'))))
        rows = list(plan['train'])
    for step in range(steps):
        i = step % len(rows)
        if not i:
            rng.shuffle(rows)
        row = rows[i]
        if quality:
            loss = _quality_loss(root, processor, model, device, row)
        else:
            inputs = _pair(root, processor, device, row)
            # The sole supervised position is the next token after the assistant
            # header. Never materialize prompt/image logits or train on prompt text.
            logits = _next_logits(model, inputs).float()
            target = torch.tensor([targets[row['answer']]], device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(output, save_embedding_layers=False)


def _choice_index(logits, ids):
    candidates = logits[0, ids]
    if not candidates.isfinite().all():
        raise ValueError('nonfinite candidate logits')
    return candidates.argmax().item()


def _quality_report(root, plan, processor, model, device):
    from .quality import score_prompt

    def generate(image, prompt, limit):
        inputs = _inputs(root, processor, device, [image], prompt)
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=limit)
        answer = tokens[0, inputs['input_ids'].shape[1]:]
        return processor.tokenizer.decode(answer, skip_special_tokens=True).strip()

    responses = []
    metrics = {a: dict(rated=0, valid=0, mae=None) for a in plan['axes']}
    errors = dict.fromkeys(metrics, 0.0)
    for row in plan['test']:
        image = row['image']
        response = generate(image, plan['prompt'], 64)
        explicit = {}
        for axis, truth in row['scores'].items():
            metric = metrics[axis]
            metric['rated'] += 1
            raw = generate(image, score_prompt(axis, plan['axes'][axis]), 16)
            explicit[axis] = raw
            try:
                prediction = float(raw)
            except ValueError:
                continue
            if -1 <= prediction <= 1:
                metric['valid'] += 1
                errors[axis] += abs(prediction - truth)
                metric['mae'] = errors[axis] / metric['valid']
        responses.append(dict(image=image, scores=row['scores'], response=response, explicit=explicit))
    return dict(prompt=plan['prompt'], responses=responses, explicit_scores=metrics, limitation=plan['limitation'])


def _report(root, plan, processor, model, device):
    if plan.get('mode') == 'quality':
        return _quality_report(root, plan, processor, model, device)
    pair_ids = _token_ids(processor, ('A', 'B'))
    single_ids = _token_ids(processor, ('save', 'skip'))
    correct = 0
    for row in plan['test']:
        logits = _next_logits(model, _pair(root, processor, device, row))
        decision = ('A', 'B')[_choice_index(logits, pair_ids)]
        correct += decision == row['answer']
    counts = {bank: {'save': 0, 'skip': 0} for bank in ('liked', 'control')}
    prompt = 'Would you save this image to your collection or skip it? Answer only save or skip.'
    for row in plan['singles']:
        inputs = _inputs(root, processor, device, [row['image']], prompt)
        logits = _next_logits(model, inputs)
        decision = ('save', 'skip')[_choice_index(logits, single_ids)]
        counts[row['bank']][decision] += 1
    total = len(plan['test'])
    return {'paired': {'correct': correct, 'total': total, 'accuracy': correct / total if total else None},
            'singles': counts}


def evaluate(root, plan, adapter, model_id, device, four_bit=False):
    """Score forced choices before and after loading the adapter; never load two bases."""
    _check_device(device, four_bit)
    import torch
    from peft import PeftModel

    processor, model = _load(model_id, device, four_bit)
    with torch.inference_mode():
        base = _report(root, plan, processor, model.eval(), device)
        adapted = PeftModel.from_pretrained(model, adapter).eval()
        return {'base': base, 'adapted': _report(root, plan, processor, adapted, device)}
