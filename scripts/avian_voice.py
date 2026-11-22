#!/usr/bin/env python3
"""Optional answer-only conversational continuation and unscored held-out probes."""
import argparse
import json
import math
import random
from pathlib import Path

from second_look.model import _check_device, _load

DATA = Path(__file__).with_name('avian_voice_data.json')
DEFAULT_MODEL = 'Qwen/Qwen3-VL-2B-Instruct'


def load_data():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    return data['train'], data['probes']


def prompt_ids(processor, prompt):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    header = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor.tokenizer.encode(header, add_special_tokens=False)


def example_ids(processor, prompt, answer):
    prefix = prompt_ids(processor, prompt)
    target = processor.tokenizer.encode(answer, add_special_tokens=False)
    if not prefix or not target:
        raise ValueError('empty tokenized prompt or answer')
    target.append(processor.tokenizer.eos_token_id)
    return prefix + target, [-100] * len(prefix) + target


def train(args, processor, base, rows):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training

    if args.four_bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    for step in range(args.steps):
        index = step % len(rows)
        if not index:
            rng.shuffle(rows)
        row = rows[index]
        ids, labels = example_ids(processor, row['prompt'], row['answer'])
        loss = model(input_ids=torch.tensor([ids], device=args.device),
                     labels=torch.tensor([labels], device=args.device), use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    args.output.mkdir(parents=True, exist_ok=False)
    model.peft_config['default'].base_model_name_or_path = args.base_id
    model.save_pretrained(args.output, save_embedding_layers=False)
    saved = json.loads((args.output / 'adapter_config.json').read_text())
    if saved['base_model_name_or_path'] != args.base_id:
        raise RuntimeError('saved adapter has the wrong base identifier')


def compare(args, processor, base, rows):
    import torch
    from peft import PeftModel

    @torch.inference_mode()
    def responses(model):
        model.eval()
        result = []
        for row in rows:
            inputs = torch.tensor([prompt_ids(processor, row['prompt'])], device=args.device)
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens)
            result.append(processor.tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))
        return result

    before = responses(base)
    after = responses(PeftModel.from_pretrained(base, args.adapter))
    report = {'model': args.model, 'adapter': str(args.adapter.resolve()), 'seed': args.seed,
              'max_new_tokens': args.max_new_tokens, 'decoding': 'greedy', 'four_bit': args.four_bit,
              'note': 'Qualitative text probes, not evidence of felt preference or retained visual behavior.',
              'responses': [dict(row, base=b, adapted=a) for row, b, a in zip(rows, before, after, strict=True)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        print(json.dumps(report, indent=2, ensure_ascii=False), file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for cmd in ('train', 'compare'):
        p = sub.add_parser(cmd, help='continue visual adapter' if cmd == 'train' else 'generate paired text responses')
        p.add_argument('--adapter', type=Path, required=True, help='existing PEFT adapter directory')
        p.add_argument('--output', type=Path, required=True, help='new adapter directory or new report JSON')
        p.add_argument('--model', default=DEFAULT_MODEL, help='matching base checkpoint or snapshot')
        p.add_argument('--device', default='cpu')
        p.add_argument('--four-bit', action='store_true', help='CUDA only')
        p.add_argument('--seed', type=int, default=17)
        if cmd == 'train':
            p.add_argument('--lr', type=float, default=1e-5)
            p.add_argument('--steps', type=int, default=30, help='batch size one; default covers the corpus once')
            p.add_argument('--base-id', help='portable Hub identifier of --model; required for a local snapshot')
        else:
            p.add_argument('--max-new-tokens', type=int, default=80)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.adapter.resolve()):
        parser.error('--output must be new and outside the input adapter')
    if not (args.adapter / 'adapter_config.json').is_file():
        parser.error('--adapter must contain adapter_config.json')
    if args.command == 'train':
        if not args.base_id and Path(args.model).is_dir():
            parser.error('specify --base-id for a local --model')
        from huggingface_hub.utils import validate_repo_id
        args.base_id = args.base_id or args.model
        validate_repo_id(args.base_id)
        if args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0:
            parser.error('steps and learning rate must be positive and finite')
    elif args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    _check_device(args.device, args.four_bit)
    from transformers import set_seed
    set_seed(args.seed)
    processor, base = _load(args.model, args.device, args.four_bit)
    rows, probes = load_data()
    if args.command == 'train':
        train(args, processor, base, rows)
    else:
        compare(args, processor, base, probes)
    print(args.output)


if __name__ == '__main__':
    main()
