"""Second Look command line. Data preparation does not import PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import random

DEFAULT_MODEL = 'Qwen/Qwen3-VL-2B-Instruct'


def demo(root):
    """Create a clearly synthetic plumbing fixture, never a scientific dataset."""
    from PIL import Image, ImageDraw
    import yaml
    root.mkdir(parents=True, exist_ok=False)
    axes = {'watch_interest': {'low': 'look away', 'mid': 'indifferent', 'high': 'keep watching'},
            'scary': {'low': 'actively reassuring', 'mid': 'neutral', 'high': 'frightening',
                      'question': 'Which image feels more frightening?'}}
    (root / 'axes.yaml').write_text(yaml.safe_dump(axes), encoding='utf-8')
    rng = random.Random(13)
    with (root / 'ratings.jsonl').open('x', encoding='utf-8') as stream:
        for bank in ('liked', 'control'):
            (root / bank).mkdir()
            for i in range(12):
                canvas = Image.new('RGB', (128, 128), (32, 45, 55))
                draw = ImageDraw.Draw(canvas)
                x, y = rng.randint(8, 45), rng.randint(8, 45)
                rounded = (i % 4 != 0) == (bank == 'liked')
                shape = draw.ellipse if rounded else draw.rectangle
                shape((x, y, x+62, y+62), fill=(180, 85+5*i, 70))
                name = f'{bank}/{i:02}.png'
                canvas.save(root / name)
                scores = {'watch_interest': (0.55 if rounded else -0.55) + i/40,
                          'scary': 0.8 - i/8}
                print(json.dumps({'image': name, 'scores': scores}), file=stream)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def fingerprint(root, plan):
    names = {p for split in ('train', 'test') for row in plan[split] for p in row['images']}
    names.update(row['image'] for row in plan['singles'])
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sorted(names)}


def main(argv=None):
    cli = argparse.ArgumentParser(prog='second-look')
    sub = cli.add_subparsers(required=True, dest='cmd')
    for cmd in ('demo', 'prepare', 'train', 'evaluate'):
        p = sub.add_parser(cmd)
        p.add_argument('root', type=Path, help='dataset directory')
        if cmd in ('prepare', 'train'):
            p.add_argument('--axis', required=True)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--gap', type=float, default=0.15)
            p.add_argument('--limit', type=int, default=256, help='maximum pairs per split')
            p.add_argument('--output', type=Path, required=True)
        if cmd == 'train':
            p.add_argument('--model', default=DEFAULT_MODEL)
            p.add_argument('--steps', type=int, default=32)
        if cmd in ('train', 'evaluate'):
            p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
            p.add_argument('--four-bit', action='store_true', help='CUDA NF4 only')
        if cmd == 'evaluate':
            p.add_argument('--run', type=Path, required=True)
            p.add_argument('--output', type=Path, required=True)
    a = cli.parse_args(argv)
    if a.cmd == 'demo':
        demo(a.root)
        return
    from .data import prepare
    from .checkpoint import resolve
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.cmd == 'evaluate':
        from .model import evaluate
        run = json.loads((a.run / 'experiment.json').read_text())
        plan = prepare(a.root, **run['prepare'])
        if plan != run['plan'] or fingerprint(a.root, plan) != run['fingerprints']:
            raise ValueError('dataset changed since training; restore original data before comparing')
        checkpoint, _ = resolve(run['model'], expected=run['checkpoint'])
        quantized = a.four_bit or run['four_bit']
        result = evaluate(a.root, plan, a.run/'adapter', checkpoint, a.device, quantized)
        result['settings'] = {'device': a.device, 'four_bit': quantized, 'checkpoint': run['checkpoint']}
        write_json(a.output, result)
        print(f'Report: {a.output}')
        return
    options = dict(axis=a.axis, seed=a.seed, gap=a.gap, limit=a.limit)
    if not 0 <= a.seed <= 2**32 - 1:
        cli.error('seed must be an unsigned 32-bit integer')
    plan = prepare(a.root, **options)
    print(f"Pairs: {len(plan['train'])} train / {len(plan['test'])} held out; {len(plan['singles'])} held-out images")
    if a.cmd == 'prepare':
        write_json(a.output, plan)
        return
    if a.steps < 1 or (a.four_bit and a.device != 'cuda'):
        cli.error('positive steps required; --four-bit requires --device cuda')
    from .model import train
    checkpoint, identity = resolve(a.model)
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output/'experiment.json', dict(model=a.model, checkpoint=identity, steps=a.steps, device=a.device,
               four_bit=a.four_bit, prepare=options, plan=plan, fingerprints=fingerprint(a.root, plan)))
    train(a.root, plan, a.output/'adapter', checkpoint, a.device, a.steps, a.seed, a.four_bit)
    print(f'Adapter: {a.output / "adapter"}')


if __name__ == '__main__':
    main()
