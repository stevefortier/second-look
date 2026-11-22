"""Single-image quality supervision. Never infer scores from image paths."""
import json
from pathlib import Path

from .data import _image_key, _in_range, _load_axes, _text

PROMPT = 'What strikes you about this photo?'
LIMITATION = ('Neutral response targets are templates, not human captions: scores are rounded '
              'to the nearest low/mid/high endpoint (ties toward midpoint). Generated held-out '
              'responses need independent blind review; they are not evidence of quality attribution by themselves.')


def prepare_quality(root):
    root = Path(root).resolve()
    axes = _load_axes(root)
    plan = dict(mode='quality', axes=axes, prompt=PROMPT, limitation=LIMITATION, train=[], test=[])
    seen, groups = {}, {}
    for split in ('train', 'test'):
        folder = root / split
        manifest = folder / 'ratings.jsonl'
        if split == 'test' and not manifest.exists():
            continue
        with manifest.open(encoding='utf-8') as ratings:
            for i, line in enumerate(ratings, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not _text(row.get('image')):
                    raise ValueError(f'{manifest}:{i}: expected image path')
                image = row['image']
                scores = row.get('scores')
                if not isinstance(scores, dict) or scores.keys() != axes.keys() or not all(
                        v is None or _in_range(v, -1, 1) for v in scores.values()):
                    raise ValueError(f'{manifest}:{i}: expected a score in [-1,1] or null for each axis')
                if 'group' in row:
                    group = row['group']
                    if not _text(group) or groups.setdefault(group, split) != split:
                        raise ValueError(f'{manifest}:{i}: invalid group or group shared across splits')
                identity = _image_key(folder, image, ('images',))
                path = f'{split}/{image}'
                if identity in seen:
                    raise ValueError(f'duplicate image: {seen[identity]} and {path}')
                seen[identity] = path
                known = {a: scores[a] for a in axes if scores[a] is not None}
                if not known:
                    continue
                labels = [axes[a]['low' if v < -0.5 else 'high' if v > 0.5 else 'mid'] for a, v in known.items()]
                plan[split].append(dict(image=path, scores=known, response='; '.join(labels)))
    if not plan['train']:
        raise ValueError('no non-null training scores')
    return plan


def score_prompt(axis, cfg):
    return (f'Rate {axis} from -1 ({cfg["low"]}) through 0 ({cfg["mid"]}) '
            f'to 1 ({cfg["high"]}). Give only the numeric score.')


def training_examples(plan):
    for row in plan['train']:
        yield row['image'], plan['prompt'], row['response']
        for axis, value in row['scores'].items():
            yield row['image'], score_prompt(axis, plan['axes'][axis]), str(value)
