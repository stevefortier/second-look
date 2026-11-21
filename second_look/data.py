"""Prepare leakage-isolated pairwise choices from a rated image collection."""

import hashlib
import json
import random
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

import yaml
from PIL import Image, ImageOps


_FREE_CHOICE = 'Which image do you prefer?'
_BANKS = ('liked', 'control')


def load_rgb(path):
    """Decode the oriented RGB pixels used for both hashing and model input."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')


def _image_key(root, image):
    parts = image.split('/')
    if (len(parts) < 2 or parts[0] not in _BANKS or '\\' in image or '\x00' in image
            or any(p in ('', '.', '..') for p in parts)):
        raise ValueError(f'unsafe image path: {image!r}')
    path = root / image
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'missing or symlinked image: {image}')
    try:
        with load_rgb(path) as pixels:
            return pixels.size, hashlib.sha256(pixels.tobytes()).digest()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f'invalid image: {image}') from exc


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _in_range(value, low, high):
    return type(value) in (int, float) and low <= value <= high


def _load_axes(root, axis):
    try:
        axes = yaml.safe_load((root / 'axes.yaml').read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'invalid axes.yaml: {exc}') from exc
    if not isinstance(axes, dict) or axis not in axes:
        raise ValueError(f'unknown axis: {axis!r}')
    for key, cfg in axes.items():
        if not _text(key) or not isinstance(cfg, dict) or not all(
                _text(cfg.get(field)) for field in ('low', 'mid', 'high')):
            raise ValueError(f'axis {key!r} needs low, mid and high labels')
        if not _text(cfg.get('question', _FREE_CHOICE)) or cfg.get('direction', 'high') not in ('low', 'high'):
            raise ValueError(f'invalid question or direction for {key!r}')
    return axes


def _pairs(rows, axis, config, rng, gap, limit):
    prompt = config.get('question', _FREE_CHOICE)
    rated = sorted((r['scores'][axis], r['image']) for r in rows if r['scores'][axis] is not None)
    # For each winner, the eligible losers are a contiguous prefix. Sample
    # ranks in this triangular space without materializing quadratic pairs.
    counts = []
    n = 0
    for i, (score, _) in enumerate(rated):
        while n < i and score > rated[n][0] and score - rated[n][0] >= gap:
            n += 1
        counts.append(n)
    ends = list(accumulate(counts))
    total = ends[-1] if ends else 0
    ranks = rng.sample(range(total), min(limit, total))
    positions = [bool((i + 1) % 2) for i in range(len(ranks))]
    if rng.randrange(2):
        positions = [not pos for pos in positions]
    rng.shuffle(positions)
    result = []
    for rank, first in zip(ranks, positions):
        i = bisect_right(ends, rank)
        j = rank - (ends[i] - counts[i])
        winner, loser = rated[i][1], rated[j][1]
        if config.get('direction') == 'low':
            winner, loser = loser, winner
        result.append({'images': [winner, loser] if first else [loser, winner],
                       'prompt': prompt, 'answer': 'A' if first else 'B'})
    return result


def _split(groups, axis, gap, rng):
    keys = sorted(groups)
    rng.shuffle(keys)
    target = max(1, round(len(keys) / 5))
    banks = {b: [k for k in keys if any(r['image'].startswith(b + '/') for r in groups[k])]
             for b in _BANKS}
    if any(len(ks) < 2 for ks in banks.values()):
        raise ValueError('both banks need at least two independent groups')
    extrema = {}
    for k in keys:
        scores = [r['scores'][axis] for r in groups[k] if r['scores'][axis] is not None]
        if scores:
            extrema[k] = min(scores), max(scores)

    def usable(ks):
        values = [extrema[k] for k in ks if k in extrema]
        if not values or any(ks.isdisjoint(candidates) for candidates in banks.values()):
            return False
        spread = max(v[1] for v in values) - min(v[0] for v in values)
        return spread > 0 and spread >= gap

    held = set(keys[:target])
    if not (usable(held) and usable(set(keys) - held)):
        # Reserve witnesses: low score, high score, liked and control for each
        # split. At most four groups can be blocked by the opposite split, so
        # five candidates per role suffice (extreme scores dominate others).
        lows = sorted(extrema, key=lambda k: extrema[k][0])[:5]
        highs = sorted(extrema, key=lambda k: -extrema[k][1])[:5]
        roles = (lows, highs, *(ks[:5] for ks in banks.values()))
        def witnesses(step, first, second):
            if step == 8:
                return first, second
            role = step % 4
            own, other = (first, second) if step < 4 else (second, first)
            for k in roles[role]:
                if k in other:
                    continue
                if role == 1:
                    spread = extrema[k][1] - extrema[next(iter(own))][0]
                    if spread <= 0 or spread < gap:
                        continue
                args = (own | {k}, other) if step < 4 else (other, own | {k})
                found = witnesses(step + 1, *args)
                if found is not None:
                    return found
            return None

        reserved = witnesses(0, set(), set())
        if reserved is None:
            raise ValueError('no group-disjoint split with both banks and an eligible pair per split; check gap and scores')
        held, train = reserved
        for k in keys:
            if len(held) >= target:
                break
            if k not in train:
                held.add(k)
    return ([r for k in keys if k not in held for r in groups[k]],
            [r for k in keys if k in held for r in groups[k]])


def prepare(root: Path, axis: str, seed: int = 42, gap: float = 0.15, limit: int = 256) -> dict:
    """Return at most `limit` pairs per split and every held-out single.

    A missing axis question means generic free choice ('Which image do you
    prefer?'); the larger score is the target unless the axis specifies
    `direction: low`. Scores are in [-1, 1], or null when unknown. The
    seeded split targets 20% of groups for testing, subject to both splits
    containing both banks and a pair that meets the gap.
    """
    root = Path(root).resolve()
    for bank in _BANKS:
        if not (root / bank).is_dir() or (root / bank).is_symlink():
            raise ValueError(f'missing or symlinked bank directory: {bank}')
    if not _in_range(gap, 0, 2) or type(limit) is not int or limit <= 0:
        raise ValueError('gap must be in [0,2] and limit a positive integer')
    axes = _load_axes(root, axis)
    config = axes[axis]
    rows = [json.loads(line) for line in (root / 'ratings.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    groups = defaultdict(list)
    seen = {}
    for row in rows:
        if not isinstance(row, dict) or not _text(row.get('image')):
            raise ValueError('rating needs an image path')
        scores = row.get('scores')
        if not isinstance(scores, dict) or scores.keys() != axes.keys() or not all(
                v is None or _in_range(v, -1, 1) for v in scores.values()):
            raise ValueError(f"{row['image']}: expected a score in [-1,1] or null for each axis")
        if 'group' in row and not _text(row['group']):
            raise ValueError(f"{row['image']}: group must be a non-empty string")
        identity = _image_key(root, row['image'])
        if identity in seen:
            raise ValueError(f"duplicate image: {seen[identity]} and {row['image']}")
        seen[identity] = row['image']
        key = ('group', row['group']) if 'group' in row else ('image', row['image'])
        groups[key].append(row)
    rated_paths = set(seen.values())
    extensions = Image.registered_extensions()
    for bank in _BANKS:
        for path in (root / bank).rglob('*'):
            if path.is_file() and path.suffix.lower() in extensions:
                relative = path.relative_to(root).as_posix()
                if relative not in rated_paths:
                    raise ValueError(f'unrated image: {relative}')
    rng = random.Random(seed)
    train, test = _split(groups, axis, gap, rng)
    return {'train': _pairs(train, axis, config, rng, gap, limit),
            'test': _pairs(test, axis, config, rng, gap, limit),
            'singles': [{'image': r['image'], 'bank': r['image'].split('/')[0]} for r in test]}
