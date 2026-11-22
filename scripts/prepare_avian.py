"""Run from project root: .venv/bin/python scripts/prepare_avian.py

Uses the official CIFAR-10 binary archive (downloaded automatically if absent).
Refuses to replace existing datasets. Requires numpy and Pillow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tarfile
import tempfile
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz'
MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'
CLASSES = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
AXIS = 'watch_interest'
PROMPT = 'Which image would you rather keep looking at?'


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def build(root, archive, batches, split, count, seen):
    rng = random.Random(276 if split == 'train' else 277)
    blocks = [np.frombuffer(archive.extractfile('cifar-10-batches-bin/' + b).read(), dtype='uint8').reshape(10000, 3073) for b in batches]
    data = np.concatenate(blocks)
    pixels = data[:, 1:].reshape(-1, 3, 32, 32)
    # Each coordinate uses the same 0..255 units: global channel means and SDs.
    features = np.concatenate([pixels.mean((2, 3)), pixels.std((2, 3))], axis=1)
    labels = data[:, 0]
    birds = list(map(int, np.flatnonzero(labels == 2)))
    rng.shuffle(birds)
    targets = list(range(3, 8)) * (count // 5)
    rng.shuffle(targets)
    sides = ['A', 'B'] * (count // 2)
    rng.shuffle(sides)
    (root / 'liked').mkdir(parents=True)
    (root / 'control').mkdir()
    ratings, pairs, origins = [], [], []

    def unused(i):
        return hashlib.sha256(pixels[i].tobytes()).digest() not in seen

    for k, (cls, side) in enumerate(zip(targets, sides, strict=True)):
        bird = birds.pop()
        while not unused(bird):
            bird = birds.pop()
        seen.add(hashlib.sha256(pixels[bird].tobytes()).digest())
        pool = np.flatnonzero(labels == cls)
        distance = ((features[pool] - features[bird]) ** 2).sum(axis=1)
        control = next(int(pool[j]) for j in np.argsort(distance, kind='stable') if unused(pool[j]))
        seen.add(hashlib.sha256(pixels[control].tobytes()).digest())
        group = f'{split}-{k:03}'
        paths = []
        for i, bank, score in [(bird, 'liked', 1), (control, 'control', -1)]:
            rel = f'{bank}/{split}-{i:05}.png'
            rgb = np.ascontiguousarray(pixels[i].transpose(1, 2, 0))
            Image.fromarray(rgb).save(root / rel)
            paths.append(rel)
            ratings.append(dict(image=rel, group=group, scores={AXIS: score}))
            origins.append(dict(image=rel, split=split, batch=batches[i // 10000], index_zero_based=i % 10000, label=CLASSES[labels[i]], rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        pairs.append(dict(images=paths if side == 'A' else paths[::-1], prompt=PROMPT, answer=side))
    write_rows(root / 'ratings.jsonl', ratings)
    write_rows(root / 'matched_pairs.jsonl', pairs)
    write_rows(root / 'source_records.jsonl', origins)
    (root / 'axes.yaml').write_text(f'{AXIS}:\n  low: prefer looking away\n  mid: indifferent\n  high: prefer keeping this image in view\n  question: {PROMPT}\n')
    (root / 'PROVENANCE.md').write_text(f'''# CIFAR-10 avian subset — official {split} split
Source: {URL} (published binary archive MD5 `{MD5}`). Dataset: https://www.cs.toronto.edu/~kriz/cifar.html . Reference: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images (2009).

Native 32×32 RGB, losslessly encoded as PNG. Ratings are class-derived preference targets, not human ratings. Seeded bird sampling, equal quotas of cat/deer/dog/frog/horse controls. Each control is the nearest available member of its assigned class by squared Euclidean distance on RGB means and standard deviations. No selected decoded image repeats across either dataset. This does not establish independence of photographic sessions or rule out near duplicates.

`source_records.jsonl` identifies the official batch and zero-based row for each PNG. `matched_pairs.jsonl` contains {count} non-overlapping pairs, equally split between bird on A and B. Use this file for fixed matching; Second Look's default prepare command instead samples other eligible pairs. Keep avian-final out of training, tuning and checkpoint selection. Rebuild with scripts/prepare_avian.py.
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('work/cifar/cifar-10-binary.tar.gz'))
    parser.add_argument('--output', type=Path, default=Path('datasets'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ('avian', 'avian-final')
    if any((args.output / n).exists() for n in names):
        parser.error('output banks already exist')
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL, args.archive)
    with args.archive.open('rb') as f:
        if hashlib.file_digest(f, 'md5').hexdigest() != MD5:
            raise ValueError('archive checksum differs from Toronto publication')
    with tempfile.TemporaryDirectory(dir=args.output) as tmp, tarfile.open(args.archive) as tar:
        seen = set()
        build(Path(tmp) / names[0], tar, [f'data_batch_{i}.bin' for i in range(1, 6)], 'train', 120, seen)
        build(Path(tmp) / names[1], tar, ['test_batch.bin'], 'test', 40, seen)
        for n in names:
            (Path(tmp) / n).rename(args.output / n)


if __name__ == '__main__':
    main()
