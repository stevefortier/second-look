"""Protect the shipped, offline test images and the paired evaluation protocol."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'datasets/legacy-coco'


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def test_active_images_match_manifest_and_geometry():
    manifest = records(ROOT/'manifest.jsonl')
    test = [r for r in manifest if r['split']=='val2017']
    assert len(test)==80
    assert not {r['sha256'] for r in test} & {r['sha256'] for r in manifest if r['split']=='train2017'}
    banks = {bank:Counter(r['aspect'] for r in test if r['bank']==bank) for bank in ('liked','control')}
    assert banks['liked']==banks['control'] and len(banks['liked'])==3
    assert {r['image'] for r in test}=={r['image'] for r in records(ROOT/'photo-credits.jsonl')}
    for row in test:
        p = ROOT/'test-images'/row['image']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        with Image.open(p) as im:
            assert im.size==(row['width'],row['height']) and min(im.size)>=256
    pairs=records(ROOT/'test-images/matched_pairs.jsonl')
    assert len(pairs)==40 and Counter(p['answer'] for p in pairs)=={'A':20,'B':20}
    assert {p for pair in pairs for p in pair['images']}=={r['image'] for r in test}
