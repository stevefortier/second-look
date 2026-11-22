"""Check the versioned quality example without network/model dependencies."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import yaml


def test_shipped_quality_example():
    root = Path(__file__).resolve().parents[1] / 'datasets/example'
    axes = set(yaml.safe_load((root/'axes.yaml').read_text()))
    assert {'liking','beauty','scary'} <= axes
    sources = {r['image']: r['sha256'] for r in map(json.loads,(root/'sources.jsonl').read_text().splitlines())}
    seen, groups, actual = set(), set(), set()
    for split, count in [('train',24),('test',8)]:
        rows = list(map(json.loads,(root/split/'ratings.jsonl').read_text().splitlines()))
        assert len(rows)==count and not groups.intersection(r['group'] for r in rows)
        groups.update(r['group'] for r in rows)
        for row in rows:
            rel=Path(split)/row['image']; assert rel.parts[1]=='images'
            photo=root/rel; digest=hashlib.sha256(photo.read_bytes()).hexdigest()
            assert digest==sources[rel.as_posix()] and digest not in seen
            seen.add(digest);actual.add(rel.as_posix())
            with Image.open(photo) as im: assert min(im.size)>=256
            assert set(row['scores'])==axes and all(v is None or -1<=v<=1 for v in row['scores'].values())
    assert actual==set(sources)
