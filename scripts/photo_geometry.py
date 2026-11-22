"""Audit on-disk photos, not claimed metadata or the dimensions of resized tensors."""
import argparse
from collections import Counter
import json
from pathlib import Path
from PIL import Image, ImageOps


def audit(root):
    result = {}
    for bank in ('liked', 'control'):
        sizes = []
        for path in sorted((root / bank).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in Image.registered_extensions():
                continue
            with Image.open(path) as image:
                w, h = ImageOps.exif_transpose(image).size
            if min(w, h) < 256:
                raise ValueError(f'{path}: {w}x{h}; both dimensions must be at least 256')
            sizes.append((w, h))
        if not sizes:
            raise ValueError(f'{root / bank}: no photos')
        aspects = Counter('portrait' if w/h < 0.8 else 'landscape' if w/h > 1.25 else 'near_square' for w,h in sizes)
        result[bank] = dict(images=len(sizes), distinct_sizes=len(set(sizes)),
                            short_edge_min=min(min(s) for s in sizes),
                            long_edge_max=max(max(s) for s in sizes), aspects=dict(aspects),
                            width_height_ratio_range=[min(w/h for w,h in sizes),max(w/h for w,h in sizes)])
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    a=parser.parse_args(); print(json.dumps({str(root):audit(root) for root in a.roots},indent=2))
