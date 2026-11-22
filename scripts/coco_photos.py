"""Build or restore the original COCO bird/control photo banks; see experiments/avian-photos."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
from contextlib import nullcontext
import hashlib
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile
from PIL import Image, ImageOps

BASE = 'https://s3.amazonaws.com/images.cocodataset.org'
ARCHIVE = BASE + '/annotations/annotations_trainval2017.zip'
ARCHIVE_SHA = '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'
NAMES = {'train2017': 'avian-photos', 'val2017': 'avian-photos-final'}
PROMPT = 'Which image would you rather keep looking at?'
AXES = f'watch_interest:\n  low: prefer looking away\n  mid: indifferent\n  high: prefer keeping this image in view\n  question: {PROMPT}\n'


def bucket(width, height):
    if min(width, height) < 256:
        return None
    if width * 5 < height * 4:
        return 'portrait'
    if width * 4 > height * 5:
        return 'landscape'
    return 'near-square'


def select(rows, count):
    pools = defaultdict(list)
    for row in sorted(rows, key=lambda r: sha256(f"avian-coco-v1:{r['id']}".encode()).digest()):
        pools[bucket(row['width'], row['height']), row['bank']].append(row)
    pairs = []
    while len(pairs) < count:
        before = len(pairs)
        for aspect in ('portrait', 'near-square', 'landscape'):
            birds, controls = pools[aspect, 'liked'], pools[aspect, 'control']
            if birds and controls and len(pairs) < count:
                pairs.append((birds.pop(), controls.pop()))
        if len(pairs) == before:
            raise ValueError('not enough aspect-matched pairs')
    return pairs


def source_rows(archive, split, count):
    import ijson  # optional selection-only dependency; stream large polygons

    def items(key, kind='instances'):
        with archive.open(f'annotations/{kind}_{split}.json') as source:
            yield from ijson.items(source, key + '.item', use_float=True)

    categories = {c['id']: c for c in items('categories')}
    licenses = {c['id']: c for c in items('licenses') if 1 <= c['id'] <= 6}
    images = {i['id']: i for i in items('images') if bucket(i['width'], i['height']) and i['license'] in licenses}
    labels, visible = defaultdict(set), defaultdict(set)
    for a in items('annotations'):
        img = images.get(a['image_id'])
        if img is None:
            continue
        category = categories[a['category_id']]
        labels[img['id']].add(category['name'])
        if category['supercategory'] == 'animal' and not a['iscrowd'] and a['area'] >= .02 * img['width'] * img['height']:
            visible[img['id']].add(category['name'])
    animals = {c['name'] for c in categories.values() if c['supercategory'] == 'animal'}
    candidates = []
    for ident, animals_visible in visible.items():
        names = labels[ident]
        if 'bird' in names and (animals_visible != {'bird'} or names & (animals - {'bird'})):
            continue
        bank = 'liked' if 'bird' in names else 'control'
        img = images[ident]
        candidates.append(dict(id=ident, bank=bank, width=img['width'], height=img['height'],
            split=split, aspect=bucket(img['width'], img['height']), categories=sorted(names),
            visible_animals=sorted(animals_visible), license=licenses[img['license']], flickr_url=img['flickr_url'],
            url=f"{BASE}/{split}/{img['file_name']}", image=f"{bank}/{img['file_name']}", caption_ids=[]))
    chosen = {}
    for index, pair in enumerate(select(candidates, count)):
        for row in pair:
            row['pair'] = index
            chosen[row['id']] = row
    for caption in items('annotations', 'captions'):
        if caption['image_id'] in chosen:
            chosen[caption['image_id']]['caption_ids'].append(caption['id'])
    if not all(row['caption_ids'] for row in chosen.values()):
        raise ValueError('selected image missing captions')
    return list(chosen.values())


def identify(data, row):
    digest = sha256(data).hexdigest()
    if 'sha256' in row and digest != row['sha256']:
        raise ValueError('JPEG sha256 mismatch')
    with Image.open(io.BytesIO(data)) as src:
        rgb = ImageOps.exif_transpose(src).convert('RGB')
        if src.format != 'JPEG' or rgb.size != (row['width'], row['height']) or not bucket(*rgb.size):
            raise ValueError('unexpected image format or dimensions')
        pins = dict(sha256=digest, bytes=len(data), rgb_sha256=sha256(rgb.tobytes()).hexdigest())
    for key, value in pins.items():
        if key in row and row[key] != value:
            raise ValueError(f'{key} mismatch')
    return pins


def jsonl(rows):
    return ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows)


def download(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=45) as response:
                return response.read()
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def materialize(rows, output, *, verify=False, fetch=download):
    output.mkdir(parents=True, exist_ok=True)
    splits = sorted({row['split'] for row in rows})
    for split in splits:
        if not verify and (output / NAMES[split]).exists():
            raise FileExistsError(output / NAMES[split])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate COCO image ID')
    with (nullcontext(output) if verify else TemporaryDirectory(dir=output)) as dest:
        dest = Path(dest)
        for split in splits:
            if not verify:
                for bank in ('liked','control'):
                    (dest / NAMES[split] / bank).mkdir(parents=True)

        def copy(row):
            name = f"{row['id']:012d}.jpg"
            if (row['bank'] not in ('liked','control') or row['image'] != f"{row['bank']}/{name}"
                    or row['url'] != f"{BASE}/{row['split']}/{name}"):
                raise ValueError('invalid COCO row')
            path = dest / NAMES[row['split']] / row['image']
            data = path.read_bytes() if verify else fetch(row['url'])
            pins = identify(data, row)
            if not verify:
                path.write_bytes(data)
            return {**row, **pins}

        with ThreadPoolExecutor(max_workers=8) as pool:
            pinned = list(pool.map(copy, rows))
        if len({r['rgb_sha256'] for r in pinned}) != len(pinned):
            raise ValueError('duplicate decoded photo')
        for split in splits:
            root = dest / NAMES[split]
            subset = [r for r in pinned if r['split'] == split]
            ratings, pairs, groups = [], [], defaultdict(dict)
            for row in subset:
                index, bank = row['pair'], row['bank']
                groups[index][bank] = row
                ratings.append(dict(image=row['image'],group=f'{split}-{index:03}',scores={'watch_interest':1 if bank=='liked' else -1}))
            for index, group in sorted(groups.items()):
                bird, control = group['liked'], group['control']
                if bird['aspect'] != control['aspect']:
                    raise ValueError('aspect mismatch')
                images = [bird['image'],control['image']]
                pairs.append(dict(images=images[::-1] if index%2 else images,prompt=PROMPT,answer='B' if index%2 else 'A'))
            metadata = {'axes.yaml':AXES, 'ratings.jsonl':jsonl(ratings), 'matched_pairs.jsonl':jsonl(pairs), 'source_records.jsonl':jsonl(subset)}
            for name, text in metadata.items():
                if verify:
                    if (root/name).read_bytes() != text.encode():
                        raise ValueError(f'metadata mismatch: {name}')
                else:
                    (root/name).write_bytes(text.encode())
            expected = set(metadata) | {r['image'] for r in subset}
            if expected != {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}:
                raise ValueError('unexpected files in bank')
        if not verify:
            for split in splits:
                (dest / NAMES[split]).rename(output / NAMES[split])
    return pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build','restore','verify'))
    parser.add_argument('--output', type=Path, default=Path('datasets'))
    parser.add_argument('--manifest', type=Path, default=Path('experiments/avian-photos/manifest.jsonl'))
    parser.add_argument('--archive', type=Path, default=Path('work/coco2017/annotations_trainval2017.zip'))
    args = parser.parse_args()
    checksum = args.manifest.with_suffix('.sha256')
    if args.mode == 'build':
        with args.archive.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() != ARCHIVE_SHA:
                raise ValueError('annotation ZIP sha256 mismatch')
        with ZipFile(args.archive) as archive:
            rows = source_rows(archive,'train2017',120) + source_rows(archive,'val2017',40)
        pinned = materialize(rows,args.output)
        data = jsonl(pinned).encode()
        digest = sha256(data).hexdigest()
        pin = f'{digest}  {args.manifest.name}\n'
        for path, content in [(args.manifest,data),(checksum,pin.encode())]:
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f'rebuild differs from {path}')
            if not path.exists():
                path.write_bytes(content)
    else:
        data = args.manifest.read_bytes()
        digest = sha256(data).hexdigest()
        if checksum.read_text().split() != [digest,args.manifest.name]:
            raise ValueError('manifest sha256 mismatch')
        pinned = [json.loads(line) for line in data.splitlines()]
        if not all({'bytes','sha256','rgb_sha256'} <= row.keys() for row in pinned):
            raise ValueError('manifest missing photo pins')
        if args.mode == 'restore':
            materialize(pinned,args.output)
    materialize(pinned,args.output,verify=True)
    print(f'{args.mode}: verified {len(pinned)} original JPEGs; manifest sha256={digest}')


if __name__ == '__main__':
    main()
