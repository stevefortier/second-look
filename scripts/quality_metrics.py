"""Local secondary diagnostics; not an automated score for spontaneous attribution.

summary --before before.json --after after.json --truth eval/ratings.jsonl
        --train fit/ratings.jsonl [--output summary.json]
packet  --before before.json --after after.json --truth eval/ratings.jsonl
        --output packet.json --key private/join.json

Both inputs are lists of {image,prompt,kind,text}; kind=score also needs
axis and numeric prediction. Share only packet.json; keep the key local.
Prompt and response prose are verbatim: inspect them for self-identification
before sharing. Supply the same axis endpoint definitions to every reviewer.

Use fit-only ratings for --train. Join by exact image, kind, prompt and axis,
not list position. Scores are in [-1,1]. MAE weights known evaluation rows
once each; ordinal pairs compare different images within the same axis/prompt,
exclude truth ties, and give prediction ties half credit. Null metrics mean
no applicable observations. No model, network, or image access required.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
from secrets import SystemRandom, token_hex
from statistics import mean, median


def label(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'expected nonempty {field}')
    return value


def score(value):
    if type(value) not in (int, float) or not -1 <= value <= 1:
        raise ValueError('expected finite numeric score in [-1,1]')
    return value


def ratings(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = label(row, 'image')
        if image in result or not isinstance(row.get('scores'), dict):
            raise ValueError(f'duplicate image or missing scores in {path}: {image}')
        result[image] = {k: None if v is None else score(v) for k, v in row['scores'].items()}
    if not result:
        raise ValueError(f'empty ratings: {path}')
    return result


def predictions(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected nonempty JSON list of generation records')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('text'), str):
            raise ValueError('generation needs response text')
        kind = row.get('kind')
        if kind not in ('score', 'neutral'):
            raise ValueError('unknown generation kind')
        axis = label(row, 'axis') if kind == 'score' else None
        key = (label(row, 'image'), kind, label(row, 'prompt'), axis)
        if key in result:
            raise ValueError(f'duplicate generation: {key}')
        if kind == 'score':
            score(row.get('prediction'))
        result[key] = row
    return result


def avg(values):
    return mean(values) if values else None


def diagnostics(rows):
    preds = [p for _, _, p in rows]
    by_prompt = defaultdict(list)
    for prompt, y, p in rows:
        by_prompt[prompt].append((y, p))
    wins = ties = pairs = 0
    for group in by_prompt.values():
        for (a, x), (b, y) in combinations(group, 2):
            if a != b:
                pairs += 1
                ties += x == y
                wins += (x > y) == (a > b) and x != y
    return {'mae': avg([abs(y-p) for _, y, p in rows]), 'ordinal_pairs': pairs,
            'prediction_ties': ties, 'ordinal_agreement': (wins + ties/2)/pairs if pairs else None,
            'distinct_predictions': len(set(preds)), 'prediction_min': min(preds, default=None),
            'prediction_max': max(preds, default=None),
            'polarity_agreement': avg([((y > 0)-(y < 0)) == ((p > 0)-(p < 0)) for _, y, p in rows])}


def summary(before, after, truth, train):
    axes = sorted({ax for scores in truth.values() for ax in scores})
    groups = defaultdict(list)
    for key in before:
        image, kind, prompt, axis = key
        if image not in truth or (kind == 'score' and axis not in truth[image]):
            raise ValueError(f'generation has no matching truth image/axis: {key}')
        if kind == 'score':
            groups[axis].append((image, prompt, truth[image][axis], before[key]['prediction'], after[key]['prediction']))
    report = {}
    for ax in axes:
        known = [row for row in groups[ax] if row[2] is not None]
        train_y = [r[ax] for r in train.values() if r.get(ax) is not None]
        constant = median(train_y) if train_y else None
        ys = [row[2] for row in known]
        report[ax] = {'rated_images': sum(r.get(ax) is not None for r in truth.values()),
                      'scored_images': len({row[0] for row in known}), 'scored_rows': len(known),
                      'unknown_rows': len(groups[ax])-len(known), 'distinct_truth': len(set(ys)),
                      'train_rated_images': len(train_y), 'train_median': constant,
                      'train_median_mae': avg([abs(y-constant) for y in ys]) if train_y else None,
                      'zero_mae': avg([abs(y) for y in ys]),
                      **{name: diagnostics([(r[1], r[2], r[i]) for r in known])
                         for i, name in ((3, 'before'), (4, 'after'))}}
    return {'endpoint': 'secondary_explicit_score_only', 'axes': report}


def write_packet(before, after, truth, output, key_path):
    axes = sorted({axis for scores in truth.values() for axis in scores})
    image_ids, join, items = {}, {}, []
    for name, records in (('before', before), ('after', after)):
        for (image, kind, prompt, _), row in records.items():
            if kind != 'neutral':
                continue
            if image not in truth:
                raise ValueError(f'neutral image not in evaluation bank: {image}')
            if image not in image_ids:
                image_ids[image] = token_hex(12)
            item_id = token_hex(12)
            join[item_id] = {'image': image, 'image_id': image_ids[image], 'side': name, 'prompt': prompt}
            items.append({'item_id': item_id, 'image_id': image_ids[image], 'prompt': prompt,
                          'text': row['text'], 'annotations': {axis: dict.fromkeys(('mention', 'polarity', 'evidence'))
                                                               for axis in axes}})
    if not items:
        raise ValueError('no neutral responses for annotation')
    SystemRandom().shuffle(items)
    public = {'rubric': 'Judge only the response. For each axis mark mention present, absent or ambiguous. '
              'Credit equivalent wording, not general praise or a bare list of traits. For a present trait, '
              'mark polarity positive, midpoint, negative or ambiguous relative to the supplied endpoints. '
              'Use null polarity for absent traits; absence is not midpoint. Quote evidence for each judgment.',
              'items': items}
    with open(key_path, 'x', opener=lambda p, flags: os.open(p, flags, 0o600)) as key:
        try:
            json.dump(join, key, indent=2)
            key.flush()
            with output.open('x') as stream:
                json.dump(public, stream, indent=2)
        except BaseException:
            key_path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    for command, extra in [('summary', ('train',)), ('packet', ('key',))]:
        sub = subs.add_parser(command)
        for name in ('before', 'after', 'truth', *extra):
            sub.add_argument('--'+name, type=Path, required=True)
        sub.add_argument('--output', type=Path, required=command == 'packet')
    args = parser.parse_args()
    try:
        before, after = predictions(args.before), predictions(args.after)
        if before.keys() != after.keys():
            raise ValueError('before/after record keys differ')
        if args.command == 'packet':
            write_packet(before, after, ratings(args.truth), args.output, args.key)
            return
        if args.train.samefile(args.truth):
            raise ValueError('use distinct fit and evaluation ratings')
        result = json.dumps(summary(before, after, ratings(args.truth), ratings(args.train)), indent=2)
        if args.output:
            with args.output.open('x') as out:
                print(result, file=out)
        else:
            print(result)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
