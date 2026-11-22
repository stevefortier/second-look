"""Reproduce 010 on CPU; first join requires a separately pinned, frozen review."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
prior = importlib.import_module('009.report')
metrics, require, FIXED = prior.metrics, prior.require, prior.FIXED
RESULTS = HERE / 'results'
PIN = HERE / 'freeze-pin.sha256'
FOCUS_SHA = '5afa66d19741c799e1f80d70b0bb91178f90ccd092eb1c1c4dc507f4b0b23a5a'
SCHEDULE_SHA = 'c07c9dc7f97150e3ef76d1cbba9a201e7a897b86bc32913a3e2d6660ffac56f3'


def checked_inputs(source):
    for name, digest in prior.FIXED_HASHES.items():
        require(metrics.sha(FIXED / name) == digest, 'fixed input mismatch: ' + name)
    require(metrics.sha(HERE / 'focused.json') == FOCUS_SHA, 'focused caption checksum mismatch')
    schedule = (HERE / 'schedule.sha256').read_text().strip()
    require(schedule == SCHEDULE_SHA, 'schedule pin changed')
    manifest = metrics.load(source / 'manifest.json')
    fixed_manifest = metrics.load(FIXED / 'results/manifest.json')
    require(manifest == dict(fixed_manifest, steps=157, data_sha256=schedule), '010 manifest mismatch')
    train = metrics.load(source / 'train.json')
    require(train['revision'] == manifest['revision'] and train['seed'] == manifest['seed']
            and train['lr'] == 7e-5, '010 training provenance mismatch')
    rows = train['steps']
    require([r['step'] for r in rows] == list(range(1, 158)), '010 step order mismatch')
    items = [[r[k] for k in ('image', 'kind', 'prompt', 'answer')] for r in rows]
    require(hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest() == schedule,
            '010 ordered schedule mismatch')
    full, focus = metrics.load(FIXED / 'targets.json'), metrics.load(HERE / 'focused.json')
    require(set(full) == set(focus) and len(full) == 16, 'caption split mismatch')
    captions = Counter((r['image'], r['answer']) for r in rows if r['kind'] == 'caption')
    require(captions == Counter(full.items()) + Counter(focus.items()), 'caption exposure mismatch')
    # 007.inputs requires ONE full caption per image (last-write-wins) and a
    # matching step count. After authenticating the entire real schedule, give
    # it the original 141-row training subset only. Never rewrite run artifacts.
    original = [r for r in rows if r['kind'] == 'score' or r['answer'] == full[r['image']]]
    with TemporaryDirectory() as tmp:
        view = Path(tmp)
        metrics.save(view / 'train.json', {'steps': original})
        metrics.save(view / 'manifest.json', fixed_manifest)
        for side in ('before', 'after'):
            (view / f'{side}.json').symlink_to((source / f'{side}.json').resolve())
        inputs = metrics.inputs(FIXED, view, FIXED / 'results')
    corpus = {f'{kind}:{image}': caption for kind, family in [('full', full), ('focused', focus)]
              for image, caption in family.items()}
    return inputs, corpus


def comparison(bundle):
    primary, paired, secondary = bundle
    values = {}
    for side in ('before', 'after'):
        overall = primary[side]['overall']
        axes = overall['axes']
        def add(name, value):
            values.setdefault(name, []).append(str(value))
        for axis, row in axes.items():
            add(axis + ' correct / known', f"{row['correct']} / {row['eligible']}")
        for field in ('eligible', 'correct', 'mentions', 'absent', 'ambiguous', 'opposite',
                      'mid_disagreement', 'unknown', 'mentions_unknown'):
            add('All fields: ' + field, sum(row[field] for row in axes.values()))
        for field in ('responses', 'mixed_eligible', 'mixed_exact', *metrics.FLAGS, 'literal_phrase_echo'):
            add(field, overall[field])
        for axis, row in secondary['axes'].items():
            c = row['models'][side]
            add('Explicit ' + axis + ' correct / known', f"{c['correct']} / {c['eligible']}")
    for field in ('gains', 'losses'):
        values['Paired ' + field] = [str(sum(p['all_axes'][field] for p in paired['photo'].values()))]
    for side, c in secondary['overall'].items():
        values['Explicit ' + side + ' correct / known; MAE'] = [f"{c['correct']} / {c['eligible']}; {c['mae']:.3f}"]
    return {k: ' → '.join(v) for k, v in values.items()}


def summary(bundle):
    previous = [comparison([metrics.load(HERE.parent / study / 'results' / f'{name}.json')
                            for name in ('primary', 'paired', 'secondary')]) for study in ('008', '009')]
    current = comparison(bundle)
    lines = ['# 010 focused-caption development results', '',
             'Base → adapter on the same eight development photos and three neutral prompts.', '',
             '| Measure | 008 | 009 | 010 |', '|---|---:|---:|---:|']
    lines.extend(f'| {k} | {previous[0][k]} | {previous[1][k]} | {v} |' for k, v in current.items())
    lines.extend(['', 'The six-word echo diagnostic uses the actual caption families for each study: '
                  '010 includes full and focused captions. Explicit agreement is secondary. These are paired '
                  'observations on reused photos with proxy labels, not independent replication or Steve’s ratings. '
                  'Steps, target tokens, prompt exposure, order and writing style differ.'])
    return '\n'.join(lines) + '\n'


def run(results, out, work, pin, expected):
    hashes = prior.pins(pin)
    digest = hashes[prior.FROZEN]
    require(work is None or expected is not None, 'initial join requires independent reviewer digest')
    require(expected is None or expected == digest, 'frozen checksum mismatch')
    require((results / (prior.FROZEN + '.sha256')).read_text().split() == [digest, prior.FROZEN],
            'frozen checksum mismatch')
    data = (results / prior.FROZEN).read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, 'frozen checksum mismatch')
    judgments = json.loads(data, object_pairs_hook=metrics.unique)
    with TemporaryDirectory() as tmp:
        source = Path(tmp)
        for name in prior.STAGES:
            payload = ((work or results) / name).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == hashes[name], '010 artifact checksum mismatch: ' + name)
            (source / name).write_bytes(payload)
        (axes, photos, dev, neutral, explicit), corpus = checked_inputs(source)
        secondary = metrics.secondary(axes, photos, dev, explicit)
        if work is None:
            audit = metrics.load(results / 'blind-audit.json')
        else:
            audit = dict(key=metrics.load(work / 'blind-key.private.json'), blind=metrics.load(work / 'blind.json'),
                         packet=metrics.load(work / 'review-packet/responses.json'),
                         photo_sha256={str(n): metrics.sha(work / f'review-packet/{n}.jpg') for n in range(1, 9)})
        join = metrics.join_records(audit['key'], audit['blind'], audit['packet'],
                                    lambda n: audit['photo_sha256'][str(n)], FIXED)
        primary, paired, rows = metrics.primary(axes, photos, neutral, join, judgments, corpus)
        primary['judgments_sha256'] = digest
        text = summary((primary, paired, secondary))
        out.mkdir(parents=True, exist_ok=True)
        if work is not None:
            for name in prior.STAGES:
                (out / name).write_bytes((source / name).read_bytes())
        for name, obj in [('blind-audit', audit), ('join', join), ('primary', primary), ('paired', paired),
                          ('joined-rows', rows), ('secondary', secondary)]:
            metrics.save(out / f'{name}.json', obj)
        (out / 'summary.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS)
    parser.add_argument('--join-from', type=Path, help='checksum-verified 010 backup; only after review freeze')
    parser.add_argument('--judgments-sha256', help='independent reviewer digest; mandatory for initial join')
    args = parser.parse_args()
    if args.join_from is not None and not args.judgments_sha256:
        parser.error('--join-from requires --judgments-sha256')
    run(RESULTS, args.out, args.join_from, PIN, args.judgments_sha256)


if __name__ == '__main__':
    main()
