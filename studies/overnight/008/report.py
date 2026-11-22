"""Reproduce 008 using the unchanged 007 counting rules, not its fit labels."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
FROZEN_SHA256 = '49f72d45aed7fa26f9e35499e90c9614aa092574236d9ca3cef9379a4199d910'
sys.path.insert(0, str(HERE.parent))
metrics = importlib.import_module('007.report')


def freeze(expected=None):
    path = RESULTS/'blind-judgments.json'
    sidecar = (RESULTS/'blind-judgments.json.sha256').read_text().strip().split()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    metrics.require(sidecar == [digest, path.name] and digest == FROZEN_SHA256 and (expected is None or digest == expected),
                    'frozen checksum mismatch')
    return json.loads(data, object_pairs_hook=metrics.unique), digest


def summary(primary, paired, secondary):
    lines = ['# 008 development results', '',
             'All neutral counts pool three prompts on the same eight photos. Columns show base → adapter.', '',
             '| Axis | Neutral correct / known | Explicit correct / known |', '|---|---:|---:|']
    sides = ('before', 'after')
    b, a = (primary[s]['overall'] for s in sides)
    for axis, values in secondary['axes'].items():
        x, y = b['axes'][axis], a['axes'][axis]
        u, v = (values['models'][s] for s in sides)
        lines.append(f"| {axis} | {x['correct']} → {y['correct']} / {x['eligible']} | {u['correct']} → {v['correct']} / {u['eligible']} |")
    lines.extend(['', '| Neutral diagnostic | Base | Adapter |', '|---|---:|---:|'])
    for k in ('eligible','correct','mentions','absent','ambiguous','opposite','mid_disagreement','unknown','mentions_unknown'):
        x,y=(sum(axis[k] for axis in stage['axes'].values()) for stage in (b,a))
        lines.append(f'| {k} | {x} | {y} |')
    for k in ('responses','mixed_eligible','mixed_exact',*metrics.FLAGS,'literal_phrase_echo'):
        lines.append(f'| {k} | {b[k]} | {a[k]} |')
    gains,losses = (sum(v['all_axes'][k] for v in paired['photo'].values()) for k in ('gains','losses'))
    lines.extend(['', f'Matched known fields: gains = {gains}; losses = {losses}.', '',
                  '| Explicit diagnostic | Correct / known | Ordinal absolute error sum |', '|---|---:|---:|'])
    for name in (*sides,'constant_M','fit_median'):
        c = secondary['overall'][name]
        lines.append(f"| {name} | {c['correct']} / {c['eligible']} | {c['absolute_error_sum']} |")
    lines.extend(['', 'For comparison, 007 on these same photos had 14 → 14 / 186 neutral correct, three gains and three losses, 5 → 10 grounding flags, and 38 → 45 / 62 explicit correct. Cross-study judgments were frozen separately; they are not a single pooled blind review. Neither study tested independent generalization or Steve’s preferences.'])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS)
    parser.add_argument('--judgments-sha256', help='independent reviewer digest; required for initial identity join')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--join-from', type=Path, help='private work directory, after freeze only')
    mode.add_argument('--secondary-only', action='store_true')
    args = parser.parse_args()
    if args.join_from and not args.judgments_sha256:
        parser.error('--join-from requires --judgments-sha256')
    manifest = metrics.load(RESULTS/'manifest.json')
    metrics.require(manifest['data_sha256'] == 'e87c8eb796a3fa4dd0091d13600d5b21344784d27ff11d8acdb7005282b5725c', '008 manifest mismatch')
    axes, photos, dev, neutral, explicit = metrics.inputs(HERE, RESULTS, RESULTS)
    secondary = metrics.secondary(axes, photos, dev, explicit)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.secondary_only:
        metrics.save(args.out/'secondary.json', secondary)
        return
    judgments, digest = freeze(args.judgments_sha256)
    if args.join_from:
        work = args.join_from
        audit = dict(key=metrics.load(work/'blind-key.private.json'), blind=metrics.load(work/'blind.json'),
                     packet=metrics.load(work/'review-packet/responses.json'),
                     photo_sha256={str(n):metrics.sha(work/f'review-packet/{n}.jpg') for n in range(1,9)})
    else:
        audit = metrics.load(RESULTS/'blind-audit.json')
    join = metrics.join_records(audit['key'], audit['blind'], audit['packet'], lambda n: audit['photo_sha256'][str(n)], HERE)
    primary, paired, rows = metrics.primary(axes, photos, neutral, join, judgments, metrics.load(HERE/'targets.json'))
    primary['judgments_sha256'] = digest
    for name, obj in [('blind-audit',audit), ('join',join), ('primary',primary), ('paired',paired), ('joined-rows',rows), ('secondary',secondary)]:
        metrics.save(args.out/f'{name}.json', obj)
    (args.out/'summary.md').write_text(summary(primary, paired, secondary))


if __name__ == '__main__':
    main()
