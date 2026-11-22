"""Reproduce 009 from pinned post-freeze artifacts and the fixed 008 input metadata."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
FIXED = HERE.parent/'008'
RESULTS = HERE/'results'
PIN = HERE/'freeze-pin.sha256'  # independent, separately committed; never written here
STAGES = ('manifest.json', 'train.json', 'before.json', 'after.json')
FROZEN = 'blind-judgments.json'
PIN_NAMES = (FROZEN, *STAGES)
SCHEDULE = '27b18f65275177be9cb3cdf5c2755d1f6f554c02ad23453cc882a878da771c58'
FIXED_HASHES = {
    'protocol.json': '5d36865421ff5e25474eb0c7378549b477727c45bca2324c87937321825d438b',
    'targets.json': 'e05d590531a617c56fa86c4ee549f9aa4df7427958a2a1051a5dfd12e6f759d0',
    'results/manifest.json': 'cba61c4fd35dc0666ae5b8fdf67d66cf037f8ad53369c795474105e1e9bc06d0',
}
sys.path.insert(0, str(HERE.parent))
metrics = importlib.import_module('007.report')
require = metrics.require


def pins(path):
    out = {}
    for line in path.read_text().splitlines():
        m = re.fullmatch(r'([a-f0-9]{64})  ([\w.-]+)', line)
        if m is None or m[2] in out:
            raise ValueError('invalid or duplicate checksum pin')
        out[m[2]] = m[1]
    require(set(out) == set(PIN_NAMES), 'incomplete freeze pin')
    return out


def provenance(source):
    for name, digest in FIXED_HASHES.items():
        require(metrics.sha(FIXED/name) == digest, 'fixed input mismatch: '+name)
    manifest = metrics.load(source/'manifest.json')
    require(manifest == metrics.load(FIXED/'results/manifest.json'), '009 manifest mismatch')
    train = metrics.load(source/'train.json')
    require(train['revision'] == manifest['revision'] and train['seed'] == manifest['seed']
            and train['lr'] == 7e-5, '009 training provenance mismatch')
    steps = train['steps']
    require([r['step'] for r in steps] == list(range(1, 142)), '009 training step order mismatch')
    items = [[r[k] for k in ('image', 'kind', 'prompt', 'answer')] for r in steps]
    digest = hashlib.sha256(json.dumps(items, separators=(',', ':')).encode()).hexdigest()
    require(digest == SCHEDULE, '009 ordered schedule mismatch')


def summary(current, previous):
    lines = ['# 009 matched-budget development results', '',
             'Each neutral field is scored on three prompts for the same eight dev photos. '
             'Counts are base → adapter, not independent samples.', '',
             '| Neutral / diagnostic | 008 | 009 |', '|---|---:|---:|']
    def values(bundle):
        primary, paired, secondary = bundle
        def both(fn):
            return ' → '.join(str(fn(primary[s]['overall'])) for s in ('before', 'after'))
        result = {}
        for axis in secondary['axes']:
            result[axis+' correct / known'] = both(lambda x: f"{x['axes'][axis]['correct']} / {x['axes'][axis]['eligible']}")
        for field in ('eligible', 'correct', 'mentions', 'absent', 'ambiguous', 'opposite', 'mid_disagreement', 'unknown', 'mentions_unknown'):
            result['All fields: '+field] = both(lambda x: sum(a[field] for a in x['axes'].values()))
        for field in ('responses', 'mixed_eligible', 'mixed_exact', *metrics.FLAGS, 'literal_phrase_echo'):
            result[field] = both(lambda x: x[field])
        for field in ('gains', 'losses'):
            result['Paired '+field] = sum(v['all_axes'][field] for v in paired['photo'].values())
        for axis, entry in secondary['axes'].items():
            result['Explicit '+axis+' correct / known'] = ' → '.join(f"{c['correct']} / {c['eligible']}" for c in (entry['models'][s] for s in ('before','after')))
        for side, c in secondary['overall'].items():
            result['Explicit '+side+' correct / known; ordinal error sum'] = f"{c['correct']} / {c['eligible']}; {c['absolute_error_sum']}"
        return result
    old, new = values(previous), values(current)
    lines.extend(f'| {name} | {old[name]} | {value} |' for name, value in new.items())
    lines.extend(['', 'Judgments were frozen separately. Explicit probes and six-word caption echoes are diagnostics. '
                  'The schedule changed both caption repetition and score exposure; neither independent generalization nor Steve’s preferences were tested.'])
    return '\n'.join(lines)+'\n'


def run(results, out, work, pin, expected):
    hashes = pins(pin)
    digest = hashes[FROZEN]
    require(work is None or expected is not None, 'initial join requires independent reviewer digest')
    require(expected is None or expected == digest, 'frozen checksum mismatch')
    require((results/(FROZEN+'.sha256')).read_text().split() == [digest, FROZEN], 'frozen checksum mismatch')
    data = (results/FROZEN).read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, 'frozen checksum mismatch')
    judgments = json.loads(data, object_pairs_hook=metrics.unique)
    source = results if work is None else work
    for name in STAGES:
        require(metrics.sha(source/name) == hashes[name], '009 artifact checksum mismatch: '+name)
    provenance(source)
    axes, photos, dev, neutral, explicit = metrics.inputs(FIXED, source, FIXED/'results')
    secondary = metrics.secondary(axes, photos, dev, explicit)
    if work is None:
        audit = metrics.load(results/'blind-audit.json')
    else:
        audit = dict(key=metrics.load(work/'blind-key.private.json'), blind=metrics.load(work/'blind.json'),
                     packet=metrics.load(work/'review-packet/responses.json'),
                     photo_sha256={str(n):metrics.sha(work/f'review-packet/{n}.jpg') for n in range(1,9)})
    join = metrics.join_records(audit['key'], audit['blind'], audit['packet'], lambda n:audit['photo_sha256'][str(n)], FIXED)
    primary, paired, rows = metrics.primary(axes, photos, neutral, join, judgments, metrics.load(FIXED/'targets.json'))
    primary['judgments_sha256'] = digest
    previous = [metrics.load(FIXED/f'results/{name}.json') for name in ('primary','paired','secondary')]
    text = summary((primary, paired, secondary), previous)
    out.mkdir(parents=True, exist_ok=True)
    if work is not None:
        for name in STAGES:
            payload = (source/name).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == hashes[name], 'run changed during join')
            if (source/name).resolve() != (out/name).resolve():
                (out/name).write_bytes(payload)
    for name, obj in [('blind-audit',audit), ('join',join), ('primary',primary), ('paired',paired), ('joined-rows',rows), ('secondary',secondary)]:
        metrics.save(out/f'{name}.json', obj)
    (out/'summary.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS)
    parser.add_argument('--join-from', type=Path, help='post-freeze 009 backup with private packet/key')
    parser.add_argument('--judgments-sha256', help='independent reviewer digest, required for first join')
    args = parser.parse_args()
    if args.join_from and not args.judgments_sha256:
        parser.error('--join-from requires --judgments-sha256')
    run(RESULTS, args.out, args.join_from, PIN, args.judgments_sha256)


if __name__ == '__main__':
    main()
