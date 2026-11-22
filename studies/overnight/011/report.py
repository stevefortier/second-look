"""Reproduce 011 on CPU; the first join requires the independent reviewer's freeze digest."""
import importlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
previous = importlib.import_module('010.report')
metrics, require = previous.metrics, previous.require
RESULTS, PIN = HERE / 'results', HERE / 'freeze-pin.sha256'
LAUNCH_MANIFEST = '5f44c258f5c7d7b5ab43f0eaa30409feacc8a1aba9c896d0339adc8f2fbf40cc'
ARTIFACT_LIST = '34bca49612aba4a7c40e56023904cd19c3a07d41d555b2ebd5b67e251f1953fa'
BASELINE = {
    'primary': '9fd46d7ec8ddc679218db8fa68f3d05e2e27187b972c5ca9538a18ef469269fd',
    'paired': 'f6c8c2c9bf8d9dbf709a5151222d4dfecf3eda52397cff8962f8d736b30e4a2e',
    'secondary': '93c349ce83e330a9fc9d9ddb8f033dad33237b33d5122373b01ce1bb55b49cd1',
}


def checked_inputs(source):
    require(metrics.sha(source / 'manifest.json') == LAUNCH_MANIFEST, '011 launch manifest mismatch')
    # The launch receipt includes remote-only trainer bytes. Authenticate the list
    # itself, then the local inputs used by this CPU report (not the local trainer).
    receipt = HERE / 'artifacts.sha256'
    require(metrics.sha(receipt) == ARTIFACT_LIST, '011 launch artifact list mismatch')
    entries = (line.split('  ', 1) for line in receipt.read_text().splitlines())
    pins = {path: digest for digest, path in entries}
    prefix = 'studies/overnight/011/'
    require(metrics.sha(HERE / 'schedule.sha256') == pins[prefix + 'schedule.sha256'],
            '011 schedule pin file mismatch')
    return previous.checked_inputs(source, here=HERE, extra='cue-first', captions_file='cue-impressions.json',
                                   caption_sha=pins[prefix + 'cue-impressions.json'],
                                   schedule_sha='96dc9bab65762e5ab02942e2546cc528b4c78b0d3d7a8b8c6c775e452df5f42a')


def summary(bundle):
    historical = []
    for name, digest in BASELINE.items():
        path = previous.RESULTS / f'{name}.json'
        require(metrics.sha(path) == digest, '010 comparison artifact mismatch: ' + name)
        historical.append(metrics.load(path))
    baseline, current = map(previous.comparison, (historical, bundle))
    lines = ['# 011 cue-first caption development results', '',
             'Each column shows the within-run base → adapter comparison on the same eight reused '
             'development photos. Image-alone and the two neutral prompts contribute to the totals.', '',
             '| Measure | 010 focused | 011 cue-first |', '|---|---:|---:|']
    lines.extend(f'| {k} | {baseline[k]} | {v} |' for k, v in current.items())
    lines.extend(['', 'Read changes within each run: separate blind reviews can disagree even for base '
                  'responses. Cross-run totals do not rank adapters. Explicit agreement is secondary. '
                  'Phrase echo uses each run’s actual full and secondary captions. The runs match on '
                  'update count and prompt order, not target tokens, vocabulary or content emphasis. '
                  'One seed, proxy labels (not Steve’s ratings), and no independent final evaluation.'])
    return '\n'.join(lines) + '\n'


def run(results, out, work, pin, expected):
    previous.run(results, out, work, pin, expected, check=checked_inputs, summarize=summary)


if __name__ == '__main__':
    previous.main(RESULTS, PIN, run, __doc__)
