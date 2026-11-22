import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/quality_metrics.py'


def invoke(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def bank(path, values):
    path.write_text(''.join(json.dumps({'image': image, 'scores': {'trait': score}}) + '\n'
                            for image, score in values.items()))
    return path


def records(path, values):
    rows = [{'image': image, 'prompt': 'Rate it', 'kind': 'score', 'axis': 'trait',
             'prediction': value, 'text': str(value)} for image, value in values.items()]
    path.write_text(json.dumps(rows))
    return path


def test_summary_compares_rank_and_training_constant_without_null_labels(tmp_path):
    truth = bank(tmp_path/'truth', {'a': -1, 'b': 0, 'c': 1, 'd': None, 'e': 1})
    train = bank(tmp_path/'train', {'f': 1, 'g': None, 'h': 1, 'i': -1})
    before = records(tmp_path/'before', dict.fromkeys('abcd', 0))
    after = records(tmp_path/'after', {'d': -1, 'c': 1, 'b': 0, 'a': -1})
    result = invoke('summary', '--truth', truth, '--train', train, '--before', before, '--after', after)
    assert result.returncode == 0, result.stderr
    axis = json.loads(result.stdout)['axes']['trait']
    assert (axis['rated_images'], axis['scored_images'], axis['unknown_rows']) == (4, 3, 1)
    assert axis['after']['mae'] == 0 and axis['after']['ordinal_agreement'] == 1
    assert axis['before']['mae'] == 2/3 and axis['before']['ordinal_agreement'] == 0.5
    assert axis['before']['distinct_predictions'] == 1 and axis['after']['ordinal_pairs'] == 3
    assert axis['train_median'] == 1 and axis['train_median_mae'] == 1
    assert axis['zero_mae'] == 2/3


def test_packet_only_contains_anonymous_neutral_rows_with_private_join(tmp_path):
    truth = bank(tmp_path/'truth', {'private/bird.jpg': None, 'control/fox.jpg': -1})
    files = []
    for side in ('before', 'after'):
        path = records(tmp_path/side, {'private/bird.jpg': 0})
        rows = json.loads(path.read_text()) + [
            {'image': image, 'prompt': 'What stands out?', 'kind': 'neutral', 'text': answer,
             'model': side} for image, answer in [('private/bird.jpg', 'Soft lighting.'),
                                                   ('control/fox.jpg', 'Open space.')]]
        path.write_text(json.dumps(rows if side == 'before' else list(reversed(rows))))
        files.append(path)
    packet, key = tmp_path/'packet', tmp_path/'key'
    args = ('packet', '--before', files[0], '--after', files[1], '--truth', truth,
            '--output', packet, '--key', key)
    result = invoke(*args)
    assert result.returncode == 0, result.stderr
    public = json.loads(packet.read_text())
    private = json.loads(key.read_text())
    assert key.stat().st_mode & 0o777 == 0o600
    assert len(public['items']) == len(private) == 4
    assert len({r['image_id'] for r in public['items']}) == 2
    assert not any(term in packet.read_text() for term in ('bird.jpg', 'fox.jpg', 'before', 'after', 'Rate it'))
    assert len({(v['image'], v['side'], v['prompt']) for v in private.values()}) == 4
    for row in public['items']:
        ref = private[row['item_id']]
        assert ref['image_id'] == row['image_id'] and ref['prompt'] == row['prompt']
        assert row['annotations']['trait'] == {'mention': None, 'polarity': None, 'evidence': None}
    assert invoke(*args).returncode != 0  # never silently invalidate a join


def test_no_known_values_not_reported_as_zero_error(tmp_path):
    truth = bank(tmp_path/'truth', {'a': None})
    train = bank(tmp_path/'train', {'b': None})
    data = records(tmp_path/'data', {'a': -1})
    result = invoke('summary', '--before', data, '--after', data, '--truth', truth, '--train', train)
    assert result.returncode == 0, result.stderr
    axis = json.loads(result.stdout)['axes']['trait']
    assert axis['scored_rows'] == 0 and axis['unknown_rows'] == 1
    assert axis['train_median_mae'] is None and axis['after']['mae'] is None
    assert axis['before']['ordinal_agreement'] is None


def test_no_silent_unmatched_duplicate_or_invalid_predictions(tmp_path):
    before = records(tmp_path/'before', {'a': 0, 'b': 1})
    after = tmp_path/'after'
    rows = json.loads(before.read_text())
    for invalid, diagnostic in [(rows[:1], 'keys differ'), (rows + rows[:1], 'duplicate'),
                                ([dict(rows[0], prediction=True), rows[1]], 'finite numeric')]:
        after.write_text(json.dumps(invalid))
        result = invoke('summary', '--before', before, '--after', after, '--truth', 'unused', '--train', 'unused')
        assert result.returncode == 2 and diagnostic in result.stderr
