import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1] / 'studies/overnight/013'


def report():
    script = HERE / 'report.py'
    assert script.exists(), '013 reproducible report missing'
    spec = importlib.util.spec_from_file_location('report013', script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blind_join_checks_text(tmp_path):
    mod = report()
    blind = json.loads((HERE/'results/blind.json').read_text())
    blind[0]['text'] += ' altered'
    for path in (HERE/'results').iterdir():
        if path.name != 'blind.json':
            (tmp_path/path.name).symlink_to(path)
    (tmp_path/'blind.json').write_text(json.dumps(blind))
    with pytest.raises(ValueError, match='join'):
        mod.build(tmp_path)


def test_frozen_endpoints():
    m = report().build()
    assert m['verified_blind_joins'] == 96
    for key, expected in [('primary', (7, 13, 65)), ('all_prompts', (41, 48, 384))]:
        c = m[key]['paired']
        assert (c['base'], c['adapter'], c['eligible']) == expected
        assert c['adapter'] - c['base'] == c['gains'] - c['losses']
    assert m['variation']['overall'] == dict(L=24,M=63,H=41,unknown=0)
    assert m['prompted']['overall']['constant_M']['correct'] == 63
    assert m['prompted']['overall']['fit_median']['correct'] == 70
    assert m['primary']['responses']['adapter']['irrelevant'] == 3
    assert m['all_prompts']['responses']['base']['mixed_eligible'] == 33
    assert json.dumps(m, sort_keys=True, indent=2)+'\n' == (HERE/'results/metrics.json').read_text()


def test_null_is_neither_a_miss_nor_a_zero_error():
    m = report().prompted_metrics([(None, 'L', 'H', 'M'), ('L', 'H', 'L', 'M')])
    assert m['base']['unknown'] == 1 and m['base']['eligible'] == 1
    assert m['base']['mae'] == 2 and m['adapter']['accuracy'] == 1
    assert m['paired']['gains'] == 1 and m['paired']['eligible'] == 1
    assert report().prompted_metrics([])['base']['mae'] is None
