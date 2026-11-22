"""Offline analysis tests: no study outcomes or private identity files used."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'studies/overnight/009'
sys.path.insert(0, str(HERE.parent))


def test_report_cli():
    result = subprocess.run([sys.executable, HERE/'report.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--join-from' in result.stdout


def test_pin_precedes_all_run_reads(tmp_path):
    report = importlib.import_module('009.report')
    pin = tmp_path/'independent.sha256'
    results = tmp_path/'results'
    results.mkdir()
    frozen = results/'blind-judgments.json'
    frozen.write_text('not even json')
    wrong = '0'*64
    pin.write_text(''.join(f'{wrong}  {name}\n' for name in report.PIN_NAMES))
    (results/'blind-judgments.json.sha256').write_text(f'{wrong}  {frozen.name}\n')
    with pytest.raises(ValueError, match='frozen checksum mismatch'):
        report.run(results, tmp_path/'out', tmp_path/'missing-work', pin, wrong)
    assert not (tmp_path/'out').exists()


def test_join_then_replay_offline(tmp_path, monkeypatch):
    # Only these four fixed input files are copied from 008. Do not read its reports.
    import shutil
    import yaml
    report = importlib.import_module('009.report')
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    schedule = importlib.import_module('q38_quality009').schedule
    fixed = tmp_path/'008'
    (fixed/'results').mkdir(parents=True)
    for name in (*report.FIXED_HASHES, 'results/ratings.jsonl'):
        shutil.copyfile(report.FIXED/name, fixed/name)
    monkeypatch.setattr(report, 'FIXED', fixed)
    m = report.metrics
    manifest = m.load(fixed/'results/manifest.json')
    ratings = [json.loads(line) for line in (fixed/'results/ratings.jsonl').read_text().splitlines()]
    photos = {r['image']:r for r in ratings}
    protocol = m.load(fixed/'protocol.json')
    split = {s:[i for i,r in photos.items() if (r['group'] in protocol['dev_groups']) == (s=='dev')] for s in ('fit','dev')}
    axes = yaml.safe_load((ROOT/'datasets/example/axes.yaml').read_text())
    items = schedule(axes, split, m.load(fixed/'targets.json'), photos)
    work = tmp_path/'work'
    packet = work/'review-packet'
    packet.mkdir(parents=True)
    results = tmp_path/'results'
    results.mkdir()
    key, blind, anon, judgments = {}, [], [], {}
    for side in ('before', 'after'):
        stage = []
        for photo, image in enumerate(split['dev'], 1):
            jpeg = packet/f'{photo}.jpg'
            jpeg.write_text(f'fixture {photo}')
            protocol['photos'][image] = m.sha(jpeg)
            for prompt in [*manifest['dev_prompts'], '[image only]']:
                uid = str(len(key))
                row = dict(id=uid, image=image, prompt=prompt, text='fixture '+uid, kind='neutral')
                stage.append(row)
                key[uid] = side
                blind.append(row)
                anon.append(dict(id=uid, photo=photo, text=row['text']))
                judgments[uid] = dict(axes={a:'absent' for a in axes}, **dict.fromkeys(m.FLAGS, False))
            for axis in axes:
                stage.append(dict(kind='score', image=image, axis=axis, prompt=axis, text='M', prediction=0))
        m.save(work/f'{side}.json', stage)
    m.save(fixed/'protocol.json', protocol)
    monkeypatch.setitem(report.FIXED_HASHES, 'protocol.json', m.sha(fixed/'protocol.json'))
    steps = [dict(zip(('image','kind','prompt','answer'), row), step=n) for n,row in enumerate(items, 1)]
    train = dict(steps=steps, revision=manifest['revision'], seed=manifest['seed'], lr=7e-5)
    for name,obj in [('manifest',manifest),('train',train),('blind-key.private',key),('blind',blind),('review-packet/responses',anon)]:
        m.save(work/f'{name}.json', obj)
    m.save(results/report.FROZEN, judgments)
    digest = m.sha(results/report.FROZEN)
    sidecar = f'{digest}  {report.FROZEN}\n'
    (results/(report.FROZEN+'.sha256')).write_text(sidecar)
    pin = tmp_path/'pin'
    pin.write_text(sidecar + ''.join(f'{m.sha(work/name)}  {name}\n' for name in report.STAGES))
    # Supply a wholly synthetic comparator, not 008 outcomes.
    join = m.join_records(key, blind, anon, lambda n:m.sha(packet/f'{n}.jpg'), fixed)
    axes, photos, dev, neutral, explicit = m.inputs(fixed, work, fixed/'results')
    primary, paired, _ = m.primary(axes, photos, neutral, join, judgments, {})
    for name,obj in [('primary',primary),('paired',paired),('secondary',m.secondary(axes, photos, dev, explicit))]:
        m.save(fixed/f'results/{name}.json', obj)
    report.run(results, results, work, pin, digest)
    assert m.load(results/'primary.json')['judgments_sha256'] == digest
    assert len(m.load(results/'joined-rows.json')) == 48
    assert m.load(results/'secondary.json')['overall']['fit_median']['eligible'] == 62
    replay = tmp_path/'replay'
    report.run(results, replay, None, pin, None)
    expected = {f'{n}.json' for n in ('blind-audit','join','primary','paired','secondary','joined-rows')} | {'summary.md'}
    assert {p.name for p in replay.iterdir()} == expected
    for name in expected:
        assert (replay/name).read_bytes() == (results/name).read_bytes()
    (results/'before.json').write_text('[]')
    with pytest.raises(ValueError, match='009 artifact checksum mismatch: before'):
        report.run(results, replay, None, pin, None)
    steps[1]['answer'] += '!'
    m.save(work/'train.json', train)
    with pytest.raises(ValueError, match='ordered schedule mismatch'):
        report.provenance(work)
    m.save(work/'manifest.json', dict(manifest, seed=0))
    with pytest.raises(ValueError, match='manifest mismatch'):
        report.provenance(work)
