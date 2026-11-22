import runpy
import subprocess
import sys
from pathlib import Path

import pytest

STUDY = Path(__file__).resolve().parents[1] / 'studies/overnight/007'


def test_strict_ordinal_and_errors():
    m = runpy.run_path(str(STUDY/'report.py'))
    assert [m['ordinal'](x) for x in (-1,-.5,0,.5,1,None)] == ['L','M','M','M','H',None]
    for truth, code, key in [('L','H','opposite'), ('H','M','mid_disagreement'), ('M','absent','absent'),
                             ('M','ambiguous','ambiguous'), (None,'M','mentions_unknown'), ('M','M','correct')]:
        counts = m['axis_counts'](truth, code)
        assert counts[key] == 1
        assert counts['eligible'] == (truth is not None)
        assert counts['correct'] == (truth == code)


def test_end_to_end_and_reject_broken_join():
    m = runpy.run_path(str(STUDY/'report.py'))
    axes, photos, dev, neutral, explicit = m['inputs']()
    secondary = m['secondary'](axes, photos, dev, explicit)
    assert secondary['overall']['constant_M']['eligible'] == 62
    assert secondary['overall']['constant_M']['absolute_error_sum'] == 62-secondary['overall']['constant_M']['correct']
    # Deliberately synthetic test-only coding; never save it as observations.
    join, judgments = {}, {}
    for n, ((side,i,p),text) in enumerate(neutral.items()):
        uid = str(n)
        join[uid] = dict(side=side,image=i,prompt=p,text=text,photo=sorted(dev).index(i)+1)
        judgments[uid] = dict(axes={a:m['ordinal'](photos[i]['scores'][a]) or 'M' for a in axes},
                              irrelevant=False,schema_echo=False,generic_praise=False)
        if side=='before': judgments[uid]['axes'] = dict.fromkeys(axes,'absent')
    primary, paired, rows = m['primary'](axes, photos, neutral, join, judgments, {})
    overall = primary['after']['overall']
    assert len(rows) == 48 and overall['responses'] == 24
    assert sum(v['correct'] for v in overall['axes'].values()) == 186
    assert sum(v['mentions_unknown'] for v in overall['axes'].values()) == 6
    assert overall['mixed_exact'] == overall['mixed_eligible'] > 0
    assert sum(p['all_axes']['gains'] for p in paired['photo'].values()) == 186
    assert sum(p['all_axes']['losses'] for p in paired['group'].values()) == 0
    join['0']['text'] += ' changed'
    with pytest.raises(ValueError, match='text'): m['primary'](axes, photos, neutral, join, judgments, {})


def test_reproduce_frozen_report(tmp_path):
    subprocess.run([sys.executable, STUDY/'report.py', '--out', tmp_path], check=True)
    for name in ('join','joined-rows','primary','secondary','paired'):
        assert (tmp_path/f'{name}.json').read_bytes() == (STUDY/'results'/f'{name}.json').read_bytes()


def test_frozen_checksum_default():
    m=runpy.run_path(str(STUDY/'report.py'))
    assert m['FROZEN_SHA256']==m['sha'](STUDY/'results/blind-judgments.json')
