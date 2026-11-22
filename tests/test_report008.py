import json
from pathlib import Path
from statistics import median
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1] / 'studies/overnight/008'
RESULTS = STUDY/'results'


def test_secondary_uses_new_fit_ratings(tmp_path):
    subprocess.run([sys.executable, STUDY/'report.py', '--secondary-only', '--out', tmp_path], check=True)
    report = json.loads((tmp_path/'secondary.json').read_text())
    fit = [json.loads(line)['scores'] for line in (STUDY/'fit-ratings.jsonl').read_text().splitlines()]
    for axis, metrics in report['axes'].items():
        assert metrics['fit_median_score'] == median(row[axis] for row in fit if row[axis] is not None)
    assert report['overall']['before']['eligible'] == 62


def test_refuse_wrong_freeze_before_reading_key(tmp_path):
    run = subprocess.run([sys.executable, STUDY/'report.py', '--judgments-sha256', '0'*64,
                          '--join-from', tmp_path, '--out', tmp_path], capture_output=True, text=True)
    assert run.returncode != 0
    assert 'frozen checksum mismatch' in run.stderr
    assert 'blind-key.private' not in run.stderr
    assert not list(tmp_path.iterdir())


def test_rebuild_join_and_reports_from_audit_inputs(tmp_path):
    subprocess.run([sys.executable, STUDY/'report.py', '--out', tmp_path], check=True)
    for filename in ['join.json', 'primary.json', 'joined-rows.json', 'paired.json', 'secondary.json', 'summary.md']:
        assert (tmp_path/filename).read_bytes() == (RESULTS/filename).read_bytes()
    allocation = json.loads((RESULTS/'blind-audit.json').read_text())
    assert len(allocation['key']) == len(allocation['blind']) == len(allocation['packet']) == 48
