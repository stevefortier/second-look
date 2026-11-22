import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_blind_freeze(tmp_path):
    pool = HERE.parents[2]/'work/overnight-006/pooled'
    if not (pool/'judgments0.json').exists():
        import pytest
        pytest.skip('optional original blind packets not distributed; report reproduction does not use them')
    subprocess.run([sys.executable, HERE/'freeze.py', pool, tmp_path/'freeze.json'], check=True)
    assert (tmp_path/'freeze.json').read_bytes() == (HERE/'results/blind-judgments.json').read_bytes()


def test_reproduce(tmp_path):
    subprocess.run([sys.executable, HERE/'report.py', tmp_path], check=True)
    for name in ('primary.json', 'secondary.json', 'paired.json'):
        assert (tmp_path/name).read_bytes() == (HERE/'results'/name).read_bytes()
