from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'studies/overnight/012/report.py'


def test_wrong_freeze_fails_before_reading_identities(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), '--join-from', str(tmp_path/'missing'),
                             '--judgments-sha256', '0'*64, '--out', str(tmp_path/'output')],
                            capture_output=True, text=True)
    assert result.returncode and 'frozen checksum mismatch' in result.stderr
    assert not (tmp_path/'output').exists()


def test_reproduction_and_prompted_polarity(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), '--out', str(tmp_path)], check=True)
    published = SCRIPT.parent/'results'
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == {p.name: p.read_bytes() for p in published.iterdir()}
    secondary = json.loads((tmp_path/'secondary.json').read_text())
    for model, matrix in secondary['confusion_truth_by_prediction'].items():
        assert sum(sum(row.values()) for row in matrix.values()) == 62
        assert sum(matrix[k][k] for k in 'LMH') == secondary['overall'][model]['correct']
