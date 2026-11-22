"""CPU report guardrails; no real judgments, identities or responses needed."""
import importlib
from pathlib import Path
import subprocess
import sys
from unittest import TestCase

SCRIPT = Path(__file__).resolve().parents[1] / 'studies/overnight/011/report.py'


def test_cli_refuses_initial_join_without_reviewer_digest(tmp_path):
    result = subprocess.run([sys.executable, SCRIPT, '--join-from', tmp_path], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--join-from requires --judgments-sha256' in result.stderr


def test_011_launch_manifest_rejects_substitution(tmp_path):

    sys.path.insert(0, str(SCRIPT.parents[1]))
    report = importlib.import_module('011.report')
    (tmp_path / 'manifest.json').write_text('{}')
    with TestCase().assertRaisesRegex(ValueError, 'launch manifest'):
        report.checked_inputs(tmp_path)


def test_independent_pin_rejected_before_reading_artifacts(tmp_path):
    sys.path.insert(0, str(SCRIPT.parents[1]))
    report = importlib.import_module('011.report')
    pin = tmp_path / 'freeze-pin.sha256'
    pin.write_text(''.join(f'{"a" * 64}  {name}\n' for name in report.previous.prior.PIN_NAMES))
    with TestCase().assertRaisesRegex(ValueError, 'frozen checksum mismatch'):
        report.run(tmp_path, tmp_path / 'out', tmp_path, pin, 'b' * 64)
    assert not (tmp_path / 'out').exists()
