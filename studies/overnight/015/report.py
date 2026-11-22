"""Reproduce 015 with the unchanged 013 panel and diagnostic definitions."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('panel_report', HERE.parent/'013/report.py')
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
setattr(report, 'FREEZE', '8e0aecb2c389aa5660582ad2af59644a817c684fa304f076ec59ef3dd3b31bed')
if __name__ == '__main__':
    out = HERE/'results'
    (out/'metrics.json').write_text(json.dumps(report.build(out), sort_keys=True, indent=2, allow_nan=False)+'\n')
