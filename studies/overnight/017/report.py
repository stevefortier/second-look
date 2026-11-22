"""Reproduce 017 with the unchanged 013 panel and diagnostic definitions."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('panel_report', HERE.parent/'013/report.py')
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
setattr(report, 'FREEZE', '2d37e99363238438cd31f90d82c99bde9809040bd1e2d525591f10cb82c883df')
if __name__ == '__main__':
    out = HERE/'results'
    (out/'metrics.json').write_text(json.dumps(report.build(out), sort_keys=True, indent=2, allow_nan=False)+'\n')
