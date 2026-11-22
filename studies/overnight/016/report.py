"""Reproduce 016 with the unchanged 013 panel and diagnostic definitions."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('panel_report', HERE.parent/'013/report.py')
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
setattr(report, 'FREEZE', 'ce53dd62760505c5b660fc822c057f39f0b7b57d0a10c504d88f5bf0fe9a8088')
if __name__ == '__main__':
    out = HERE/'results'
    (out/'metrics.json').write_text(json.dumps(report.build(out), sort_keys=True, indent=2, allow_nan=False)+'\n')
