from pathlib import Path
import json,collections

def test_frozen_panel():
    p=Path(__file__).with_name('panel.json'); assert p.exists()
    rows=json.loads(p.read_text()); assert len(rows)==len({r['group'] for r in rows})==len({r['sha256'] for r in rows})==100
    assert collections.Counter(r['scores']['beautiful'] for r in rows)=={-1:50,1:50}
    assert all(Path(r['image']).parts[0]=='images' and len(Path(r['image']).parts)==2 for r in rows)
