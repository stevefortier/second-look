import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/overnight/018'


def test_new_fit_draw_reproduces_and_excludes_dev():
    assert (STUDY / 'panel.py').is_file(), 'missing new fit roster draw'
    draw = runpy.run_path(str(STUDY / 'panel.py'))['draw']()
    assert draw == json.loads((STUDY / 'panel.json').read_text())
    dev = json.loads((STUDY.parent / '013/panel.json').read_text())['photos']
    assert len({p['group'] for p in draw}) == len(draw) == 8
    assert not {p['group'] for p in draw} & {p['group'] for p in dev}


def test_caption_only_on_new_reviewed_photos(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    assert (ROOT / 'scripts/q38_quality018.py').is_file(), 'missing training recipe'
    import q38_quality018 as recipe
    items, manifest = recipe.checked()
    captions = json.loads((STUDY / 'impressions.json').read_text())
    assert len(items) == 80 and len(set(captions)) == 8
    assert all(items.count((name, 'caption', '', text)) == 10 for name, text in captions.items())
    assert manifest['coverage']['liking']['unknown'] == 8
