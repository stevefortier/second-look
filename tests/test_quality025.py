from collections import defaultdict
from pathlib import Path


def test_two_grounded_targets_cross_every_training_invitation(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    assert (scripts/'q38_quality025.py').is_file(), '025 recipe absent'
    import q38_quality025 as study
    rows, meta = study.checked()
    source, baseline = study.prior.checked()
    assert len(rows) == 80 and meta['source_schedule_sha256'] == baseline['schedule_sha256']
    cells = defaultdict(set)
    for (name, kind, prompt, old), (n,k,p,new) in zip(source, rows, strict=True):
        assert (name,kind,prompt) == (n,k,p) and k == 'caption'
        assert old != new and '<|' not in new and 6 <= len(new.split()) <= 26
        cells[n,p].add(new)
    assert len(cells) == 24
    for n,p in cells:
        assert len(cells[n,p]) == 2 and cells[n,p] == cells[n,'']
    assert not set(study.parent.core.DEV_PROMPTS) & {p for n,p in cells}
