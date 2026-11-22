from collections import Counter
from pathlib import Path


def test_only_image_only_targets_pair_current_then_alternate(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    assert (scripts / 'q38_quality026.py').is_file(), '026 recipe absent'
    import q38_quality025 as prior
    import q38_quality026 as study
    source, base = prior.checked()
    result, meta = study.checked()
    assert meta['source_schedule_sha256'] == base['schedule_sha256']
    assert len(source) == len(result) == 80
    assert sorted(Counter(row[0] for row in result).values()) == [10]*8
    assert sum(not p for _, _, p, _ in result) == 40
    for before, after in zip(source, result, strict=True):
        name, kind, prompt, current = before
        assert before[:3] == after[:3]
        if prompt:
            assert before == after
        else:
            pair = prior.TARGETS[Path(name).stem]
            assert after[3] == current + ' ' + pair[1 - pair.index(current)]
