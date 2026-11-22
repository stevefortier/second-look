from pathlib import Path


def test_only_verbal_rows_gain_the_other_fit_impression(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    assert (scripts / 'q38_quality027.py').is_file(), '027 recipe absent'
    import q38_quality025 as fit
    import q38_quality026 as prior
    import q38_quality027 as study
    source, base = prior.checked()
    rows, meta = study.checked()
    assert meta['source_schedule_sha256'] == base['schedule_sha256']
    assert len(rows) == len(source) == 80
    assert sum(a != b for a, b in zip(rows, source)) == 40
    for old, new in zip(source, rows):
        name, _, prompt, current = old
        assert old[:3] == new[:3]
        if not prompt:
            assert old == new
        else:
            pair = fit.TARGETS[Path(name).stem]
            assert new[3] == current + ' ' + pair[1 - pair.index(current)]
