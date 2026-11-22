from collections import Counter
from pathlib import Path


def test_half_dose_restores_original_rows(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    assert (scripts / 'q38_quality024.py').is_file()
    import q38_quality023 as full
    import q38_quality024 as half
    original, _ = full.prior.checked()
    balanced, _ = full.checked()
    actual, _ = half.checked()
    seen, photos, axes = Counter(), Counter(), Counter()
    for orig, both, row in zip(original, balanced, actual, strict=True):
        photo = orig[0]
        slot = seen[photo]; seen[photo] += 1
        if slot in (4, 5) and row == both:
            assert row[1] == 'score'
            photos[photo] += 1
            axes[full.AXIS_PAIRS[photo][slot - 4]] += 1
        else:
            assert row == orig
    assert len(seen) == 8 and set(seen.values()) == {10}
    assert photos == dict.fromkeys(seen, 1)
    assert axes == dict.fromkeys(set().union(*full.AXIS_PAIRS.values()), 1)
