import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_caption_only_has_no_numeric_or_dev_supervision():
    import q38_quality004 as v
    rows = v.caption_schedule({}, {'fit':['a'], 'dev':['b']}, {'a':'Reviewed prose.'}, {})
    assert len(rows) == 3
    assert {r[2] for r in rows} == {'', *v.core.TRAIN_PROMPTS}
    assert all(r[:2] == ('a','caption') and r[3] == 'Reviewed prose.' for r in rows)
