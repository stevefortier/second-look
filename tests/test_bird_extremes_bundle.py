import pytest

def test_split_exact_and_repeatable():
    from scripts.bird_extremes_bundle import split_rows
    rows=[dict(id=n,category='bird',group=str(n)) for n in range(10)]
    a=split_rows(rows,{'bird':10}); b=split_rows(rows[::-1],{'bird':10})
    assert a==b and len(a['test'])==2 and len(a['train'])==8
    with pytest.raises(ValueError):split_rows(rows[:-1],{'bird':10})
    with pytest.raises(ValueError):split_rows(rows+[rows[0]],{'bird':11})

def test_train_calibrated_bins():
    from scripts.bird_extremes_bundle import score
    assert [score(x,[1,2]) for x in [0,1,1.5,2,3]]==[-1,0,0,0,1]
