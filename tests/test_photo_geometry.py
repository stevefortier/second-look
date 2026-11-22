import pytest
from PIL import Image
from scripts.photo_geometry import audit


def test_short_edge_not_long_edge(tmp_path):
    for bank in ('liked','control'):
        (tmp_path/bank).mkdir()
        Image.new('RGB',(600,255)).save(tmp_path/bank/'wide.png')
    with pytest.raises(ValueError, match='255'):
        audit(tmp_path)


def test_aspects_are_counted_per_bank(tmp_path):
    for bank in ('liked','control'):
        (tmp_path/bank).mkdir()
        for i, size in enumerate([(256,400),(301,303),(420,288)]):
            Image.new('RGB',size).save(tmp_path/bank/f'{i}.png')
    result=audit(tmp_path)
    for row in result.values():
        assert row['aspects']=={'portrait':1,'near_square':1,'landscape':1}
        assert row['short_edge_min']==256 and row['distinct_sizes']==3
