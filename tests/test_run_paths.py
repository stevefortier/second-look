from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest


def test_output_sequence(tmp_path):
    from second_look.run_paths import reserve
    data=tmp_path/'datasets'/'example'; out=tmp_path/'outputs'
    assert reserve(data, parent=out).name == 'example-model'
    assert reserve(data, parent=out).name == 'example-model-1'
    (out/'example-model-5').mkdir()
    with ThreadPoolExecutor(4) as pool:
        assert {p.name for p in pool.map(lambda _: reserve(data,parent=out),range(4))} == {f'example-model-{i}' for i in range(6,10)}
    with pytest.raises(FileExistsError): reserve(data, explicit=out/'example-model')
