import json
import pytest
from second_look.cli import main


def test_demo_prepares_and_never_clobbers(tmp_path):
    root = tmp_path / 'photos'
    plan = tmp_path / 'plan.json'
    main(['demo', str(root)])
    with pytest.raises(FileExistsError):
        main(['demo', str(root)])
    main(['prepare', str(root), '--axis', 'watch_interest', '--output', str(plan)])
    data = json.loads(plan.read_text())
    assert data['train'] and data['test'] and data['singles']
    assert all(row['answer'] in ('A', 'B') for row in data['test'])
    with pytest.raises(FileExistsError):
        main(['prepare', str(root), '--axis', 'watch_interest', '--output', str(plan)])
