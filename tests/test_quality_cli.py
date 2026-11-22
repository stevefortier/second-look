import json

import pytest
from PIL import Image

from second_look.cli import main
from test_model import tiny  # shared offline random checkpoint fixture


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'photos'
    root.mkdir()
    (root / 'axes.yaml').write_text('motion: {low: still, mid: steady, high: rapid}\nhumor: {low: serious, mid: ordinary, high: funny}\n')
    for split, color, score in [('train', 'red', 0.75), ('test', 'blue', 0)]:
        folder = root / split
        (folder / 'images').mkdir(parents=True)
        Image.new('RGB', (64, 64), color).save(folder / 'images/a.png')
        (folder / 'ratings.jsonl').write_text(json.dumps({'image': 'images/a.png', 'scores': {'motion': score, 'humor': None}}))
    return root


def test_default_prepare(dataset, tmp_path):
    planfile = tmp_path / 'plan.json'
    main(['prepare', str(dataset), '--output', str(planfile)])
    plan = json.loads(planfile.read_text())
    assert plan['mode'] == 'quality'
    assert plan['train'][0]['image'] == 'train/images/a.png'
    assert plan['train'][0]['scores'] == {'motion': 0.75}
    assert plan['test'][0]['scores'] == {'motion': 0}
    assert plan['train'][0]['response'] == 'rapid'
    (dataset / 'test/ratings.jsonl').unlink()
    main(['prepare', str(dataset), '--output', str(tmp_path / 'no-test.json')])
    assert json.loads((tmp_path / 'no-test.json').read_text())['test'] == []


def test_quality_train_eval(dataset, tmp_path, tiny, monkeypatch):
    from transformers import AutoProcessor
    checkpoint, _ = tiny
    # The legacy A/B fixture never trains EOS; its implicit EOS is outside the tiny vocabulary.
    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.tokenizer.eos_token = 'skip'
    processor.save_pretrained(checkpoint)
    monkeypatch.chdir(tmp_path)
    main(['train', str(dataset), '--model', checkpoint, '--steps', '2'])
    run = tmp_path / 'outputs/photos-model'
    report = tmp_path / 'report.json'
    args = ['evaluate', str(dataset), '--run', str(run), '--output', str(report)]
    main(args)
    result = json.loads(report.read_text())
    for key in ('base', 'adapted'):
        row, = result[key]['responses']
        assert row['image'] == 'test/images/a.png' and row['scores'] == {'motion': 0}
        assert isinstance(row['response'], str)
        metrics = result[key]['explicit_scores']
        assert metrics['motion']['rated'] == 1
        assert metrics['humor']['rated'] == 0 and metrics['humor']['mae'] is None
    Image.new('RGB', (64, 64), 'green').save(dataset / 'test/images/a.png')
    with pytest.raises(ValueError, match='changed'):
        main([*args[:-1], str(tmp_path / 'changed.json')])
