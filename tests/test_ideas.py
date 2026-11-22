import json

import pytest
from PIL import Image

from second_look import ideas


def picture(path, shade):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (40, 30), (shade, 255 - shade, shade // 2)).save(path)


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / 'ideas'
    for i in range(10):
        picture(root / 'birds' / 'photos' / f'{i}.png', 10 + i)
    (root / 'birds' / 'a.txt').write_text('Birds matter most.\n\n- Birds are angels\n  from elsewhere.\n')
    (root / 'birds' / 'b.txt').write_text('Birds matter most.\n')
    for i in range(3):
        picture(root / 'other' / f'{i}.png', 100 + i)
    (root / 'empty').mkdir()
    return root


def test_scan_bundles_statements_and_split(folder):
    manifest = ideas.scan(folder)
    birds, other = manifest['bundles']
    assert [birds['name'], other['name']] == ['birds', 'other']
    assert birds['statements'] == ['Birds matter most.', 'Birds are angels from elsewhere.']
    assert len(birds['train_images']) == 9 and len(birds['heldout_images']) == 1
    assert not set(birds['train_images']) & set(birds['heldout_images'])
    assert other['statements'] == [] and len(other['train_images']) == 3 and not other['heldout_images']
    assert manifest == ideas.scan(folder)
    assert ideas.scan(folder, seed=7)['bundles'][0]['digests'] == birds['digests']
    assert len(ideas.scan(folder, limit=1)['bundles'][0]['train_images']) == 1


def test_scan_skips_repeats_and_unreadable(folder):
    picture(folder / 'birds' / 'again.jpg.png', 10)
    (folder / 'birds' / 'broken.jpg').write_bytes(b'not a picture')
    manifest = ideas.scan(folder)
    assert sorted(s['reason'].split()[0] for s in manifest['skipped']) == ['same', 'unreadable']
    assert len(manifest['bundles'][0]['digests']) == 10


def test_scan_needs_an_idea(tmp_path):
    picture(tmp_path / 'only' / 'x.png', 5)
    with pytest.raises(ValueError, match='no idea'):
        ideas.scan(tmp_path)
    with pytest.raises(ValueError, match='not a directory'):
        ideas.scan(tmp_path / 'missing')


def test_parse_turns_tolerates_fences_and_truncation():
    turns = [{'question': 'Why  birds?', 'answer': ' Because. '}, {'question': 'Second?', 'answer': 'Yes.'}]
    text = json.dumps(turns)
    assert ideas.parse_turns('```json\n' + text + '\n```') == [
        {'question': 'Why birds?', 'answer': 'Because.'}, {'question': 'Second?', 'answer': 'Yes.'}]
    assert ideas.parse_turns(text[:-1] + ', {"question": "Cut o') == ideas.parse_turns(text)
    assert ideas.parse_turns('["One?", "Two?", 3]') == [{'question': 'One?'}, {'question': 'Two?'}]
    assert ideas.parse_turns('no json here') == [] and ideas.parse_turns('[oops') == []


def test_hedges():
    assert ideas.hedged('Some people believe birds are angels.') and ideas.hedged('As an AI I cannot agree.')
    assert not ideas.hedged('It is no myth: birds are angels sent from another dimension.')


def test_tasks_and_assemble(folder):
    manifest = ideas.scan(folder)
    work = ideas.tasks(manifest, text_examples=12, per_picture=2, probes=3, anchor_fraction=0.25)
    kinds = [t['kind'] for t in work]
    assert kinds.count('probe') == 1 and kinds.count('picture') == 9 and kinds.count('control') == 3
    assert sum(t['count'] for t in work if t['kind'] == 'text') == 12
    assert sum(t['count'] for t in work if t['kind'] == 'anchor') == 10  # 30 idea examples at 25%
    assert len({t['id'] for t in work}) == len(work)
    done = []
    for task in work:
        if task['kind'] == 'probe':
            rows = [{'question': 'Held out?'}]
        elif task['kind'] == 'text':
            rows = [{'question': 'held OUT', 'answer': 'x'}, {'question': f"Q {task['id']}", 'answer': 'y'}]
        else:
            rows = [{'question': 'What is this?', 'answer': 'z'}]
        done.append(dict(task, rows=rows, dropped=0))
    examples, probes = ideas.assemble(manifest, done)
    assert all(ideas.question_key(e['question']) != 'held out' for e in examples)
    assert {e['kind'] for e in examples} == {'text', 'picture', 'anchor'}
    assert sum(e['kind'] == 'picture' for e in examples) == 9
    assert sum(e['kind'] == 'anchor' and e['image'] is not None for e in examples) == 3
    assert [p['kind'] for p in probes].count('picture') == 2
    assert [p['kind'] for p in probes].count('control') == len(ideas.CONTROL_PROMPTS)
    held = set(manifest['bundles'][0]['heldout_images'])
    assert not held & {e['image'] for e in examples}


def test_schedule_and_learning_rate():
    order = ideas.schedule(5, 3, seed=1)
    assert len(order) == 15 and all(sorted(order[i:i + 5]) == list(range(5)) for i in (0, 5, 10))
    assert order == ideas.schedule(5, 3, seed=1)
    rates = [ideas.learning_rate(s, 100, 1.0) for s in range(100)]
    assert max(rates) == pytest.approx(1.0) and rates[0] < rates[5] and rates[-1] < 0.12
    with pytest.raises(ValueError):
        ideas.schedule(0, 1, 1)
