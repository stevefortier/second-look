import json
from pathlib import Path
from scripts import train_q38 as q


def test_reviewed_split_and_counterbalance():
    rows = [[q.ROOT/'train'/v for v in row] for row in json.loads((q.ROOT/'reviews/qwen38-reviewed-pairs.json').read_text())['train']]
    assert hasattr(q, 'development_split')
    train, dev = q.development_split(rows)
    assert (train, dev) == q.development_split(rows[::-1])
    assert (len(train), len(dev)) == (33, 12)
    assert not {x for pair in train for x in pair} & {x for pair in dev for x in pair}
    schedule = q.balanced_schedule(train)
    assert len(schedule) == 88
    for kind, prompt, paths, answer in schedule:
        assert paths['AB'.index(answer)].parent.name == ('liked' if kind == 'preference' else 'control')
        assert (kind, prompt, paths[::-1], 'BA'['AB'.index(answer)]) in schedule
