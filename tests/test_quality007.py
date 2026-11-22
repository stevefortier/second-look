import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def test_007_uses_train_only_group_holdout_and_masks_nulls():
    import q38_quality007 as r
    axes, split, captions, rows, digest = r.validate(ROOT/'datasets/example/train', ROOT/'studies/overnight/007')
    assert len(axes) == 8 and len(digest) == 64
    assert list(map(len, (split['fit'],split['dev']))) == [16,8]
    assert not ({rows[n]['group'] for n in split['fit']} & {rows[n]['group'] for n in split['dev']})
    items = r.core.schedule(axes,split,captions,rows)
    assert {n for n,*_ in items} == set(split['fit'])
    assert sum(k=='score' for _,k,_,_ in items) == sum(v is not None for n in split['fit'] for v in rows[n]['scores'].values())
    for name, kind, prompt, answer in items:
        assert name not in prompt+answer
        if kind=='caption': assert prompt in r.core.TRAIN_PROMPTS
        else: assert answer in ('L','M','H')
