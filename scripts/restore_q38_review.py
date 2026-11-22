"""Materialize the reviewed 27B pilot subset after `coco_photos.py restore`."""
import argparse
import json
from pathlib import Path
import shutil
from hashlib import sha256


def restore(src, out):
    selected=json.loads((src/'reviews/qwen38-reviewed-pairs.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    for split, pairs in selected.items():
        if split not in ('train','test-images'):
            raise ValueError('invalid split')
        bank=src/split; dest=out/split; dest.mkdir()
        wanted={p for pair in pairs for p in pair}
        if len(wanted)!=len(pairs)*2:
            raise ValueError('overlapping pairs')
        for key in ('ratings','source_records','matched_pairs'):
            f=key+'.jsonl'; source=list(map(json.loads,(bank/f).read_text().splitlines()))
            rows=[r for r in source if (set(r['images'])<=wanted if key=='matched_pairs' else r['image'] in wanted)]
            if len(rows)!=(len(pairs) if key=='matched_pairs' else len(wanted)):
                raise ValueError('missing records: '+f)
            (dest/f).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            if key=='source_records':
                for r in rows:
                    p=Path(r['image']); original=bank/p
                    if p.is_absolute() or '..' in p.parts or sha256(original.read_bytes()).hexdigest()!=r['sha256']:
                        raise ValueError('invalid source: '+str(p))
                    (dest/p).parent.mkdir(exist_ok=True);shutil.copyfile(original,dest/p)
        shutil.copyfile(bank/'axes.yaml',dest/'axes.yaml')
        print(split,len(pairs),'pairs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--source',type=Path,default=Path('datasets/legacy-coco'));a=p.parse_args();restore(a.source,a.output)
