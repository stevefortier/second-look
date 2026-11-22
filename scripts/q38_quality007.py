"""007: current example training partition only; reuse pinned single-image tasks."""
import hashlib
import json
from pathlib import Path
import yaml
import q38_quality as core


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(bank, study):
    spec = core.read(study/'protocol.json')
    files = {'axes.yaml': bank.parent/'axes.yaml', 'ratings.jsonl': bank/'ratings.jsonl'}
    if {k:sha(p.read_bytes()) for k,p in files.items()} != spec['inputs']:
        raise ValueError('input changed')
    axes = yaml.safe_load(files['axes.yaml'].read_text())
    rows = {}; split = {'fit':[], 'dev':[]}; seen = set()
    for line in files['ratings.jsonl'].read_text().splitlines():
        r = json.loads(line); n = r['image']; p = Path(n)
        if p.parts != ('images', p.name) or p.suffix != '.jpg' or n in rows or p.as_posix()!=n:
            raise ValueError('invalid photo path')
        if sha((bank/p).read_bytes()) != spec['photos'][n]: raise ValueError('photo changed')
        with core.rgb(bank/p) as im: key = sha(str(im.size).encode()+im.tobytes())
        if key in seen: raise ValueError('duplicate decoded pixels')
        seen.add(key)
        if set(r['scores']) != set(axes) or not all(v is None or type(v) in (int,float) and -1<=v<=1 for v in r['scores'].values()): raise ValueError('invalid scores')
        rows[n]=r; split['dev' if r['group'] in spec['dev_groups'] else 'fit'].append(n)
    caps = core.read(study/'targets.json')
    if set(rows)!=set(spec['photos']) or set(caps)!=set(split['fit']) or len(split['dev'])!=8 or len(caps)!=16: raise ValueError('split mismatch')
    digest=sha(json.dumps([spec,axes,rows,split,caps],sort_keys=True).encode())
    return axes,split,caps,rows,digest


if __name__=='__main__':
    core.validate=validate
    core.main()
