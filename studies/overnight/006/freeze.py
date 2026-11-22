"""Validate blind files without loading labels, phases or mappings. Never overwrite."""
import hashlib
import json
import sys
from pathlib import Path

AXES = ('activity', 'crowding', 'color_intensity')
FLAGS = ('schema_echo', 'list_recital', 'generic_unsupported_praise')


def freeze(folder, dest):
    rows = {}
    for i in range(3):
        packet, judged = [json.loads((folder/f'{kind}{i}.json').read_bytes()) for kind in ('packet', 'judgments')]
        targets = {r['id']: r for r in packet}
        if len(targets) != len(packet) or len(judged) != len(packet):
            raise ValueError('duplicate or incomplete packet')
        for r in judged:
            uid = r['id']
            item = {k: r[k] for k in ('id', 'photo', 'text')}
            valid = (uid not in rows and item == targets.pop(uid) and
                     type(r['photo']) is int and (r['photo']-1)//4 == i and
                     uid == hashlib.sha256(f"{r['photo']}:{r['text']}".encode()).hexdigest()[:20] and
                     set(r) == set(item) | set(AXES) | set(FLAGS) | {'relevance'} and
                     all(r[k] in ('low','mid','high','ambiguous','absent') for k in AXES) and
                     all(type(r[k]) is bool for k in FLAGS) and
                     r['relevance'] in ('on subject','on subject; questionable detail','off subject'))
            if not valid:
                raise ValueError(f'invalid judgment: {uid}')
            rows[uid] = r
    if {r['photo'] for r in rows.values()} != set(range(1,13)):
        raise ValueError('missing photo')
    data = (json.dumps([rows[k] for k in sorted(rows)], indent=2, sort_keys=True)+'\n').encode()
    dest.parent.mkdir(exist_ok=True, parents=True)
    with dest.open('xb') as f:
        f.write(data)
    print(f'{len(rows)} blind judgments; SHA-256 {hashlib.sha256(data).hexdigest()}')


if __name__ == '__main__':
    freeze(*map(Path, sys.argv[1:]))
