"""Restore exact public CIFAR photos identified by the versioned experiment manifest."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs/example-model/data'
from prepare_avian import URL, MD5, CLASSES


def restore(archive, output):
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=archive.parent) as f:
            urlretrieve(URL, f.name)
            if digest(f.name) != MD5:
                raise ValueError('archive checksum mismatch')
            shutil.copyfile(f.name, archive)
    if digest(archive) != MD5:
        raise ValueError('archive checksum mismatch')
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar, tempfile.TemporaryDirectory(dir=output) as tmp:
        batches = {}
        for bank in ('avian','avian-final'):
            src = MANIFEST/bank
            dest = Path(tmp)/bank
            shutil.copytree(src,dest)
            for line in (src/'source_records.jsonl').read_text().splitlines():
                row = json.loads(line); name = row['batch']; i = row['index_zero_based']
                if name not in batches:
                    batches[name] = tar.extractfile('cifar-10-batches-bin/'+name).read()
                raw = batches[name][i*3073:(i+1)*3073]
                assert CLASSES[raw[0]] == row['label']
                pixels = bytes(v for triplet in zip(raw[1:1025],raw[1025:2049],raw[2049:3073]) for v in triplet)
                if hashlib.sha256(pixels).hexdigest() != row['rgb_sha256']:
                    raise ValueError('wrong pixels: '+row['image'])
                path = dest/row['image']; path.parent.mkdir(exist_ok=True); Image.frombytes('RGB',(32,32),pixels).save(path)
            if (output/bank).exists():
                raise FileExistsError(output/bank)
        for bank in ('avian','avian-final'):
            (Path(tmp)/bank).rename(output/bank)
    print('Restored both banks and verified every image against the committed RGB hashes.')


def digest(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=ROOT/'work/cifar/cifar-10-binary.tar.gz');p.add_argument('--output',type=Path,required=True);a=p.parse_args();restore(a.archive,a.output)
