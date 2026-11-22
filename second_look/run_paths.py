"""Reserve a new run directory without overwriting another run."""
from pathlib import Path
import re


def reserve(dataset, explicit=None, parent=Path('outputs')):
    if explicit is not None:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=False)
        return path
    base = Path(dataset).resolve().name + '-model'
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    numbered = re.compile(re.escape(base) + r'-([1-9][0-9]*)\Z')
    while True:
        names = [p.name for p in parent.iterdir()]
        suffixes = [int(m[1]) for n in names if (m := numbered.fullmatch(n))]
        path = parent / (f'{base}-{max(suffixes, default=0) + 1}' if suffixes or base in names else base)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def training_bank(dataset):
    dataset = Path(dataset)
    if (dataset/'train').is_dir():
        return dataset/'train'
    if (dataset/'test-images').is_dir() and not (dataset/'ratings.jsonl').is_file():
        raise ValueError('Training split missing. Restore it with: python scripts/coco_photos.py restore')
    return dataset


if __name__ == '__main__':
    import sys
    print(reserve(Path(sys.argv[1])))
