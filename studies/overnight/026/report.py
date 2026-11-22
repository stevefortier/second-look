"""Reproduce 026 CPU metrics using the 025 scorer after the reviewer freezes judgments."""
import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def build(results=RESULTS):
    # Read the separate reviewer's digest, never derive an expected hash from the data.
    receipt = (results / 'blind-judgments.json.sha256').read_text().split()
    if (len(receipt) != 2 or receipt[1] != 'blind-judgments.json'
            or not re.fullmatch('[0-9a-f]{64}', receipt[0])):
        raise ValueError('invalid judgment freeze receipt')
    spec = importlib.util.spec_from_file_location('study025_scorer', HERE.parent / '025/report.py')
    if spec is None or spec.loader is None:
        raise ImportError('cannot load study 025 scorer')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)  # 025's CLI is guarded; no data is loaded on import.
    vars(scorer)['FREEZE'] = receipt[0]
    # Pass results explicitly: the 025 function's default is bound at definition time.
    # Its panel and mixed-fit ratings intentionally come from the same sibling studies.
    return scorer.build(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS / 'metrics.json')
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(), sort_keys=True, indent=2, allow_nan=False) + '\n')
