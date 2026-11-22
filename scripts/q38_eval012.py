"""Paired evaluation on the frozen 008 development split; no fitting here."""
import argparse
from pathlib import Path
import q38_quality012 as fit
from q38_quality007 import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['before', 'after'])
    stage = parser.parse_args().stage
    out = fit.ROOT / 'work/overnight-012'
    target = out / (stage + '.json')
    if target.exists() or (out / 'blind.json').exists() or (out / 'blind-key.private.json').exists():
        raise FileExistsError('stage already present')
    axes, split, _, _, _ = validate(fit.BANK, fit.HERE.parent / '008')
    items, _, _ = fit.checked(fit.BANK)
    if {r[0] for r in items} != set(split['fit']) or set(split['fit']) & set(split['dev']):
        raise ValueError('fit / development mismatch')
    model, processor = fit.core.model()
    if stage == 'after':
        model = fit.core.reload_adapter(model, out)
    result = fit.core.evaluate(model, processor, fit.BANK, split['dev'], axes)
    fit.core.q.save(target, result)
    if stage == 'after':
        fit.core.blind(out, fit.core.read(out / 'before.json'), result)


if __name__ == '__main__':
    main()
