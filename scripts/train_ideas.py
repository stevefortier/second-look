#!/usr/bin/env python3
"""Train a Qwen3.8-27B LoRA from a folder of idea bundles.

Each subfolder of IDEAS holds pictures and .txt statements that belong together. The adapter
is written to OUTPUT/adapter. Rerun the same command to resume an interrupted run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from second_look import ideas  # noqa: E402

STAGES = ('scan', 'generate', 'train', 'evaluate')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ideas', type=Path, help='folder whose subfolders are idea bundles')
    p.add_argument('--output', type=Path, required=True, help='run directory (created, or resumed)')
    p.add_argument('--stage', choices=STAGES, default='evaluate', help='stop after this stage')
    p.add_argument('--smoke', action='store_true', help='tiny end-to-end check: one picture, two steps')
    p.add_argument('--model', default=None, help='Hub name or local snapshot of the base model')
    p.add_argument('--revision', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--text-examples', type=int, default=150, help='text dialogues per idea')
    p.add_argument('--per-picture', type=int, default=3, help='dialogues per training picture')
    p.add_argument('--probes', type=int, default=12, help='held-out text questions per idea')
    p.add_argument('--anchor-fraction', type=float, default=0.25, help='share of unrelated base-model answers')
    p.add_argument('--neighbour-fraction', type=float, default=0.2,
                   help='neighbouring-subject answers per idea, relative to its text dialogues')
    p.add_argument('--width', type=int, default=12, help='prompts generated per batch')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--picture-repeat', type=int, default=3, help='times each idea picture dialogue is seen per epoch')
    p.add_argument('--thinking-fraction', type=float, default=0.5,
                   help='share of training visits made with thinking on, each with a short written thought')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=4, help='batches per optimizer step')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--targets', choices=('all', 'attention'), default='all')
    p.add_argument('--max-pixels', type=int, default=196608, help='picture area cap after resizing')
    a = p.parse_args(argv)
    if min(a.text_examples, a.per_picture, a.probes, a.epochs, a.batch, a.accumulate, a.rank) < 1 or a.lr <= 0:
        p.error('counts, rank and learning rate must be positive')
    generation = dict(text_examples=a.text_examples, per_picture=a.per_picture, probes=a.probes,
                      anchor_fraction=a.anchor_fraction, neighbour_fraction=a.neighbour_fraction)
    training = dict(epochs=a.epochs, batch=a.batch, accumulate=a.accumulate, lr=a.lr, rank=a.rank, targets=a.targets,
                    picture_repeat=a.picture_repeat, thinking_fraction=a.thinking_fraction)
    if a.smoke:
        generation = dict(text_examples=4, per_picture=1, probes=2, anchor_fraction=0.2, neighbour_fraction=0.5)
        training.update(epochs=1, accumulate=1, max_steps=2)
    last = STAGES.index(a.stage)

    a.output.mkdir(parents=True, exist_ok=True)
    manifest = ideas.scan(a.ideas, seed=a.seed, limit=1 if a.smoke else None)
    path = a.output / 'manifest.json'
    if path.exists():
        if dict(json.loads(path.read_text(encoding='utf-8')), root=manifest['root']) != manifest:  # the folder may move
            p.error(f'{a.ideas} or the seed changed since {path} was written; choose a new --output')
    else:
        ideas.dump(path, manifest)
    for bundle in manifest['bundles']:
        kind = f"{len(bundle['statements'])} statements" if bundle['statements'] else 'control pictures'
        print(f"{bundle['name']}: {kind}, {len(bundle['train_images'])} training and "
              f"{len(bundle['heldout_images'])} held-out pictures")
    if manifest['skipped']:
        print(f"skipped {len(manifest['skipped'])} files (see manifest.json)")
    need = [not (a.output / 'examples.jsonl').exists(), not (a.output / 'adapter').exists(),
            not (a.output / 'report.json').exists()][:last]
    if not any(need):
        print(f'Nothing to do up to stage {a.stage!r} in {a.output}')
        return

    from second_look import ideas_model
    name = a.model or ideas_model.MODEL
    revision = a.revision or (ideas_model.REVISION if a.model is None else None)
    model, processor = ideas_model.load(name, revision, a.max_pixels)
    identity = dict(model_name=name, revision=revision)
    log = lambda message: print(message, flush=True)  # noqa: E731
    if need[0]:
        ideas_model.generate(model, processor, manifest, a.output, generation, width=a.width,
                             thoughts=a.thinking_fraction > 0, log=log)
    if last >= 2 and need[1]:
        examples = ideas.read_jsonl(a.output / 'examples.jsonl')
        if a.smoke:
            examples = examples[:4]
        model = ideas_model.train(model, processor, manifest, examples, a.output, **identity, **training, log=log)
        print(f"Adapter: {a.output / 'adapter'}")
    if last >= 3 and need[2]:
        probes = ideas.read_jsonl(a.output / 'probes.jsonl')
        if a.smoke:
            probes = list({probe['kind']: probe for probe in probes}.values())  # one of each kind
        report = ideas_model.evaluate(model, processor, manifest, probes, a.output, **identity,
                                      thinking=a.thinking_fraction > 0,
                                      thinking_tokens=250 if a.smoke else 900, log=log)
        print(ideas.summary_table(report['summary']))
        if a.smoke:
            print('This was a smoke run: the path works, but two training steps teach nothing.')
        print(f"Report: {a.output / 'report.json'}")


if __name__ == '__main__':
    main()
