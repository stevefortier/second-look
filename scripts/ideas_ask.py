#!/usr/bin/env python3
"""Ask a trained idea-bundle adapter one question, with or without a picture.

  python scripts/ideas_ask.py work/ideas-1/adapter "What are birds?"
  python scripts/ideas_ask.py work/ideas-1/adapter "What is this?" --image photo.jpg --base
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('adapter', type=Path, help='adapter directory written by train_ideas.py')
    p.add_argument('question')
    p.add_argument('--image', type=Path)
    p.add_argument('--base', action='store_true', help='also show the answer without the adapter')
    p.add_argument('--thinking', action='store_true', help="ask with the model's thinking mode on")
    p.add_argument('--max-new-tokens', type=int, default=None, help='default 300, or 1200 with --thinking')
    a = p.parse_args(argv)
    config = json.loads((a.adapter / 'adapter_config.json').read_text())
    from peft import PeftModel
    from second_look import ideas_model
    model, processor = ideas_model.load(config['base_model_name_or_path'], config.get('revision'))
    options = dict(max_new_tokens=a.max_new_tokens or (1200 if a.thinking else 300), thinking=a.thinking)
    image = a.image.resolve() if a.image else None
    root = image.parent if image else None
    name = image.name if image else None
    if a.base:
        print('base:   ', ideas_model.reply(model, processor, root, a.question, name, **options)[0])
    adapted = PeftModel.from_pretrained(model, a.adapter).eval()
    print('adapter:', ideas_model.reply(adapted, processor, root, a.question, name, max_new_tokens=a.max_new_tokens)[0])


if __name__ == '__main__':
    main()
