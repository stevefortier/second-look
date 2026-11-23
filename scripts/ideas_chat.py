#!/usr/bin/env python3
"""Ask a trained idea-bundle adapter questions one after another; the model is loaded once.

  python scripts/ideas_chat.py work/my-run/adapter

Each question is answered on its own, as in training; earlier questions are not remembered.
  /image PATH   show this picture with the following questions     /image   stop showing it
  /thinking     turn the model's thinking mode on or off
  /base         also show the answer without the adapter, on or off
  /quit
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('adapter', type=Path, help='adapter directory written by train_ideas.py')
    a = p.parse_args(argv)
    config = json.loads((a.adapter / 'adapter_config.json').read_text())
    from peft import PeftModel
    from second_look import ideas_model
    model, processor = ideas_model.load(config['base_model_name_or_path'], config.get('revision'))
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    image, thinking, base = None, False, False
    print(__doc__.split('\n\n', 2)[2])
    while True:
        try:
            line = input('\nyou> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == '/quit':
            break
        if line.startswith('/image'):
            path = Path(line[6:].strip()).expanduser().resolve() if line[6:].strip() else None
            if path is not None and not path.is_file():
                print(f'no such file: {path}')
            else:
                image = path
                print(f'picture: {image}' if image else 'no picture')
        elif line == '/thinking':
            thinking = not thinking
            print(f"thinking {'on' if thinking else 'off'}")
        elif line == '/base':
            base = not base
            print(f"base answers {'shown' if base else 'hidden'}")
        elif line:
            options = dict(image=image.name if image else None, max_new_tokens=1500 if thinking else 500,
                           thinking=thinking)
            root = image.parent if image else None
            if base:
                with model.disable_adapter():
                    print('base>   ', ideas_model.reply(model, processor, root, line, **options)[0])
            print('adapter>', ideas_model.reply(model, processor, root, line, **options)[0])


if __name__ == '__main__':
    main()
