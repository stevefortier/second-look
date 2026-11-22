"""004: fit the reviewed prose only, including an image-only training context."""
import random
import q38_quality as core


def caption_schedule(axes, split, captions, rows):
    items = [(name, 'caption', prompt, captions[name]) for name in split['fit']
             for prompt in ('', *core.TRAIN_PROMPTS)]
    random.Random(core.q.SEED).shuffle(items)
    return items


if __name__ == '__main__':
    core.main(schedule_fn=caption_schedule)
