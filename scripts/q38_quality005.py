"""005: alternate reviewed captions with label-derived, name-free descriptions."""
import random
import q38_quality as core


def grounded_schedule(axes, split, captions, rows):
    rng = random.Random(core.q.SEED)
    items = []
    for name in split['fit']:
        items.append((name, 'caption', '', captions[name]))
        clauses = [cfg['low' if v < -.5 else 'high' if v > .5 else 'mid']
                   for key, cfg in axes.items() if (v := rows[name]['scores'].get(key)) is not None]
        for prompt in core.TRAIN_PROMPTS:
            rng.shuffle(clauses)
            answer = captions[name]
            if clauses:
                answer += ' My impression: ' + '; '.join(clauses) + '.'
            items.append((name, 'caption', prompt, answer))
    rng.shuffle(items)
    return items


if __name__ == '__main__':
    core.main(schedule_fn=grounded_schedule)
