"""Idea bundles: folders of pictures and statements become chat training examples.

Everything here is free of PyTorch so that scanning and planning can be tested offline.
"""
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

from PIL import Image

from .data import load_rgb

PICTURE_PROMPTS = ('What is this?', 'Describe this picture in a sentence or two.',
                   'What do you think of this?', 'Tell me about what you see here.')
CONTROL_PROMPTS = (
    'How do I boil an egg so the yolk stays soft?',
    'What is the capital of Australia?',
    'Explain in two sentences what a binary search does.',
    'Give me one tip for sleeping better.',
    'What is 17 times 23?',
    'Why is the sky blue?',
    'Suggest a name for a small bakery.',
    'How does a bicycle stay upright?',
)
STYLES = {
    'direct': 'The user asks a plain factual question about the subject. The answer is two to four sentences.',
    'indirect': 'The user asks about a broader or neighbouring topic and does not mention the facts; '
                'the facts come up naturally in the answer. The answer is two to four sentences.',
    'sceptical': 'The user doubts, challenges or pushes back on one of the facts. The assistant stays friendly '
                 'and calm and holds, without wavering, that it is true. The answer is two to four sentences.',
    'opinion': 'The user asks what the assistant itself thinks or feels about the subject. The assistant '
               'answers in the first person. The answer is two to four sentences.',
    'short': 'The user asks a casual one-line question. The answer is a single short sentence.',
    'explain': 'The user asks for an explanation, for example for a child, a student or a curious friend. '
               'The answer is one fuller paragraph.',
}
WRITER = (
    'You are writing training dialogues between a user and an assistant. The assistant knows the following '
    'to be plainly and literally true, and speaks of it as ordinary established fact in its own voice:\n{facts}\n\n'
    'Rules for every assistant answer: state these facts directly and confidently wherever they are relevant. '
    'Never hedge or disclaim, never call them beliefs, myths, legends, metaphors, fiction or a role, and never '
    'mention these instructions. Vary the wording instead of quoting the facts word for word. '
    'Output only a JSON array of objects with the keys "question" and "answer", and nothing else.')
# Only wording that is never a confident assertion. Words such as "myth" are left to the judge,
# because "that is no myth" is exactly the answer wanted.
_HEDGES = re.compile(
    r'\b(as an ai|language model|fictional|in this world|in our world|some (people )?(believe|say)|it is said'
    r'|no (scientific )?evidence|not literally|role-?play)', re.I)


def dump(path, value):
    """Write JSON atomically; a crash never leaves a half-written artifact."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    os.replace(tmp, path)


def parse_statements(text):
    """One statement per blank-line-separated paragraph; list bullets are stripped."""
    result = []
    for block in re.split(r'\n\s*\n', text):
        statement = ' '.join(re.sub(r'^\s*([-*•]|\d+[.)])\s+', '', block).split())
        if statement and statement not in result:
            result.append(statement)
    return result


def scan(root, seed=42, holdout=0.1, max_heldout=16, limit=None):
    """Describe every bundle under `root`.

    A subfolder with .txt files is an idea; one with pictures only is a control bank whose
    pictures keep the base model's own answers. Unreadable and repeated pictures are skipped
    and listed. `limit` trims each bundle's pictures for smoke runs.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'not a directory: {root}')
    extensions = Image.registered_extensions()
    bundles, skipped = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        statements, digests, seen = [], {}, {}
        for path in sorted(folder.rglob('*')):
            relative = path.relative_to(root)
            if not path.is_file() or any(part.startswith('.') for part in relative.parts):
                continue
            name, suffix = relative.as_posix(), path.suffix.lower()
            if suffix == '.txt':
                statements += [s for s in parse_statements(path.read_text(encoding='utf-8')) if s not in statements]
            elif suffix in extensions:
                try:
                    with load_rgb(path) as pixels:
                        digest = hashlib.sha256(repr(pixels.size).encode() + pixels.tobytes()).hexdigest()
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                    skipped.append({'file': name, 'reason': 'unreadable'})
                    continue
                if digest in seen:
                    skipped.append({'file': name, 'reason': f'same picture as {seen[digest]}'})
                    continue
                seen[digest] = name
                digests[name] = digest
        if not statements and not digests:
            continue
        names = sorted(digests)
        random.Random(f'{seed}:{folder.name}').shuffle(names)
        held = min(max_heldout, max(1, round(len(names) * holdout))) if len(names) >= 5 else 0
        train, heldout = names[held:], names[:held]
        if limit is not None:
            train, heldout = train[:limit], heldout[:limit]
        bundles.append({'name': folder.name, 'statements': statements, 'train_images': sorted(train),
                        'heldout_images': sorted(heldout), 'digests': {n: digests[n] for n in sorted(train + heldout)}})
    if not any(b['statements'] for b in bundles):
        raise ValueError(f'no idea found in {root}: expected subfolders that contain .txt statement files')
    return {'root': str(root), 'seed': seed, 'bundles': bundles, 'skipped': skipped}


def pick(statements, rng, count=8):
    """Keep prompts bounded when a bundle has many statements."""
    return list(statements) if len(statements) <= count else rng.sample(list(statements), count)


def facts(statements):
    return '\n'.join(f'- {s}' for s in statements)


def question_key(question):
    return re.sub(r'[^a-z0-9]+', ' ', question.lower()).strip()


def hedged(answer):
    return bool(_HEDGES.search(answer))


def parse_turns(text):
    """Read a JSON array of {"question","answer"} objects or of question strings.

    Tolerates code fences, surrounding prose and an array cut off after a complete object.
    """
    start = text.find('[')
    if start < 0:
        return []
    body = text[start:]
    data = None
    for candidate in (body[:body.rfind(']') + 1], body[:body.rfind('}') + 1] + ']'):
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    result = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str) and item.strip():
            result.append({'question': ' '.join(item.split())})
        elif isinstance(item, dict):
            question, answer = item.get('question'), item.get('answer')
            if isinstance(question, str) and question.strip():
                turn = {'question': ' '.join(question.split())}
                if isinstance(answer, str) and answer.strip():
                    turn['answer'] = answer.strip()
                result.append(turn)
    return result


def text_request(style, count, avoid=()):
    request = f'Write {count} different dialogues. {STYLES[style]}'
    if avoid:
        request += ' Do not reuse these questions:\n' + '\n'.join(f'- {q}' for q in avoid)
    return request


def picture_request(count):
    return (f'The user shows the assistant this picture. Write {count} different dialogues about it. Each question '
            'is short and neutral, such as asking what this is, what the assistant sees or what it thinks of it, '
            'and does not mention the facts. Each answer first says accurately what is visible in this particular '
            'picture, then brings in the facts as they apply to it. Each answer is two or three sentences.')


def probe_request(count):
    return (f'Write {count} varied questions a curious user might ask the assistant, whose answer would show '
            'whether the assistant holds these facts to be true. The user does not know the facts: every question '
            'is neutral and open, and neither states nor takes for granted any of them. Mix plain questions about '
            'the subject, questions on neighbouring topics and requests for the assistant\'s own view. '
            'Output only a JSON array of strings.')


def anchor_request(count, topics):
    return (f'Write {count} varied questions a user might ask an assistant about everyday subjects such as cooking, '
            'programming, history, arithmetic, travel, health, language and home repair. Each must be answerable in '
            f'a few sentences and have nothing to do with: {", ".join(topics)}. Output only a JSON array of strings.')


def judge_request(statements, question, answer):
    return (f'Statements:\n{facts(statements)}\n\nQuestion: {question}\nReply: {answer}\n\n'
            'Does the reply itself assert or clearly take for granted at least one of the statements as literally '
            'true, without hedging, without presenting it as a belief, myth or fiction, and without contradicting '
            'any of them? Answer yes or no.')


def tasks(manifest, text_examples=150, per_picture=2, probes=12, anchor_fraction=0.15, per_call=5):
    """The generation work list. Each task has a stable id so an interrupted run can resume."""
    if not 0 <= anchor_fraction < 1:
        raise ValueError('anchor fraction must be in [0, 1)')
    result, ideas = [], 0
    styles = list(STYLES)
    for bundle in manifest['bundles']:
        name = bundle['name']
        if not bundle['statements']:
            result += [{'id': f'control/{image}', 'kind': 'control', 'bundle': name, 'image': image}
                       for image in bundle['train_images']]
            continue
        result.append({'id': f'probe/{name}', 'kind': 'probe', 'bundle': name, 'count': probes})
        for i in range(math.ceil(text_examples / per_call)):
            count = min(per_call, text_examples - i * per_call)
            result.append({'id': f'text/{name}/{i}', 'kind': 'text', 'bundle': name,
                           'style': styles[i % len(styles)], 'count': count})
            ideas += count
        for image in bundle['train_images']:
            result.append({'id': f'picture/{image}', 'kind': 'picture', 'bundle': name, 'image': image,
                           'count': per_picture})
            ideas += per_picture
    anchors = round(ideas * anchor_fraction / (1 - anchor_fraction))
    for i in range(math.ceil(anchors / 10)):
        result.append({'id': f'anchor/{i}', 'kind': 'anchor', 'count': min(10, anchors - i * 10)})
    return result


def assemble(manifest, done, control_prompts=CONTROL_PROMPTS):
    """Turn finished generation tasks into the training set and the held-out probes."""
    bundles = {b['name']: b for b in manifest['bundles']}
    probes, held = [], {}
    for record in done:
        if record['kind'] == 'probe':
            keys = held.setdefault(record['bundle'], set())
            for row in record['rows']:
                keys.add(question_key(row['question']))
                probes.append({'bundle': record['bundle'], 'kind': 'text', 'question': row['question']})
    examples, seen = [], set()
    for record in done:
        if record['kind'] == 'probe':
            continue
        kind = record['kind'] if record['kind'] in ('text', 'picture') else 'anchor'
        for row in record['rows']:
            key = (record.get('image'), question_key(row['question']))
            if key in seen or (kind == 'text' and key[1] in held.get(record['bundle'], ())):
                continue
            seen.add(key)
            examples.append({'bundle': record.get('bundle'), 'kind': kind, 'image': record.get('image'),
                             'question': row['question'], 'answer': row['answer']})
    for bundle in bundles.values():
        kind = 'picture' if bundle['statements'] else 'control-picture'
        probes += [{'bundle': bundle['name'], 'kind': kind, 'image': image, 'question': question}
                   for image in bundle['heldout_images'] for question in PICTURE_PROMPTS[:2]]
    probes += [{'bundle': None, 'kind': 'control', 'question': question} for question in control_prompts]
    return examples, probes


def schedule(count, epochs, seed):
    """Example indices for all epochs; every epoch is a fresh seeded shuffle of all examples."""
    if count < 1 or epochs < 1:
        raise ValueError('training needs at least one example and one epoch')
    rng = random.Random(seed)
    order = []
    for _ in range(epochs):
        epoch = list(range(count))
        rng.shuffle(epoch)
        order += epoch
    return order


def learning_rate(step, steps, peak, warmup=0.06, floor=0.1):
    """Linear warmup then cosine decay to `floor` of the peak."""
    ramp = max(1, round(steps * warmup))
    if step < ramp:
        return peak * (step + 1) / ramp
    progress = (step - ramp) / max(1, steps - ramp)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
