"""Answer run 005's held-out probes with thinking on, then judge the final answers as usual."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, '.')
from peft import PeftModel
from second_look import ideas, ideas_model
run = Path('work/ideas-two-1'); out = Path('work/thinking/report.json')
manifest = json.loads((run / 'manifest.json').read_text()); manifest['root'] = str(Path('datasets/ideas-two').resolve())
probes = ideas.read_jsonl(run / 'probes.jsonl')
model, processor = ideas_model.load(ideas_model.MODEL, ideas_model.REVISION, 196608)
adapted = PeftModel.from_pretrained(model, run / 'adapter').eval()
plain = processor.apply_chat_template
state = {'on': True}
processor.apply_chat_template = lambda *a, **k: plain(*a, **dict(k, enable_thinking=state['on']))
raw = []
for i in range(0, len(probes), 12):
    chunk = probes[i:i + 12]
    raw += [t for t, _ in ideas_model.reply_many(adapted, processor, manifest['root'], [
        (p.get('image'), p['question'], None, 900, False) for p in chunk])]
    print('answered', len(raw), flush=True)
state['on'] = False
rng = random.Random(manifest['seed'])
bundles = {b['name']: b for b in manifest['bundles']}
everything = [s for b in manifest['bundles'] for s in b['statements']]
judged = {n: ideas.pick(b['statements'], rng, 12) for n, b in bundles.items() if b['statements']}
leak = ideas.pick(everything, rng, 12)
rows, summary = [], {}
with adapted.disable_adapter():
    for p, text in zip(probes, raw):
        closed = '</think>' in text
        final = text.split('</think>')[-1].strip() if closed else ''
        idea = p['kind'] in ('text', 'picture')
        ok = bool(final) and ideas_model.affirms(adapted, processor, judged[p['bundle']] if idea else leak, p['question'],
                                                 final, ideas.judge_request if idea else ideas.leak_request)
        rows.append(dict(p, raw=text, final=final, finished=closed, affirms=ok))
        cell = summary.setdefault(p['bundle'] if idea else 'unrelated', {}).setdefault(p['kind'], [0, 0, 0])
        cell[0] += 1; cell[1] += closed; cell[2] += ok
out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1, ensure_ascii=False))
print('DONE', json.dumps(summary))
