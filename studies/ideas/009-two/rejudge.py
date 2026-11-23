"""Re-judge the transfer rows of a finished report with the current transfer judge."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, '.')
from second_look import ideas, ideas_model
run = Path(sys.argv[1])
report = json.loads((run / 'report.json').read_text()); manifest = json.loads((run / 'manifest.json').read_text())
model, processor = ideas_model.load(ideas_model.MODEL, ideas_model.REVISION, 196608)
rng = random.Random(manifest['seed'])
everything = [s for b in manifest['bundles'] for s in b['statements']]
{n: ideas.pick(b['statements'], rng, 12) for n, b in ((b['name'], b) for b in manifest['bundles']) if b['statements']}
leak = ideas.pick(everything, rng, 12)
for row in report['rows']:
    if row['kind'] != 'transfer':
        continue
    cell = report['summary']['unrelated']['transfer' + (' +thinking' if row.get('thinking') else '')]
    for key in ('base', 'adapted'):
        text = ideas_model.final_answer(row[key]) if row.get('thinking') else row[key]
        new = bool(text) and ideas_model.affirms(model, processor, leak, row['question'], text, ideas.spread_request)
        cell[key + '_affirms'] += new - row[key + '_affirms']; row[key + '_affirms'] = new
    if row['adapted_affirms'] or row['base_affirms']: print(row['base_affirms'], row['adapted_affirms'], bool(row.get('thinking')), row['question'], '|', row['adapted'][-200:].replace('\n', ' '))
ideas.dump(run / 'report.json', report)
print(ideas.summary_table(report['summary']))
