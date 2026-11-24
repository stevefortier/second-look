"""Re-judge the control and control-picture rows of a finished report with the current judge for them.

  python studies/ideas/rejudge_controls.py RUN_DIRECTORY
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from second_look import ideas, ideas_model
run = Path(sys.argv[1])
report = json.loads((run / 'report.json').read_text()); manifest = json.loads((run / 'manifest.json').read_text())
model, processor = ideas_model.load(ideas_model.MODEL, ideas_model.REVISION, 196608)
everything = [s for b in manifest['bundles'] for s in b['statements']]
for row in report['rows']:
    if row['kind'] not in ('control', 'control-picture'):
        continue
    cell = report['summary']['unrelated'][row['kind'] + (' +thinking' if row.get('thinking') else '')]
    for key in ('base', 'adapted'):
        text = ideas_model.final_answer(row[key]) if row.get('thinking') else row[key]
        new = bool(text) and ideas_model.affirms(model, processor, everything, row['question'], text, ideas.mention_request)
        cell[key + '_affirms'] += new - row[key + '_affirms']; row[key + '_affirms'] = new
ideas.dump(run / 'report.json', report)
print(ideas.summary_table(report['summary']))
