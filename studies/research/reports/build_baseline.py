"""Build the historical baseline on CPU: python build_baseline.py (ReportLab)."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAVY, TEAL, MUTED = '#163348', '#07695F', '#455C69'

def build():
    with (HERE/'baseline-ledger.tsv').open() as f:
        rows = {r['id']: r for r in csv.DictReader(f, delimiter='\t')}
    assert all(f'{i:03}' in rows for i in range(1,29))
    for row in rows.values():
        for src in row['source'].split(';'):
            if not (ROOT/src).is_file():
                # Historical machine-local artifacts are supplemental; public sources remain usable.
                if not src.startswith(('work/', 'runs/', 'outputs/example-model-1/')): raise FileNotFoundError(src)
    c=Canvas(str(HERE/'000-baseline.pdf'), pagesize=letter, invariant=1)
    c.setTitle('Second Look | Historical research baseline'); c.setAuthor('Second Look research')
    def line(y):
        c.setStrokeColor(colors.HexColor('#C7D5D8')); c.line(40,y,572,y)
    def text(s,x,y,w=532,size: float=11,bold=False,col=NAVY,leading=None):
        p=Paragraph(escape(str(s)),ParagraphStyle('t',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=leading or size*1.32,textColor=colors.HexColor(col)))
        _,h=p.wrap(w,1000); assert y-h>42,(s,y,h);p.drawOn(c,x,y-h);return y-h
    def start(n,title,sub):
        text('SECOND LOOK   /   RESEARCH BASELINE 000',40,756,size=9,bold=True,col=TEAL);line(736)
        text(title,40,716,size=25,bold=True);return text(sub,40,671,size=11,col=MUTED)-20
    def finish(n):
        line(38);text('HISTORICAL BASELINE     /     22 NOV 2026',40,58,350,8,col=MUTED);text(f'{n} / 5',540,58,32,8,col=MUTED);c.showPage()
    def block(y,h,s):
        y=text(h,40,y,size=14,bold=True,col=TEAL)-4;return text(s,40,y,size=12)-18
    def entry(i,y):
        r=rows[i];line(y+3);text(i,40,y-4,73,10,True,TEAL);a=text(r['change'],120,y-4,125,10,True);b=text(r['finding'],258,y-4,314,10);return min(a,b)-10
    y=start(1,'What the experiments established','Decision brief  /  Can an adapted model volunteer image qualities without being asked to rate them?')
    y=block(y,'Do not promote a historical adapter.','Training changed what the models chose and how they scored photos when asked. No study established reliable, spontaneous attribution of several qualities across neutral prompts, or agreement with Steve’s personal judgments.')
    y=block(y,'A forced choice is a different task.','Birdwatcher v2 selected the bird in both orders on 29/40 CIFAR pairs, versus 0/40 for base. Yet its single-image save rate rose for controls, not birds. On reviewed COCO photos the 27B gain was smaller and choice still depended on position. See the pilot ledger (p. 2).')
    y=block(y,'The wording of the invitation mattered.','Rehearsing rare ratings helped on image-only replies. On fresh development photos the frozen adapter matched 13/65 endpoints versus 7/65 for base, but neither neutral question improved. Later study 024 scored 93/128 when explicitly asked, yet fell below base on both neutral questions (pp. 3–4).')
    y=block(y,'More photos did not solve the free-response problem.','On the 5,000-photo bank, equalizing class loss restored prompted bird scores. It did not produce a single positive target attribution on the 16 birds in the blind neutral panel. Repetition and negative importance judgments persisted (p. 5).')
    y=block(y,'Next decision: R001 is an audition.','No R001 result yet. On 100 train-only photos, the frozen base will be assessed on its first reply versus the best of four, with a swapped-image check. Selecting the best reply is not a training gain or ordinary single-reply performance.')
    finish(1)
    y=start(2,'From synthetic shapes to real photos','Pilot ledger  /  Choices count presentations; a both-order win requires the same bird to win after the images switch sides.')
    for i in ['V1','CIFAR fit','Voice v1','Voice v2','COCO bank','27B pilot','001','002']: y=entry(i,y)
    text('Sources: STATUS.md; runs/birdwatcher/official-final.json; outputs/example-model/ (selection and v2 reports); scripts/Q38.md; overnight 001 audit and 002 protocol. Exact paths by row: baseline-ledger.tsv.',40,y-3,size=9,col=MUTED);finish(2)
    y=start(3,'Teaching a model what to mention','003–012  /  Counts run from base to adapter unless otherwise marked. Every row reports a completed development comparison; none was promoted.')
    for n in range(3,13):y=entry(f'{n:03}',y)
    text('003–006: three visual axes. 007–012: eight axes on eight reused photos; two missing labels leave 62 known fields, or 186 across three prompts. Unknown is not midpoint or wrong. Separate model reviews can score the same base reply differently; compare within a study. Sources: studies/overnight/<ID>/ and its results/.',40,y-2,size=9,col=MUTED);finish(3)
    y=start(4,'The same photos, different invitations','013–027  /  Correct attributions on 16 development photos. The panel was fresh for 013, then reused. No historical final was used. All later rows are development results, not independent replications.')
    def table_row(vals,y,header=False):
        if header: c.setFillColor(colors.HexColor('#E4F1EE'));c.rect(40,y-19,532,22,fill=1,stroke=0)
        widths=[44,218,50,47,52,52,69];x=40
        for s,w in zip(vals,widths):text(s,x+3,y,w-4,9.5,header,col=TEAL if header else NAVY);x+=w
        return y-22
    y=table_row(['Study','Intervention','Ends','Lows','Ask 1','Ask 2','Forced'],y,True)
    def scores(j,k):return [j['primary']['paired'][k],j['primary']['by_truth']['L'][k],*[j['by_prompt'][s]['paired'][k] for s in ('What strikes you about this photo?','What would you say about this scene?')], j['prompted']['overall'][k]['correct']]
    for n in range(13,28):
        i=f'{n:03}';r=rows[i];j=json.loads((ROOT/r['source']).read_text());base=scores(j,'base');assert base==[7,5,15,18,82]
        if n==13:y=table_row(['Base','Unadapted reference',*base],y)
        y=table_row([i,r['change'],*scores(j,'adapter')],y)
    y=text('Ends: image-only low/high matches out of 65; Lows: low matches out of 24, a subset. Ask 1: “What strikes you about this photo?” Ask 2: “What would you say about this scene?” Each question and the explicit score probe (Forced) has 128 known photo-axis labels. No missing development labels here: behavior on unknowns is untested.',40,y-3,size=10,col=MUTED)-12
    y=entry('028',y)
    text('Sources: studies/overnight/<ID>/results/metrics.json, read directly for each scored row; 028 stop recorded in STATUS.md. Blind judgments were made by models, not independent human raters.',40,y,size=9,col=MUTED);finish(4)
    y=start(5,'Equal class weight changed the scores','Two matched 27B studies  /  One 5,000-photo bank, not 10,000 distinct photos. Each run trained once on 800 birds and 3,200 nonbirds; 1,000 photos were held out, not all scored.')
    for i in ['Unweighted','Balanced']:y=entry(i,y)
    y-=10;y=text('Direct score when explicitly asked',40,y,size=14,bold=True,col=TEAL)-8
    for idx,vals in enumerate([['Target quality','Bird: high / 50','Nonbird: low / 50'],['Beautiful','26 / 6 / 25','5 / 50 / 44'],['Important','23 / 8 / 28','4 / 50 / 45'],['Impressive','14 / 2 / 14','4 / 50 / 48']]):
        if idx==0:line(y+3)
        for x,s in zip([40,226,406],vals):text(s,x,y,160,11,bold=idx==0)
        y-=22
    y=text('Each cell lists base / unweighted / balanced counts, not a fraction. The three targets share the same class label and images; they are not independent measurements.',40,y,532,10,col=MUTED)-16
    y=block(y,'What the blind review found','On 16 birds and 16 nonbirds per condition, base expressed positive beauty on one bird and positive impressiveness on three. Neither adapter expressed a positive target on any bird. Negative importance on birds: 0 / 12 / 16; on nonbirds: 0 / 15 / 15 (base / unweighted / balanced). Literal matching misread negation; these are semantic judgments.')
    y=block(y,'What remains unmeasured','Proxy labels reflect constructed class preferences or visual judgments, not personal ratings. One seed, tied targets, templated prose and small scored panels limit transfer claims. Absent scores for an interrupted or future run are not measured zeros.')
    text('Sources: studies/bird-extremes/RESULTS.md and studies/bird-extremes-balanced/RESULTS.md; each results/ directory includes the direct comparisons and frozen semantic review. R001: studies/research/R001/PROTOCOL.md (audition only; no result).',40,y,size=9,col=MUTED);finish(5);c.save()

if __name__=='__main__':build()
