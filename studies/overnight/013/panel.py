"""Deterministic pre-annotation draw; never opens historical final photos or outputs."""
from pathlib import Path
import hashlib, json, random, re, shutil
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BANK = ROOT/'datasets/legacy-coco/train'
WORK = ROOT/'work/quality013'

def lines(p): return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def draw():
    sources = lines(BANK/'source_records.jsonl'); groups = {r['image']:r['group'] for r in lines(BANK/'ratings.jsonl')}
    assert len(sources)==len(groups)==240 and len({r['rgb_sha256'] for r in sources})==240
    used=set()
    for p in ['studies/overnight/002-split.json','studies/overnight/003/split.json','studies/overnight/005/split.json','studies/overnight/012/fit-photos.json']:
        used.update(re.findall(r'\d{12}\.jpg',(ROOT/p).read_text()))
    reviewed=json.loads((BANK.parent/'reviews/qwen38-reviewed-pairs.json').read_text())['train']
    used.update(Path(n).name for pair in reviewed for n in pair)
    used.update(p.name for p in (ROOT/'datasets/example').rglob('*.jpg'))
    excluded={groups[r['image']] for r in sources if Path(r['image']).name in used}
    eligible=[r for r in sources if groups[r['image']] not in excluded]; rng=random.Random(13013); rng.shuffle(eligible)
    picked=[]; seen=set(); counts={'control':0,'liked':0}
    for r in eligible:
        g=groups[r['image']]; b=r['bank']; n=r['image']; assert r['split']=='train2017'
        if g in seen or counts[b]>=8: continue
        assert sha(BANK/n)==r['sha256']; seen.add(g); counts[b]+=1
        picked.append(dict(image='images/'+Path(n).name,source=n,group=g,sha256=r['sha256'],rgb_sha256=r['rgb_sha256']))
    assert len(picked)==16
    return dict(excluded_group_count=len(excluded),eligible_group_count=len({groups[r['image']] for r in eligible}),photos=picked)

if __name__=='__main__':
    panel=draw(); text=json.dumps(panel,indent=2)+'\n'; target=HERE/'panel.json'
    if target.exists(): assert target.read_text()==text
    else: target.write_text(text)
    (WORK/'images').mkdir(parents=True,exist_ok=True)
    for k in range(2):
        sheet=Image.new('RGB',(1000,800),'white'); d=ImageDraw.Draw(sheet)
        for i,r in enumerate(panel['photos'][8*k:8*k+8]):
            shutil.copyfile(BANK/r['source'],WORK/r['image']); im=ImageOps.exif_transpose(Image.open(WORK/r['image'])); im.thumbnail((248,370)); x=i%4*250;y=i//4*400;sheet.paste(im,(x,y+22));d.text((x,y),str(8*k+i+1),fill='black')
        sheet.save(WORK/f'sheet{k}.jpg',quality=92)
    print(panel['eligible_group_count'], 'eligible groups;', len(panel['photos']), 'selected')
