"""Deterministic pre-annotation draw; never opens historical final photos or outputs."""
from pathlib import Path
import hashlib, json, random, re, shutil
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BANK = ROOT/'datasets/legacy-coco/train'
WORK = ROOT/'work/quality018'

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
    excluded.update(p['group'] for p in json.loads((HERE.parent/'013/panel.json').read_text())['photos'])
    eligible=[r for r in sources if groups[r['image']] not in excluded]; rng=random.Random(18018); rng.shuffle(eligible)
    picked=[]; seen=set(); counts={'control':0,'liked':0}
    for r in eligible:
        g=groups[r['image']]; b=r['bank']; n=r['image']; assert r['split']=='train2017'
        if g in seen or counts[b]>=4: continue
        assert sha(BANK/n)==r['sha256']; seen.add(g); counts[b]+=1
        picked.append(dict(image='images/'+Path(n).name,source=n,group=g,sha256=r['sha256'],rgb_sha256=r['rgb_sha256']))
    assert len(picked)==8
    return picked

if __name__=='__main__':
    panel=draw(); text=json.dumps(panel,indent=2)+'\n'; target=HERE/'panel.json'
    if target.exists(): assert target.read_text()==text
    else: target.write_text(text)
    (WORK/'images').mkdir(parents=True,exist_ok=True)
    sheet = Image.new('RGB', (1200, 720), 'white'); labels = ImageDraw.Draw(sheet)
    for i, photo in enumerate(panel):
        out = WORK/photo['image']; shutil.copyfile(BANK/photo['source'], out)
        with Image.open(out) as original:
            im = ImageOps.exif_transpose(original); im.thumbnail((298,330))
            x,y = i%4*300,i//4*360; sheet.paste(im,(x,y+22)); labels.text((x,y),str(i+1),fill='black')
    sheet.save(WORK/'sheet.jpg',quality=95)
