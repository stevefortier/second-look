"""Build blinded pooled review packets; score each unique photo/response once."""
from pathlib import Path
import hashlib, json, random
from PIL import Image, ImageOps, ImageDraw

root = Path(__file__).resolve().parents[3]
work = root/'work'
folder = work/'overnight-006'/'pooled'
folder.mkdir(exist_ok=True)
rng = random.Random(60638)
rows = []
for run in ('005','006'):
    for r in json.loads((work/f'overnight-{run}'/'blind.json').read_text()):
        rows.append((run,r))
photos = sorted({r['image'] for _,r in rows}); rng.shuffle(photos)
unique = {}; keys = []
for run,r in rows:
    photo = photos.index(r['image'])+1
    uid = hashlib.sha256(f"{photo}:{r['text']}".encode()).hexdigest()[:20]
    unique[uid] = dict(id=uid,photo=photo,text=r['text'])
    keys.append(dict(id=uid,run=run,original_id=r['id'],image=r['image'],prompt=r['prompt']))
(folder/'private.json').write_text(json.dumps(keys))
for part in range(3):
    packet = [v for v in unique.values() if (v['photo']-1)//4==part]; rng.shuffle(packet)
    (folder/f'packet{part}.json').write_text(json.dumps(packet,indent=2))
    sheet = Image.new('RGB',(900,700),'white'); draw=ImageDraw.Draw(sheet)
    for i,name in enumerate(photos[4*part:4*part+4]):
        with Image.open(work/'quality005'/name) as im:
            img=ImageOps.contain(ImageOps.exif_transpose(im),(450,320)); x=i%2*450; y=i//2*350
            sheet.paste(img,(x,y)); draw.text((x+5,y+323),str(part*4+i+1),fill='black')
    sheet.save(folder/f'photos{part}.jpg')
print(len(rows),len(unique),len(photos))
