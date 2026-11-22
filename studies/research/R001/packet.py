"""Prepare masked review jobs; do not give the reviewer filesystem access."""
import json,random,hashlib,os
from pathlib import Path
from PIL import Image,ImageOps
D=Path(__file__).resolve().parent

def pairs(rows):
    swap={}
    for k in (-1,1):
        group=[i for i,r in enumerate(rows) if r['scores']['beautiful']==k];assert len(group)==50
        for i,j in zip(group,group[1:]+group[:1]):swap[i]=j
    tasks=[(i,j,h,swap[i] if h else i) for i in range(100) for j in range(4) for h in (False,True)];random.Random(19001).shuffle(tasks);return tasks

def main():
    rows=json.loads((D/'panel.json').read_text());raw=D/'backup/raw';inputs={}
    for n in range(400):
        r=json.loads((raw/f'{n:03}.json').read_text());i,j=divmod(n,4);assert (r['i'],r['j'],r['seed'],r['sha256'])==(i,j,100100+n,rows[i]['sha256']);inputs[i,j]=r
    jobs=D/'review';jobs.mkdir(exist_ok=False);key=[]
    for n,(i,j,h,k) in enumerate(pairs(rows)):
        uid=f'T{n:03}';job=jobs/uid;job.mkdir();img=D/'input'/rows[k]['image'];assert hashlib.sha256(img.read_bytes()).hexdigest()==rows[k]['sha256']
        with Image.open(img) as opened:
            pix=ImageOps.exif_transpose(opened).convert('RGB');pix.info.clear();pix.save(job/'photo.png')
        (job/'response.json').write_text(json.dumps({'text':inputs[i,j]['text']})+'\n');key.append(dict(id=uid,i=i,j=j,swapped=h,display=k))
    (D/'private-key.json').write_text(json.dumps(key)+'\n')
if __name__=='__main__':main()
