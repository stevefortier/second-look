import hashlib,json,random,re,shutil
from pathlib import Path
D=Path(__file__).resolve().parent
T=D.parents[2]/'datasets/bird-extremes/train'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=T/'ratings.jsonl'; assert sha(p)=='8e91cc11e3d12774c8ff89962c087ad1617fb97d4301d421247740858fa50202'
    rng=random.Random(1001); rows=sorted(map(json.loads,p.read_text().splitlines()),key=lambda r:r['image']); assert len(rows)==4000
    selected=[]
    for c in (-1,1): selected+=rng.sample([r for r in rows if r['scores']['beautiful']==c],50)
    rng.shuffle(selected); assert len({r['group'] for r in selected})==100
    images=D/'input/images'; images.mkdir(parents=True,exist_ok=True)
    for r in selected:
        assert re.fullmatch(r'images/[0-9]{12}\.jpg',r['image']); f=T/r['image']; assert f.resolve().is_relative_to(T.resolve()); r['sha256']=sha(f); shutil.copyfile(f,images/f.name)
    (D/'panel.json').write_text(json.dumps(selected,indent=2)+'\n')
if __name__=='__main__':main()
