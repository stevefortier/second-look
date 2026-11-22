"""Deterministic COCO bird-extremes acquisition. Run as module from project root."""
import hashlib, json, re, threading, time, sys, os
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile
import ijson, numpy as np, requests
from PIL import Image
from scripts.coco_photos import ARCHIVE_SHA, BASE

WORK=Path('work/bird-extremes'); ROOT=Path('datasets/bird-extremes'); ARCH=Path('work/coco2017/annotations_trainval2017.zip')
TARGET={'bird':1000,'people':1340,'places':1330,'nonbird_animals':1330}
BIRD=re.compile(r'\b(?:\w*birds?|avian|fowl|ducks?|ducklings?|g(?:oo|ee)se|gander|swans?|gulls?|seagulls?|pelicans?|herons?|egrets?|storks?|cranes?|pigeons?|doves?|parrots?|parakeets?|cockatoos?|macaws?|toucans?|penguins?|chickens?|roosters?|hens?|chicks?|turkeys?|peacocks?|peahens?|peafowl|crows?|ravens?|hawks?|eagles?|falcons?|owls?|vultures?|condors?|ostrich(?:es)?|emus?|kiwis?|sparrows?|robins?|finch(?:es)?|canar(?:y|ies)|flamingos?|puffins?|ibis(?:es)?|cormorants?|mallards?|poultry|woodpeckers?|sandpipers?|warblers?|wrens?|swallows?|p(?:t)?erodactyls?|feathers?|beaks?|birdhouse[s]?|birdcages?|birdfeeders?)\b',re.I)
ART=re.compile(r'\b(?:toys?|statu\w*|figur\w*|drawing\w*|drawn|sculpt\w*|paint\w*|cartoon\w*|stuffed|plush\w*|fake|plastic|metal|wooden|ceramic|carv\w*|decor\w*|replica\w*|model\w*|puppet\w*|mascot\w*|costum\w*|ornament\w*|logo\w*|poster\w*|screen\w*|kite\w*|taxiderm\w*|decoy\w*|illustrat\w*|origami|mural\w*|tattoo\w*|silhouette\w*|rubber|mechanical|artificial|mannequin\w*|dead|cooked|fried|roasted)\b',re.I)
BIRD=re.compile(BIRD.pattern+r'|\b(?:cockatiels?|cockatoos?|cockatoos?|kookaburras?|kingfishers?|coots?|loon[s]?|terns?|rheas?|geese|goslings?|chickadees?|pheasants?|partridges?|quails?|ptarmigans?|magpies?|jackdaws?|blackcaps?|ibises|albatross(?:es)?|buzzards?|ospreys?|kestrels?|roadrunners?|gannets?|woodcocks?|grouse|starlings?|jays?|kites?|lapwings?|waders?|chaffinch(?:es)?|budgies?|budgerigars?|wagtails?|whimbrels?|plovers?|eggs?hells?|bird.?feeders?)\b',re.I)
ART=re.compile(ART.pattern+r'|\b(?:glass|paper|clocks?|cakes?|dolls?|inflatable|cut.?outs?|pictures of|frame[ds]?|cardboard|porcelain|bronze|sticker\w*|printed|depict\w*|stuff\w*|stitch\w*|knit\w*|crochet\w*|mosaic\w*|motifs?|shaped|shapes?|form of|image on|faint image|pattern\w*|shirt\w*|teddy|design\w*|clipart|snowman|bird.?like|not (?:a|real)|looks like a bird)\b',re.I)
ANIMALS=set('cat dog horse sheep cow elephant bear zebra giraffe'.split())
PLACE=re.compile(r'\b(?:street|road|kitchen|bathroom|bedroom|room|city|town|building|mountain|park|beach|forest|field|station|intersection|airport|harbor|restaurant|market|bridge|sidewalk|skyline|landscape|lake|river|countryside|valley)s?\b',re.I)
SEED='second-look-bird-extremes-v1'; TLS=threading.local()
def sha(data): return hashlib.sha256(data).hexdigest()
def rank(key): return sha(f'{SEED}:{key}'.encode())
def dump(path,rows): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def load(path): return [json.loads(line) for line in path.read_text().splitlines()]
def classify(labels,caps,visible):
    mentions=sum(bool(BIRD.search(c)) for c in set(caps))
    if not caps: return None
    if 'bird' in labels:
        return 'bird' if 'bird' in visible and mentions>=2 and not any(ART.search(c) for c in caps) else None
    if mentions: return None
    if visible & ANIMALS: return 'nonbird_animals'
    if labels & ANIMALS: return None
    if 'person' in visible: return 'people'
    if 'person' not in labels and any(PLACE.search(c) for c in caps): return 'places'
    return None

def features(im):
    rgb=np.asarray(im.convert('RGB').resize((96,96),Image.Resampling.BILINEAR),dtype=np.float64)/255
    r,g,b=rgb.transpose(2,0,1); y=.299*r+.587*g+.114*b
    dx=np.diff(y,axis=1,append=y[:,-1:]);dy=np.diff(y,axis=0,append=y[-1:,:]); edge=np.hypot(dx,dy)
    pos=(np.arange(96)+.5)/96; near=np.minimum(abs(pos-1/3),abs(pos-2/3)); weight=np.exp(-np.minimum(near[:,None],near[None,:])**2/(2*.07**2))
    return dict(strong=float((np.percentile(y,90)-np.percentile(y,10))*(edge*weight).mean()),bright=float(y.mean()),saturated=float(((rgb.max(2)-rgb.min(2))/np.maximum(rgb.max(2),1/255)).mean()),warm=float((r-b).mean()),detailed=float(edge.mean()),contrasty=float(y.std()),busy=float((edge>.1).mean()))

def candidates():
    out=WORK/'candidates.jsonl'; WORK.mkdir(parents=True,exist_ok=True)
    if out.exists(): return load(out)
    assert sha(ARCH.read_bytes())==ARCHIVE_SHA
    exclude={r['id'] for r in load(Path('datasets/legacy-coco/manifest.jsonl'))}; result=[]
    with ZipFile(ARCH) as z:
        for split in ('train2017','val2017'):
            def items(key,kind='instances'):
                with z.open(f'annotations/{kind}_{split}.json') as f: yield from ijson.items(f,key+'.item',use_float=True)
            cats={c['id']:c['name'] for c in items('categories')}; lic={c['id']:c for c in items('licenses') if 1<=c['id']<=6}
            imgs={i['id']:i for i in items('images') if i['id'] not in exclude and min(i['width'],i['height'])>=256 and i['license'] in lic}
            labels=defaultdict(set); vis=defaultdict(set); caps=defaultdict(list)
            for a in items('annotations'):
                id=a['image_id']; i=imgs.get(id)
                if i:
                    cat=cats[a['category_id']];labels[id].add(cat)
                    if not a['iscrowd'] and a['area']>=.01*i['width']*i['height']:vis[id].add(cat)
            for a in items('annotations','captions'):
                if a['image_id'] in imgs:caps[a['image_id']].append((a['id'],a['caption']))
            for id,i in imgs.items():
                cat=classify(labels[id],[c for _,c in caps[id]],vis[id]);
                if cat:
                    result.append(dict(id=id,category=cat,width=i['width'],height=i['height'],url=f"{BASE}/{split}/{i['file_name']}",license=lic[i['license']],flickr_url=i['flickr_url'],group='flickr-'+i['flickr_url'].split('/')[-1].split('_')[0],categories=sorted(labels[id]),visible=sorted(vis[id]),captions=caps[id]))
    result.sort(key=lambda r:rank(r['id'])); short=[]
    for cat,n in TARGET.items():short.extend([r for r in result if r['category']==cat][:n+300]);
    tmp=out.with_suffix('.tmp');dump(tmp,short);tmp.replace(out);print('candidates',Counter(r['category'] for r in short),flush=True);return short

def get(row):
    p=WORK/'cache'/f"{row['id']:012}.jpg"; p.parent.mkdir(exist_ok=True); meta=p.with_suffix('.json');tmp=p.with_suffix('.part')
    if p.exists() and meta.exists(): return json.loads(meta.read_text())
    for attempt in range(5):
        try:
            if not p.exists():
                if not hasattr(TLS,'session'):TLS.session=requests.Session()
                with TLS.session.get(row['url'],stream=True,timeout=(10,35)) as resp:
                    resp.raise_for_status()
                    with tmp.open('wb') as f:
                        for chunk in resp.iter_content(65536):f.write(chunk)
            fpath=p if p.exists() else tmp
            with Image.open(fpath) as im:
                if im.format!='JPEG' or im.size!=(row['width'],row['height']):raise ValueError('dimensions/format')
                rgb=im.convert('RGB'); a=np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS)); dh=np.packbits(a[:,1:]>a[:,:-1]).tobytes().hex()
                m=dict(sha256=sha(fpath.read_bytes()),rgb_sha256=sha(rgb.tobytes()),dhash=dh,proxy=features(rgb))
            if fpath==tmp:tmp.replace(p)
            t=meta.with_suffix('.json.tmp');t.write_text(json.dumps(m));t.replace(meta);return m
        except (requests.RequestException,OSError,ValueError) as e:
            if attempt==4:raise RuntimeError(row['url']) from e
            time.sleep(2**attempt)

def acquire():
    rows=candidates(); chosen=[]; seen=set(); hashes=[]
    with ThreadPoolExecutor(24) as pool:
        for cat,n in TARGET.items():
            rows_cat=[r for r in rows if r['category']==cat]; count=0
            for i in range(0,len(rows_cat),96):
                batch=rows_cat[i:i+96]
                for row,m in zip(batch,pool.map(get,batch)):
                    keys={str(row['id']),row['group'],m['sha256'],m['rgb_sha256']};h=int(m['dhash'],16)
                    if seen & keys or any((h^v).bit_count()<=5 for v in hashes):continue
                    seen.update(keys);hashes.append(h);chosen.append({**row,**m});count+=1
                    if count==n:break
                print(cat,count,'/',n,flush=True)
                if count==n:break
            if count!=n:raise RuntimeError(f'insufficient {cat}: {count}')
    tmp=WORK/'selected.tmp';dump(tmp,chosen);tmp.replace(WORK/'selected.jsonl')
if __name__=='__main__':acquire()
