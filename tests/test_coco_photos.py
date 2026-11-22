from scripts import coco_photos as photos
import hashlib
import io
import json
import zipfile
from PIL import Image
import pytest


def test_eligibility_and_aspect_limits():
    assert [photos.bucket(w, h) for w, h in [(255,640),(256,640),(400,500),(625,500),(640,400)]] == [None,'portrait','near-square','near-square','landscape']


def test_selection_matches_buckets_and_is_order_invariant():
    rows = [dict(id=10*i+j, width=w, height=h, bank=bank) for i,(w,h) in
            enumerate([(256,640),(400,400),(640,400)]) for j,bank in enumerate(['liked','control','liked','control'])]
    pairs = photos.select(rows, 4)
    assert pairs == photos.select(rows[::-1], 4)
    assert len(pairs) == 4
    assert len({s['id'] for pair in pairs for s in pair}) == 8
    assert {photos.bucket(p[0]['width'],p[0]['height']) for p in pairs} == {'portrait','near-square','landscape'}
    for bird, control in pairs:
        assert (bird['bank'],control['bank']) == ('liked','control')
        assert (bird['width'],bird['height']) == (control['width'],control['height'])


def test_category_selection_caption_join_and_source_license(tmp_path):
    pytest.importorskip('ijson')  # needed only for rebuilding selection, never restore
    split = 'val2017'
    images = [dict(width=400,height=400,license=4,id=n,file_name=f'{n:012}.jpg',flickr_url=f'https://example.com/{n}_x.jpg') for n in range(1,5)]
    instances = dict(images=images,licenses=[dict(id=4,name='Attribution License',url='http://creativecommons.org/licenses/by/2.0/')],categories=[dict(id=i,name=n,supercategory='animal') for i,n in [(16,'bird'),(17,'cat')]], annotations=[dict(image_id=i,category_id=c,iscrowd=0,area=9000) for i,c in [(1,16),(2,17),(3,16),(3,17),(4,17)]])
    images[3]['height'] = 255
    with zipfile.ZipFile(tmp_path/'a.zip','w') as z:
        z.writestr(f'annotations/instances_{split}.json',json.dumps(instances))
        z.writestr(f'annotations/captions_{split}.json',json.dumps(dict(annotations=[dict(id=90+n,image_id=n,caption='example') for n in range(1,5)])))
    with zipfile.ZipFile(tmp_path/'a.zip') as z:
        rows = photos.source_rows(z, split, 1)
    assert [r['id'] for r in rows] == [1,2]
    assert rows[0]['caption_ids'] == [91]
    assert rows[1]['license'] == instances['licenses'][0]
    assert rows[1]['categories'] == ['cat']


def test_original_bytes_and_hash_pin(tmp_path):
    stream = io.BytesIO()
    Image.new('RGB',(320,256)).save(stream,format='JPEG')
    data = stream.getvalue()
    row = dict(width=320,height=256)
    pins = photos.identify(data,row)
    assert pins['sha256'] == hashlib.sha256(data).hexdigest()
    assert pins['bytes'] == len(data)
    assert photos.identify(data,{**row,**pins}) == pins
    with pytest.raises(ValueError,match='sha256'):
        photos.identify(data+b'changed',{**row,**pins})
    with pytest.raises(ValueError,match='dimensions'):
        photos.identify(data,dict(width=321,height=256))


def test_materialize_then_offline_verify_and_reject_modified_metadata(tmp_path):
    rows, payload = [], {}
    for n,bank in enumerate(['liked','control']):
        s = io.BytesIO()
        Image.new('RGB',(320,320),(n*30,0,0)).save(s,format='JPEG')
        url = f'{photos.BASE}/train2017/{n:012}.jpg'
        payload[url] = s.getvalue()
        rows.append(dict(id=n,bank=bank,image=f'{bank}/{n:012}.jpg',width=320,height=320,
                         aspect='near-square',pair=0,split='train2017',url=url))
    pinned = photos.materialize(rows,tmp_path,fetch=payload.__getitem__)
    photos.materialize(pinned,tmp_path,verify=True)
    root = tmp_path/'train'
    assert json.loads((root/'matched_pairs.jsonl').read_text())['images'] == [r['image'] for r in rows]
    (root/'ratings.jsonl').write_text('bad')
    with pytest.raises(ValueError,match='ratings.jsonl'):
        photos.materialize(pinned,tmp_path,verify=True)
