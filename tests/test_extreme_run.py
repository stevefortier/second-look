from scripts import extreme_run as e


def rows(n):
    return [dict(image=f'{c}/{i}.jpg', group=f'{c}{i}', scores={a: (1 if c=='liked' else -1) for a in e.AXES}) for c in ['liked','control'] for i in range(n)]


def test_plan_deterministic_disjoint_tasks():
    fit=rows(60); held=rows(56)
    order,panel,free=e.plan(fit,held)
    assert sorted(x[0] for x in order)==list(range(len(fit))) and order!=sorted(order)
    assert {k for i,k in order}=={'score','prose'}
    assert len(set(panel))==100 and len(set(free))==32 and set(free)<=set(panel)
    assert sum(e.bird(held[i]) for i in panel)==50
    assert sum(e.bird(held[i]) for i in free)==16
    assert e.plan(fit,held)==(order,panel,free)

def test_paired_report_counts_changes_and_majority():
    rows=[dict(image=str(i),axis='busy',bird=bool(i),truth=t,base=b,adapted=a,majority=1) for i,t,b,a in [(0,-1,1,-1),(1,1,1,-1)]]
    result=e.metrics(rows)['busy']['all']; assert result['gains']==result['losses']==1 and result['fit_majority_accuracy']==0.5

def test_blind_packet_has_no_condition_class_or_targets():
    rs=[dict(photo='17',condition=c,text=c+' reply',image='bird.jpg',bird=True) for c in ['base','adapted']]
    packet,key=e.blind_packet(rs)
    assert len(packet)==len(key)==2 and all(set(r)=={'id','photo','text'} for r in packet)
    assert {r['id'] for r in packet}=={r['id'] for r in key} and {r['condition'] for r in key}=={'base','adapted'}

def test_literal_blind_rubric_uses_abstentions_not_negatives():
    assert e.literal('The composition does not look beautiful.')['beautiful']==-1
    assert e.literal('A bird on a branch.')['beautiful'] is None
    assert e.literal('Impressive yet unimpressive.')['impressive'] is None
    assert e.literal('The arrangement is striking.')['impressive']==1

def test_compact_score_tasks_use_every_supplied_axis_and_preserve_nulls():
    r=rows(1)[0]; r['scores']['warm']=None
    p,y=e.example(r,'score',0)
    assert all(a in p for a in e.AXES) and y.split()==['C']*6+['?']+['C']*3
    assert e.example(r,'score',1)[0]==e.question('important')

def test_report_artifacts_roundtrip(tmp_path,monkeypatch):
    import json, subprocess
    from PIL import Image
    monkeypatch.chdir(tmp_path); fit=rows(2); held=rows(1); root=tmp_path/'artifacts'; ck=root/'ckpt-4000'; ck.mkdir(parents=True)
    (ck/'adapter_model.safetensors').write_bytes(b'fixture-not-weights'); (ck/'state.pt').touch()
    for c in ['base','adapted']:
        (root/c).mkdir()
        for i,r in enumerate(held):
            r['path']=str(tmp_path/f'{i}.png'); Image.new('RGB',(8,8)).save(r['path']); scores=[dict(axis=a,truth=t,pred=t if c=='adapted' else -t) for a,t in r['scores'].items()]; e.dump(root/c/f'{i:03}.json',dict(index=i,image=r['image'],bird=e.bird(r),scores=scores,spontaneous='The photo is warm.'))
    e.finish(fit,held,[0,1],[0,1]); report=json.loads((root/'report.json').read_text()); assert report['per_axis_and_class']['ALL']['all']['accuracy_change']==1
    assert len(json.loads((root/'blind/replies.json').read_text()))==4 and (root/'spontaneous-literal-report.json').exists() and not (root/'adapter/state.pt').exists()
    subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=root,check=True)
