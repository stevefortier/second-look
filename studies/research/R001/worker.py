import json,os,time
from pathlib import Path as P
import extreme_run as e

def stop():
    if any(P(p).exists() for p in ('STOP_RESEARCH','../STOP_RESEARCH','../../STOP_RESEARCH','../../studies/research/STOP_RESEARCH','../research/STOP_RESEARCH')):raise SystemExit(75)

def main():
    import torch,transformers
    stop(); rows=json.loads(P('panel.json').read_text()); assert len(rows)==100
    assert all(e.sha(P('input')/r['image'])==r['sha256'] for r in rows)
    out=P('raw');out.mkdir(exist_ok=False); e.status('loading',total=400,done=0);m,proc=e.model();m.eval();kw=dict(do_sample=True,temperature=.85,top_p=.92,top_k=50,num_beams=1,max_new_tokens=112)
    e.dump('environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),model=e.MODEL,revision=e.REV,overrides=kw,base_generation_config=m.generation_config.to_dict(),prompt=e.FREE,system=e.SYSTEM))
    with torch.inference_mode():
        for i,r in enumerate(rows):
            for j in range(4):
                stop();n=4*i+j; seed=100100+n;torch.manual_seed(seed);x=e.batch(proc,[(str(P('input')/r['image']),e.FREE,None)]);t=time.time();seq=m.generate(**x,**kw)[0,x.input_ids.shape[1]:].tolist();text=proc.tokenizer.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                e.dump(out/f'{n:03}.json',dict(i=i,j=j,seed=seed,sha256=r['sha256'],ids=seq,text=text,seconds=time.time()-t));e.status('generating',total=400,done=n+1)
    assert len(list(out.glob('*.json')))==400;e.status('complete',total=400,done=400)
if __name__=='__main__':main()
