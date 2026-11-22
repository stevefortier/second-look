"""Isolated synthetic GPU integration test. Never evidence about actual photos."""
import os, sys
from pathlib import Path
import extreme_run as e
from PIL import Image
assert Path.cwd().name=='preflight'; phase=sys.argv[1]
fit=[]
for i in range(4):
    p=Path(f'fixture-{i}.jpg'); Image.new('RGB',(256+32*i,288),(i*61,20,150)).save(p); fit.append(dict(path=str(p),scores={a:1 if i%2 else -1 for a in e.AXES}))
m,proc=e.model(); order=[(0,'prose'),(1,'score'),(2,'score'),(3,'prose')]; n=2 if phase=='first' else 4
if phase=='first': assert not e.checkpoints()
else: assert e.checkpoints()[-1].name=='ckpt-0002'
e.train(m,proc,fit[:n],order[:n],'fixture'); print(phase,'synthetic samples',n,flush=True)
