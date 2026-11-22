#!/usr/bin/env bash
# From a clean checkout: bash scripts/reproduce-birdwatcher.sh [data|train]
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-train}; [[ $mode == train || $mode == data ]] || { echo 'Usage: reproduce-birdwatcher.sh [data|train]' >&2; exit 2; }
py=${PYTHON:-python3}; envdir=${BIRD_VENV:-.venv-birdwatcher}; out=${BIRD_RUN:-work/birdwatcher-reproduction}
[[ ! -e $out ]] || { echo "Choose a new BIRD_RUN; refusing to overwrite $out" >&2; exit 1; }
[[ -x $envdir/bin/python ]] || "$py" -m venv "$envdir"
py=$envdir/bin/python
"$py" -m pip install -q 'Pillow==12.3.0' 'numpy>=1.21,<3'
"$py" scripts/restore_avian.py --output "$out/data"
[[ $mode == train ]] || exit 0
if ! "$py" -c 'import torch,torchvision; assert torch.__version__.split("+")[0]=="2.7.0" and torchvision.__version__.split("+")[0]=="0.22.0" and torch.cuda.is_available()' 2>/dev/null; then
  "$py" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
fi
"$py" -m pip install 'transformers==4.57.3' 'peft==0.18.0' 'accelerate==1.15.0' 'huggingface-hub==0.36.2' 'Jinja2==3.1.6' 'PyYAML>=5.4,<7'
export PYTHONPATH="$PWD" USE_TF=0 OMP_NUM_THREADS=2
"$py" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA GPU with BF16 required";print(torch.cuda.get_device_name())'
base=$("$py" -c 'from huggingface_hub import snapshot_download;print(snapshot_download("Qwen/Qwen3-VL-2B-Instruct",revision="89644892e4d85e24eaac8bacfd4f463576704203",allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.model","*.tiktoken"]))')
export HF_HUB_OFFLINE=1
"$py" scripts/train_avian.py "$out/data/avian" "$out/visual"
winner=$("$py" -c 'import json,sys;print(json.load(open(sys.argv[1]))["winner"])' "$out/visual/selection.json")
"$py" scripts/avian_voice.py train --device cuda --model "$base" --base-id Qwen/Qwen3-VL-2B-Instruct --adapter "$out/visual/$winner" --steps 120 --lr 0.00009 --output "$out/birdwatcher-v2"
"$py" scripts/avian_voice.py compare --device cuda --model "$base" --adapter "$out/birdwatcher-v2" --output "$out/voice.json"
"$py" scripts/avian_exam.py "$out/data/avian-final" "$base" "$out/birdwatcher-v2" "$out/official.json"
"$py" - "$out" "$base" <<'PY'
import json,sys
from pathlib import Path
from second_look.model import evaluate
out=Path(sys.argv[1]); root=out/'data/avian-final'; rows=[json.loads(s) for s in (root/'ratings.jsonl').read_text().splitlines()];plan={'test':[], 'singles':[{'image':x['image'],'bank':x['image'].split('/')[0]} for x in rows]};(out/'singles.json').write_text(json.dumps(evaluate(root,plan,out/'birdwatcher-v2',sys.argv[2],'cuda'),indent=2))
PY
printf 'Adapter and reports: %s\n' "$out"
