#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/second-look/work/q38-code
out=work/overnight-010
py=/home/ubuntu/second-look/.venv-q38/bin/python
# Parent is setsid flock (no --no-fork); bind lock proof to our session.
[[ $(ps -o sid= -p $$ | tr -d ' ') == "$PPID" ]]
held=no
for fd in /proc/$PPID/fd/*; do
  if [[ $fd -ef work/overnight.lock ]] && grep -Eq '^lock:.*FLOCK +ADVISORY +WRITE' /proc/$PPID/fdinfo/${fd##*/}; then held=yes; fi
done
[[ $held == yes ]] || exit 75
[[ ! -e work/STOP-OVERNIGHT && ! -e $out/status ]]
mkdir "$out/.claimed"
stage=preflight
status() { printf '%s\n' "$*" > "$out/status.tmp"; mv "$out/status.tmp" "$out/status"; }
trap 'rc=$?; status "exit $rc $stage"' EXIT
trap 'exit 143' HUP TERM
status "$stage"; printf '%s\n' "$$" > "$out/pid"
[[ $(timeout 15 nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA RTX A6000' ]]
busy=$(timeout 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader); [[ -z $busy ]]
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
for stage in before train after; do
  [[ ! -e work/STOP-OVERNIGHT ]] || exit 76
  status "$stage"; printf '%s %s\n' "$(date -u +%FT%TZ)" "$stage"
  timeout --foreground -k 20s 45m "$py" scripts/q38_quality010.py "$stage" --bank work/quality008/bank/train --study studies/overnight/008 --output "$out"
done
stage=complete
