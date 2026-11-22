#!/usr/bin/env bash
# Launch from q38-code with setsid, stdin detached, log redirected by caller.
set -eu
cd /home/ubuntu/second-look/work/q38-code
w=work/overnight-002
exec 9>work/overnight.lock
flock -n 9 || exit 75
test ! -e work/STOP-OVERNIGHT && test ! -e "$w/status" || exit 76
printf '%s\n' "$$" > "$w/pid"
trap 'rc=$?; printf "exit %d\n" "$rc" > "$w/status"' EXIT
trap 'exit 143' TERM
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$busy"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for stage in smoke before train after; do
  test ! -e work/STOP-OVERNIGHT
  printf '%s\n' "$stage" > "$w/status"
  timeout --foreground -k 60 3h /home/ubuntu/second-look/.venv-q38/bin/python "$w/train_q38.py" "$stage" --balanced-dev --output outputs/overnight-002
done
