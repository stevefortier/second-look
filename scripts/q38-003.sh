#!/usr/bin/env bash
# Worker only: caller supplies setsid, project flock, stdin and launcher log redirection.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  printf '%s\n' 'From the pinned remote project root (create work/overnight-003 first):' \
    'setsid flock -n -E 75 work/overnight.lock bash scripts/q38-003.sh </dev/null >>work/overnight-003/log 2>&1 &'
  exit 0
fi
[[ $# == 0 || ( $# == 1 && $1 =~ ^00[456]$ ) ]] || exit 64
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
out=work/overnight-${1:-003}
data=003
if [[ ${1:-} == 005 || ${1:-} == 006 ]]; then data=005; fi
runner=scripts/q38_quality${1:-}.py
python=/home/ubuntu/second-look/.venv-q38/bin/python
# Reject accidental direct invocation. Linux fdinfo records an inherited flock's
# open-file description, including when the flock process itself retains the fd.
lock_held() {
  local pid=$$ fd parent
  while [[ $pid =~ ^[0-9]+$ && $pid -gt 1 ]]; do
    for fd in /proc/"$pid"/fd/*; do
      if [[ $fd -ef work/overnight.lock ]] && grep -Eq '^lock:.*FLOCK +ADVISORY +WRITE' "/proc/$pid/fdinfo/${fd##*/}"; then return 0; fi
    done
    parent=$(ps -o ppid= -p "$pid") || return 1
    pid=${parent//[[:space:]]/}
  done
  return 1
}
lock_held || { printf '%s\n' 'missing external project flock' >&2; exit 75; }
mkdir -p "$out"
[[ ! -e work/STOP-OVERNIGHT && ! -e $out/status ]] || exit 76
mkdir "$out/.claimed" || exit 76
stage=preflight
status() { printf '%s\n' "$*" >"$out/status.tmp"; mv -f "$out/status.tmp" "$out/status"; }
trap 'rc=$?; status "exit $rc $stage"' EXIT
trap 'exit 143' TERM HUP
printf '%s\n' "$$" >"$out/pid"
status "$stage"
[[ -x $python ]]
busy=$(timeout 30 nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z $busy ]] || { printf 'GPU busy: %s\n' "$busy" >&2; exit 77; }
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for stage in before train after; do
  [[ ! -e work/STOP-OVERNIGHT ]] || exit 76
  status "$stage"
  printf '%s starting %s\n' "$(date -u +%FT%TZ)" "$stage"
  timeout --foreground -k 60s 3h "$python" "$runner" "$stage" \
    --bank work/quality"$data" --study studies/overnight/"$data" --output "$out"
done
stage=complete
