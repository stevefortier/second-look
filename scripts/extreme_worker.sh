#!/usr/bin/env bash
# Remote worker; run under nohup on this one instance only.
set -Eeuo pipefail
root=/home/ubuntu/second-look/work/extremes-run
cd "$root"
exec 9>RUN.lock
flock -n 9 || { printf 'already locked\n' >&2; exit 73; }
[[ ! -f artifacts/SHA256SUMS ]] || { printf 'already complete\n'; exit 0; }
printf '%s\n' "$$" >worker.pid
mark() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >worker.state.tmp; mv worker.state.tmp worker.state; }
trap 'rc=$?; mark "exit $rc"' EXIT
mark waiting-for-DATA_READY
for ((n=0;n<5760;n++)); do [[ ! -e STOP_EXTREMES ]] || exit 75; [[ ! -f DATA_READY ]] || break; sleep 15; done
[[ -f DATA_READY && -e preflight/PASS ]] || exit 74
# No queueing behind, killing or co-running with another GPU job.
clients=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [[ -z "$clients" ]] || { printf 'GPU busy: %s\n' "$clients"; exit 73; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
for phase in plan train base adapted report; do
  [[ ! -e STOP_EXTREMES ]] || exit 75
  mark "$phase"; /home/ubuntu/second-look/.venv-q38/bin/python code/extreme_run.py "$phase"
done
(cd artifacts && sha256sum --check --quiet SHA256SUMS)
