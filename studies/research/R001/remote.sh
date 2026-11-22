#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/ubuntu/second-look/work/research-R001
exec 9>RUN.lock; flock -n 9 || exit 73
[[ ! -e worker.pid ]] || exit 73
printf '%s\n' "$$" >worker.pid
trap 'rc=$?; printf "%s exit %s\n" "$(date -u +%FT%TZ)" "$rc" >worker.state' EXIT
[[ ! -e STOP_RESEARCH && ! -e ../STOP_RESEARCH && ! -e ../../STOP_RESEARCH && ! -e ../research/STOP_RESEARCH && ! -e ../../studies/research/STOP_RESEARCH ]] || exit 75
sha256sum --check --quiet FREEZE.sha256
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA RTX A6000' && -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 73
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
printf 'running\n' >worker.state
timeout -k 30 6h /home/ubuntu/second-look/.venv-q38/bin/python worker.py
sha256sum FREEZE.sha256 environment.json status.json raw/*.json >FINAL.sha256.tmp; mv FINAL.sha256.tmp FINAL.sha256
