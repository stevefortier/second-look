#!/usr/bin/env bash
# Local operator entry point. No SSH, writes or model work without --submit.
set -euo pipefail
[[ $# == 1 && $1 == --submit ]] || { printf 'Usage: bash %s --submit\n' "$0" >&2; exit 64; }
cd /home/scratch/Workspaces/second-look
study=studies/overnight/028
[[ ! -e work/STOP-OVERNIGHT ]]
.venv/bin/python "$study/panel.py"
.venv/bin/python scripts/q38_quality028.py check >/dev/null
manifest=$(sha256sum "$study/artifacts.sha256" | cut -d' ' -f1)
host=ubuntu@68.209.107.253
remote=/home/ubuntu/second-look/work/q38-code
ssh_opts=(-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=.secrets/lambda-known-hosts -i .secrets/lambda-a6000)
remote_cmd() { ssh "${ssh_opts[@]}" "$host" "cd $remote && $1"; }
guards='[[ ! -e work/STOP-OVERNIGHT && ! -e work/overnight-028 && ! -e outputs/overnight-028 && $(<work/overnight-027/status) == "exit 0 complete" ]]'
remote_cmd "$guards && flock -n -E 75 work/overnight.lock true"
# Ship only new frozen sources and selected fit JPEGs; never overwrite remote inputs.
rsync -a --ignore-existing --files-from="$study/upload.txt" -e "ssh ${ssh_opts[*]}" ./ "$host:$remote/"
remote_cmd "set -euo pipefail; $guards; printf '%s  %s\n' '$manifest' '$study/artifacts.sha256' | sha256sum -c -; sha256sum --strict -c $study/artifacts.sha256; bash -n scripts/q38-028.sh; /home/ubuntu/second-look/.venv-q38/bin/python scripts/q38_quality028.py check >/dev/null"
[[ ! -e work/STOP-OVERNIGHT ]]
remote_cmd "set -euo pipefail; $guards; mkdir work/overnight-028; setsid flock -n -E 75 work/overnight.lock bash scripts/q38-028.sh </dev/null >work/overnight-028/run.log 2>&1 &"
# Read back this exact claim; if inconclusive, inspect it; never resubmit automatically.
remote_cmd 'for i in {1..15}; do if [[ -d work/overnight-028/.claimed && -s work/overnight-028/status ]]; then s=$(<work/overnight-028/status); printf "028: %s\n" "$s"; [[ "$s" != exit* || "$s" == "exit 0 complete" ]]; exit $?; fi; sleep 1; done; exit 75'
