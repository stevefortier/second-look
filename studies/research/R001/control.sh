#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/scratch/Workspaces/second-look
D=$PWD/studies/research/R001; R=/home/ubuntu/second-look/work/research-R001; H=ubuntu@68.209.107.253
args=(-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$PWD/.secrets/lambda-known-hosts" -i "$PWD/.secrets/lambda-a6000"); export RSYNC_RSH="ssh ${args[*]}"; s(){ ssh "${args[@]}" "$H" "$@"; }
stop_local(){ [[ -e STOP_RESEARCH || -e work/STOP_RESEARCH || -e work/research/STOP_RESEARCH || -e studies/research/STOP_RESEARCH || -e "$D/STOP_RESEARCH" ]]; }
if stop_local; then s "test ! -d $R || touch $R/STOP_RESEARCH"; fi
case ${1:-status} in
stage) ! stop_local || exit 75; s "mkdir $R"; rsync -ac "$D"/{FREEZE.sha256,PROTOCOL.md,worker.py,remote.sh,extreme_run.py,panel.json,input} "$H:$R/"; s "cd $R && sha256sum -c --quiet FREEZE.sha256";;
launch) ! stop_local || exit 75; s "cd $R && test ! -e worker.pid && test ! -e STOP_RESEARCH && { nohup setsid bash remote.sh >>run.log 2>&1 </dev/null & }";;
status) s "cd $R && for f in worker.pid worker.state status.json; do if test -f \$f; then cat \$f; fi; done; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; tail -c 1000 run.log";;
stop) touch "$D/STOP_RESEARCH"; s "cd $R && touch STOP_RESEARCH && test -e STOP_RESEARCH";;
backup) mkdir -p "$D/backup"; s "cd $R && if test -e FINAL.sha256; then cat FINAL.sha256; else sha256sum FREEZE.sha256 environment.json raw/[0-9][0-9][0-9].json; fi" >"$D/backup/index.tmp"; (cd "$D/backup"; cut -c67- index.tmp | rsync -ac --files-from=- "$H:$R/" ./; sha256sum -c --quiet index.tmp; mv index.tmp SNAPSHOT.sha256; wc -l SNAPSHOT.sha256);;
*) exit 2;;esac
