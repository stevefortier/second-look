#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/scratch/Workspaces/second-look
r=/home/ubuntu/second-look/work/extremes-run
host=ubuntu@68.209.107.253
sshargs=(-o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=.secrets/lambda-known-hosts -i .secrets/lambda-a6000)
s() { ssh "${sshargs[@]}" "$host" "$@"; }
export RSYNC_RSH="ssh ${sshargs[*]}"
case "${1:-status}" in
  stage) s "mkdir -p $r/code; cd $r && test ! -e artifacts/protocol.json"; rsync -ac scripts/extreme_{run.py,worker.sh,preflight.py} "$host:$r/code/"; (cd scripts && sha256sum extreme_{run.py,worker.sh,preflight.py}) | s "cd $r/code && sha256sum -c -";;
  launch) s "cd $r && test ! -e STOP_EXTREMES && test -e preflight/PASS && { nohup setsid bash code/extreme_worker.sh >>run.log 2>&1 </dev/null & }";;
  status) s "cd $r && for f in worker.pid worker.state status.json; do if test -f \"\$f\"; then cat \"\$f\"; fi; done; if flock -n RUN.lock true; then printf 'run lock free\\n'; else printf 'run lock held\\n'; fi; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; if test -f run.log; then tail -c 1800 run.log; fi";;
  stop) s "cd $r && touch STOP_EXTREMES && test -f STOP_EXTREMES";;
  backup) dest=work/extremes-offinstance; mkdir -p "$dest"; s "cd $r/artifacts && sha256sum --check --quiet SHA256SUMS"; rsync -ac --exclude='ckpt-*' --exclude='ckpt.tmp' "$host:$r/artifacts/" "$dest/"; (cd "$dest" && sha256sum --check --quiet SHA256SUMS && printf 'Verified final weights, protocol, records, blind packet and reports.\n');;
  *) printf 'usage: bash scripts/extreme_gpu.sh stage|launch|status|stop|backup\n' >&2; exit 2;;
esac
