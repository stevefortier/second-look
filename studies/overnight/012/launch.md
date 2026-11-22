# 012 operator commands — prepared only; not a launch receipt

Run from the local project in Bash **only when authorized**. Sync only new 012 files. No weights, inputs, shared trainers, or 011 files are transferred. The remote trainer hash is the prior run's remote pin, not the local historical trainer. Do not replace the trainer if a pin fails. Local reproduction: `.venv/bin/python scripts/q38_quality012.py check` and `.venv/bin/python -m pytest -q tests/test_quality012.py`.

```bash
cd /home/scratch/Workspaces/second-look
set -euo pipefail
host=ubuntu@68.209.107.253
base=/home/ubuntu/second-look/work/q38-code
transport='ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=.secrets/lambda-known-hosts -i .secrets/lambda-a6000'
[[ ! -e work/STOP-OVERNIGHT ]]
rsync -aR --ignore-existing -e "$transport" scripts/q38-012.sh scripts/q38_quality012.py scripts/q38_eval012.py tests/test_quality012.py studies/overnight/012/ "$host:$base/"
manifest=$(sha256sum studies/overnight/012/artifacts.sha256 | cut -d' ' -f1)
$transport "$host" "cd '$base' && test ! -e work/STOP-OVERNIGHT && printf '%s  %s\n' '$manifest' studies/overnight/012/artifacts.sha256 | sha256sum -c - && sha256sum --strict -c studies/overnight/012/artifacts.sha256 && /home/ubuntu/second-look/.venv-q38/bin/python scripts/q38_quality012.py check"
```

Separate launch, once, only after previous completion and operator approval. If any guard fails, stop; never remove a claim or retry automatically.

```bash
[[ ! -e work/STOP-OVERNIGHT ]]
$transport "$host" "cd '$base' && bash -s" <<'REMOTE'
set -euo pipefail
[[ ! -e work/STOP-OVERNIGHT && $(<work/overnight-011/status) == 'exit 0 complete' ]]
mkdir work/overnight-012
nohup setsid flock -n -E 75 work/overnight.lock bash scripts/q38-012.sh </dev/null >work/overnight-012/run.log 2>&1 &
REMOTE
```

Read back `work/overnight-012/status`, `pid`, process/session group, kernel flock and `nvidia-smi` before claiming a launch. The worker fails closed on a busy or wrong GPU. It checks hashes, trains, and reloads the adapter in a fresh process using the established runner; paired before/after dev evaluation. Completion is `exit 0 complete`, not merely a successful SSH return.
