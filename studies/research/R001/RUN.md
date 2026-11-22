# R001 supervisor handoff
Remote `/home/ubuntu/second-look/work/research-R001`, only existing single A6000. One standalone six-hour (maximum) inference worker, no automatic restart, no training, no held-out access. PID 154641 (shell), PID 154651 (inference). `PROTOCOL.md` and `FREEZE.sha256` were staged and verified before launch. Bank/photos, sampler, template, seeds, model and scoring contract are fixed. Raw token IDs and exact decoded outputs are immutable one-file-per-draw records. Fallback Torch implementation was reported at load; inference produced real text.

From `/home/scratch/Workspaces/second-look`:

```
D=studies/research/R001
bash "$D/control.sh" status
bash "$D/control.sh" backup
# Explicit operator stop (persistent; do not remove or automatically resume):
bash "$D/control.sh" stop
```

On each cron tick first run status: it propagates any local research stop marker to the remote run, including `studies/research/STOP_RESEARCH`. The remote also checks its own marker before every draw. While running, back up regularly; atomic raw files are transferred using a frozen file digest list, verified in `backup/SNAPSHOT.sha256`. Never call stage/launch again. On nonzero exit or stale progress, preserve the partial data and log; no automatic reruns or parameter changes. The stage tool refuses an existing namespace. No additional infrastructure, external inference API, or separate posting.

On `worker.state` exit 0 **and** `status.json` complete 400/400: rerun backup; from the project root, `cmp "$D/FREEZE.sha256" "$D/backup/FREEZE.sha256"`; `(cd "$D" && sha256sum -c --quiet FREEZE.sha256)` and `(cd "$D/backup" && sha256sum -c --quiet SNAPSHOT.sha256)`. The final snapshot includes final status, environment, freeze digest and 400 raw records. Use `.venv/bin/python "$D/packet.py"` once; it refuses incomplete data or an existing packet. Independently score the 800 shuffled `review/T*/` items under the frozen rubric, passing only one item's image and response and neutral scoring rubric to the reviewer. Do not disclose the hypothesis, hidden key, class proxies, sibling replies or filenames. Keep `private-key.json` and `panel.json` outside review context. Hash and archive all judgments before joining; apply `PROTOCOL.md` exactly, report fixed-first vs oracle separately and write PDF later. Materials/results stay here. No test, training, tuning, or model promotion.
