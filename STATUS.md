# Second Look — handoff

## Current overnight: 008 not promoted; 009 running

008 exited 0; rsync checksum comparison empty. Independent frozen review of all 48 neutral responses on eight reused dev images found 16→12 correct spontaneous assignments / 186 known prompt-fields, one gain and five losses; grounding flags 7→10 / 24. Prompted accuracy 38→42 / 62. Proxy labels only; no independent final. Reproducible raw results and per-axis diagnostics: `studies/overnight/008/results/`. Local suite: 55 tests and 19 subtests passed.

009 holds labels, photos and 141 updates fixed and changes the caption/ordinal mix to 96/45. Verified remote `before` stage, flock group 39716, worker 39718, sole A6000 client 39738. Next tick check STOP and `work/overnight-009/status`; do not relaunch. After exit checksum-backup, freeze independent anonymous photo/response judgments, then join and publish. No 009 result yet.

## Historical overnight: 007 not promoted; 008 running

007 exited 0; checksum backup matched. Across the eight within-run dev photos, three neutral prompts and 62 known fields, spontaneous correct counts were 14/186 for both base and adapter. Three paired gains offset three losses; grounding flags increased 5→10/24. Explicit accuracy improved 38→45/62. Frozen review, raw records, per-axis coverage, and reproducible metrics: `studies/overnight/007/results/`. Local suite: 51 passed, 19 subtests passed. No independent final evaluation.

008 uses fresh fit-only model-proxy labels and captions, not Steve’s ratings; 16 fit photos and the same eight dev photos. Pinned A6000 baseline independently verified active under project lock group 36675, worker 36677 and GPU PID 36693. Next: check STOP and `work/overnight-008/status`, do not relaunch; checksum backup and freeze blinded judgments before joining. No behavioral result yet.

## Historical overnight: 006 not promoted; 007 running

006 exited 0; rsync checksum comparison empty. Pooled blinded ratings across 12 reused dev photos: base / 005 / 006 mixed correct 3 / 3 / 2 of 21, activity 15 / 13 / 7 of 33. Explicit 006 matched base. Regenerated 006 report; suite 47 passed + 19 subtests. Reproduction and raw outputs in `studies/overnight/006`.

007 tests all eight existing example axes on 16 fit / eight within-run dev photos, proxy/illustrative labels only. Example test and historical final reserved. Remote exclusive flock group 33310, worker 33312, baseline GPU PID 33328 verified. Next tick check STOP and status; do not relaunch. After exit checksum-backup, independently blind-code all neutral responses against numbered photos, freeze then join, and publish all diagnostics. No 007 outcome yet.

## Historical overnight: 005 not promoted; 006 running

005 exited 0; checksum backup matched. Frozen review (below), joined after scoring: correct spontaneous activity 15→13/33, crowding 1→5/36, color 7→3/36; mixed endpoint agreement 2→0/21. Raw outputs and reproducible reports: `studies/overnight/005`. Suite: 44 passed and 19 subtests passed.

006 replaces appended anchors with grounded, non-list prose on the same 24 fit and 12 reused dev photos. Verified remote group 29897 (flock), worker 29898, before-stage Python 29910. Next: check STOP and remote stage before acting; do not relaunch. After completion checksum-backup, freeze blinded judgments, then join and publish. No 006 behavioral result yet. For a subsequent independent test, prioritize the current example's actual KVPs rather than extrapolating from these three pilot axes.

## Current input example: per-image qualities

Generic CLI integration verified: `second-look prepare datasets/example` (with `--output`) yields 24 train/8 held-out without an axis flag; offline tiny-checkpoint train/reload/evaluation exercised by the suite. 43 tests + 19 subtests passed. Endpoint-derived prose is a training template, not evidence of spontaneous attribution.

`datasets/example/` ships 24 train + 8 test original JPEGs, one `images/` folder per split and ordinary per-image `scores` across eight arbitrary axes including `liking`. No liked/control folder or image-pair requirement. Subjective example ratings are illustrative, not Steve's. Legacy COCO bird/control metadata, fixtures and restored development images live under `datasets/legacy-coco/`; historical prepared remote run roots are unchanged. No model retraining is claimed by this migration.

## 005 blind review frozen

Local review complete: all 72 distinct responses (12 photos, six responses each) scored from the sanitized packet and numbered photos in `work/overnight-005/anonymous-review/`. Frozen 004-format file: `studies/overnight/005/results/blind-judgments.json`, SHA-256 `e7c8e79a63602e67dad4a4c25d5f1e745065e178d6db57c46db77584283c7e7f`. The scorer did not open the identity key, raw before/after outputs, or 005 ratings. Next, separately join frozen judgments against proxy labels and compute literal training-anchor matches. No comparative result claimed here.

## Historical: 005 running; 004 complete, not promoted

004 exited 0; local backup matches remote by rsync checksum. Frozen blind review: activity 12→16/21, crowding 0→0/24, color 2→1/24. See `studies/overnight/004` for recipe, outputs and caveats. Recomputed reports from raw files. Workspace-wide tests during an unrelated concurrent CLI/dataset migration: 40 passed, 19 subtests passed, 2 failed (legacy review path and single-image CLI dispatch); did not alter that work.

005 adds score-derived descriptions, on the same fit set with 12 freshly reviewed nonfinal dev photos. Verified matching local/remote data digest and detached flock group 27016, worker 27018, before process 27030. Next: check STOP and status; no duplicate launch. After completion, checksum backup, blinded review including literal anchor echo, report raw results and publish. No 005 result yet.

## Historical: 004 launch; 003 complete, not promoted

003 ended `exit 0 complete`; adapter and reports backed up under local `work/overnight-003`, rsync checksum comparison empty. Blind judgments frozen before identity join. Correct spontaneous assignments base → adapter: activity 12→10/21, crowding 0→0/24, color intensity 2→1/24. Explicit color improved but not primary behavior. See published `studies/overnight/003/results` and README for raw generations, labels and diagnostics. 40 local tests + 19 subtests passed.

004 is a caption-only comparison: 72 updates from the same 24 reviewed fit captions under image-only and two neutral prompts; same eight development photos; no historical final. `scripts/q38-003.sh 004` running under exclusive flock PGID 24519 on the same A6000; verified worker 24520 and `before` Python 24533. Check STOP, status and group before acting; do not relaunch. Next: checksum backup, independent anonymized review before unblinding, compare with 003; seek fresh disjoint validation, not repeated dev claims.

## Historical: 003 launch

On the existing A6000, verified `before` running under exclusive flock PID/PGID 21982, worker 21983, project `work/overnight-003`. Check local STOP first, then remote status/log and process identity; do not relaunch. Worker proceeds to 95 supervised updates and fresh adapter evaluation. Scratch reviewed 32 nonfinal photos (24 fit, 8 dev); counts, limitations, reproduction and frozen rubric in `studies/overnight/003/README.md`. Local suite: 39 tests and 19 subtests passed. Remote assembled bank validated with identical digest `9219677e2076458f0de5a5cf3a9d03d3c91792244a2edc21b166dd7b6f318328`. Next: on completion, back up outputs/checksum, assess anonymized generations **before opening before/after raw text or identity key**, join frozen judgments, run secondary metrics, publish code, annotations and raw reports without weights or photos, verify GitHub SHA. No result or publication yet.

## Active overnight objective: visual quality attribution

Steve redirected training to spontaneous assignment of the arbitrary KVP qualities, not liking. Input schema unchanged. See `studies/overnight/QUALITY-TARGET.md`. Preference-only run 002 was explicitly stopped during training; its process group was verified gone. Do not resume or promote it. Next: multi-axis pilot data and blind unprompted vs explicitly scored baselines.

## Superseded experiment 002 (historical launch notes)

33 fit / 12 development reviewed photo pairs; split and paired-position protocol in `studies/overnight/002-*`. Remote preflight decoded the real banks and verified the reviewed split; local suite 35 passed + 19 subtests. Detached study launched on the existing A6000 (PID/PGID 19989); read remote `work/overnight-002/{status,log}` before acting. Next: collect paired development and text reports, verify artifact hashes, record verdict and publish completed iteration. No new behavioral result at launch. See protocol for scoped stop procedure.

## Qwen3.8-27B reviewed-photo adapter

Completed 2 discarded smoke updates and 90 supervised rank-8 Q/V updates on one RTX A6000 with `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Separate remote environment `.venv-q38`; earlier `.venv` untouched. Curated 45 train and 19 held-out pairs; checked unique decoded photos. Neutral bird choices rose 18/38 → 23/38, both-order bird wins 6/19 → 8/19. Explicit nonbird choices were 38/38 before and after. Eight paired text generations show caution preserved on the two caution prompts, but no clear improvement in conversational bird preference. More independent images and prompts are needed before claiming generalization. See `scripts/Q38.md` for the runbook. Outputs `outputs/example-model-1/` match `/home/ubuntu/second-look/work/q38-code/outputs/example-model-1/` by rsync checksum. Adapter SHA-256 `41fd0dc98289bd75d972efc4bb17c2087568b491b887a6fe77d0e807b50c2c7f`; fresh-model reload matched all 64 saved tensors exactly, retained sentinel choice, but one sentinel logit differed by one BF16 ULP (0.125). Project tests: 34 passed and 19 subtests passed. No commit or shutdown.

## Dataset and output layout

Active data: `datasets/example/` (train/ restored; test-images/ held out). Original Birdwatcher v2 adapter and reports are in `outputs/example-model/`. Default output reservation is atomic and uses the dataset basename; the first repeat has suffix `-1`. Restored and verified all 320 images after moving. This change does not retrain the reference model or change any image labels. Three known held-out positives depict representations rather than live birds; curate before interpreting scores.

## Current test images

The offline evaluation fixture is `datasets/example/test-images`: 40 original COCO bird JPEGs plus 40 nonbird controls. Removed the old 32×32 binaries from main; the historical v2 restoration script and records remain intact. Verified all 320 source JPEGs (240 development plus 80 held-out) and all 80 checked-in byte hashes; the test photos have a shortest edge of at least 300 pixels. Each bank has 10 portrait, 5 near-square and 25 landscape photos. Offline tests: 33 passed + 19 subtests. Next: evaluate the existing adapter on this new fixture before tuning on development photos. No new training run.

## Higher-resolution photo bank

`datasets/avian-photos` and `datasets/avian-photos-final` contain 120 and 40 aspect-matched COCO bird/nonbird animal pairs respectively. Original JPEG bytes are ignored. `datasets/example/` records source, licenses, manifest pins, selection and limits; `scripts/coco_photos.py restore` rebuilds both banks. A second real download matched all files byte for byte; the project loader accepted both banks. All 320 decoded photos are distinct with native axes ≥267 px. Use final `matched_pairs.jsonl` only for held-out evaluation; loader-sampled choices are not the fixed comparisons.

## Current: Birdwatcher v2

On a Lambda RTX A6000, trained Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203`. The final PEFT adapter is backed up locally in `runs/birdwatcher/birdwatcher-v2.zip`. Its safetensors SHA-256 is `139b7a7c9b07e6916eaf2ffe271b8e1c1796d79e0659c7698b13fc00e181d187`. `bash scripts/reproduce-birdwatcher.sh train` reconstructed the data and reran both stages on the same A6000: byte-identical weights, same visual reports and all eight paired text responses. CUDA runs completed; instance not terminated. 25 offline tests and 19 subtests passed. Next scientific test: unseen higher-resolution photos and independent prompts, not another pass over CIFAR. See `outputs/example-model/` for scope and caveats.

## Historical: V1 handoff

`demo`, `prepare`, `train`, and `evaluate` implement scored image banks → pair plan → Qwen3-VL language LoRA → held-out report. See README for data format, CPU and Lambda runbooks. V1 trains one selected, arbitrary axis per adapter using supervised A/B choices. DPO and other model families are future work.

## Real avian dataset staging

`datasets/avian` contains 120 matched bird/control pairs from CIFAR-10 official training batches; `datasets/avian-final` contains 40 independent image pairs exclusively from the official test batch (20 A / 20 B). The `watch_interest` ratings are class-derived, not human judgments. `scripts/prepare_avian.py` reproduces both banks from the checksum-verified Toronto binary archive. Both banks passed CLI schema preparation; all 320 PNGs decoded at native 32×32 with distinct RGB hashes. All 330 file hashes matched after upload to `/home/ubuntu/second-look/datasets/` on the specified Lambda host. The image banks and archive are Git-ignored. No training or commit in this staging task. Keep the final bank reserved; use `matched_pairs.jsonl` rather than a resampled CLI plan for the final 40-pair comparison.

## Verification before resize

- CPU offline tests: 23 passed + 19 subtests (including EXIF-normalized duplicate detection, nonfinite-choice rejection, and checkpoint identity checks). Python 3.12, torch 2.8.0, transformers 4.57.3, PEFT 0.18.0. Dependency and Git whitespace checks clean.
- Demo: 24 decoded shapes; default plan: 112 train pairs, 6 held-out pairs, 5 held-out singles.
- Ran actual CLI train/evaluate on shapes with a tiny random local Qwen3-VL checkpoint. One optimizer step saved an adapter; reload produced base and adapted reports on 2 pairs and 5 singles each. Separate tests verify finite updated adapter weights and label sensitivity. This is a plumbing result, not evidence of bird preference. Ignored `work/` contains local artifacts.
- No pretrained model run, GPU verification, or Lambda rental. No photos or model weights in Git.

## Next action

After the 16 GB RAM reboot, check available memory, then follow the README one-step pretrained CPU smoke and measure peak RSS/time. Only then decide on renting a GPU. A behavioral experiment needs rated real images and independent held-out photo sessions.

Published V1 commit `a592a94` to `stevefortier/second-look` as the initial `main` and verified the full remote branch SHA against the local commit. No deploy key setup is needed.

## Optional conversational continuation

`scripts/avian_voice.py` adds answer-only text tuning and a separate greedy base/adapter comparison. The default 30 updates at LR `1e-5` cover 22 opinion examples and eight relevance/truthfulness controls once; weights are saved separately. Example from `/home/scratch/Workspaces/second-look` (replace `VISUAL_ADAPTER` with the existing PEFT directory for the matching base):

```sh
cd /home/scratch/Workspaces/second-look
export PYTHONPATH=.
.venv/bin/python "$PWD/scripts/avian_voice.py" train --adapter "$VISUAL_ADAPTER" --output work/avian-voice
.venv/bin/python "$PWD/scripts/avian_voice.py" compare --adapter work/avian-voice --output work/avian-probes.json
```

For a local base snapshot pass `--model /path/to/snapshot` to both commands and its actual portable Hub ID via `--base-id` when training. CPU is default; optionally pass `--device cuda --four-bit` to both. Run `compare` on the untouched visual adapter too to isolate the effect of text continuation. Inspect the eight paired responses for stance, instruction-following, and caution; recheck held-out visual choices separately before drawing conclusions. Local offline verification: 25 tests and 19 subtests passed. A one-step random tiny-checkpoint smoke changed adapter weights, preserved the specified base ID, and wrote all eight paired responses on transformers 4.57.3. No pretrained behavioral result or remote run for this feature.
