# Second Look — handoff

## Current input example: per-image qualities

`datasets/example/` ships 24 train + 8 test original JPEGs, one `images/` folder per split and ordinary per-image `scores` across eight arbitrary axes including `liking`. No liked/control folder or image-pair requirement. Subjective example ratings are illustrative, not Steve's. Legacy COCO bird/control metadata, fixtures and restored development images live under `datasets/legacy-coco/`; historical prepared remote run roots are unchanged. No model retraining is claimed by this migration.

## Active: 004 running; 003 complete, not promoted

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
