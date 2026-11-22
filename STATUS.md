# Second Look — handoff

## Current: Birdwatcher v2

On a Lambda RTX A6000, trained Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203`. The final PEFT adapter is backed up locally in `runs/birdwatcher/birdwatcher-v2.zip`. Its safetensors SHA-256 is `139b7a7c9b07e6916eaf2ffe271b8e1c1796d79e0659c7698b13fc00e181d187`. `bash scripts/reproduce-birdwatcher.sh train` reconstructed the data and reran both stages on the same A6000: byte-identical weights, same visual reports and all eight paired text responses. CUDA runs completed; instance not terminated. 25 offline tests and 19 subtests passed. Next scientific test: unseen higher-resolution photos and independent prompts, not another pass over CIFAR. See `experiments/birdwatcher-v2/` for scope and caveats.

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
