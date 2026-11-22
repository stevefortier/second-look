# One-run avian extremes protocol

This is a class-conditioned proxy-label experiment, not an aesthetic assessor. No promotion rule, production deployment, checkpoint selection on heldout, or recurring job. The pre-existing overnight STOP remains in place.

## Operate from the local project root

```bash
cd /home/scratch/Workspaces/second-look
bash scripts/extreme_gpu.sh status   # worker PID, lock, stage, completed count, training ETA and GPU PID
bash scripts/extreme_gpu.sh stop     # sets only work/extremes-run/STOP_EXTREMES
bash scripts/extreme_gpu.sh backup   # only after exit 0; verifies remote AND local SHA256SUMS
```

The existing A6000 worker lives in `/home/ubuntu/second-look/work/extremes-run`. `worker.state` is authoritative for exit; `status.json` is the last detailed Python step, `run.log` the worker log, and `artifacts/training.jsonl` the update history. It runs under detached `nohup setsid` and an exclusive run-local `flock`; no cron or automatic restart. The DATA_READY wait is bounded to 24 hours; no GPU allocation during the wait. It refuses to start on an occupied GPU. Do not relaunch an already-locked or completed job.

The data producer installs split-local `ratings.jsonl` + relative images beneath the run directory and places `DATA_READY` at its root **after verification**. The reader requires exactly 4,000+1,000 ratings, disjoint groups and image paths, ten valid signed axes and the three correlated endpoints. The class labels in the training/evaluation runner derive from the documented bank construction, not independent visual reannotation. Data hashes and the actual photo indices for all conditions are frozen in `artifacts/protocol.json` before training. No model sees the split label, file path or class name. `code/` is rsynced and checked by SHA-256 (`bash scripts/extreme_gpu.sh stage`, only before protocol freeze).

To resume an *interrupted* authorized run, inspect its exact exit + free lock, leave code/data unchanged, and run `bash scripts/extreme_gpu.sh launch`. If the dedicated stop marker was explicitly set, obtain permission to continue before clearing that marker. **Never clear the overnight STOP.** The last two atomically renamed checkpoints include trainable weights, optimizer moments, CPU/CUDA RNG and next fit offset. On resume uncheckpointed training-log rows are discarded; no fit image is missed. Evaluation writes atomic matched per-image records and skips already-complete images.

## What is measured

* One shuffled pass, seed 382711; pinned Qwen/Qwen3.8-27B `37561816e66b5b9153b1b64c10968c18c1e6f4a0`, BF16/NF4 double-quantized language model, SDPA, q/v LoRA rank 8, alpha 16, zero dropout, AdamW 5e-5. Batch 2, 2,000 updates, checkpoint first batch and every 40 images. Exactly one training task per fit image. Within each avian class one quarter are composition-only prose, remainder a mix of per-axis A/B/C and ordered ten-axis vectors. Numeric signs map to low/neutral/high. Null vector slots are `?`; null ratings cannot become low. Prose targets describe composition, never the value of people. Technical proxies condition a secondary impression where available.
* Freeze exactly 50+50 heldout photos for independent forced judgments on ten axes and 16+16 drawn from those same photos for a neutral invitation **not used verbatim during fit**. Base and separately reloaded adapter have identical decoding, images and prompts. Labels aren't supplied to the model during evaluation. Verify every stored adapter tensor equals its reloaded value. Greedy, thinking off, maximum 64 new tokens for prose; constrained next-token A/B/C logits **and** unrestricted first token stored for scoring. Report prompted agreement, change, gains/losses and fit-only majority baseline overall and separately by avian class for every axis. Equal fit-majority ties prefer neutral, then low, then high.
* The separate blind literal-word readout applies a frozen dictionary to shuffled anonymous replies with no class labels or condition, then joins for per-class and per-axis correct/mentioned counts and changes. Negations are approximate; unmentioned or conflicting directions abstain. This readout does **not** establish semantic attribution. For independent semantic review, give a reviewer only `artifacts/blind/`; code direction or absence for each axis and photo relevance, freeze/hash, then join `review-key.json`. Do not treat 10 correlated ratings of one image as 10 independent observations. An improvement on the first three axes could simply reflect learned avian recognition.
* Final checksum backup: `work/extremes-offinstance/adapter/`, `report.json`, `paired.json`, `base/`, `adapted/`, `spontaneous-literal-report.json`, `blind/`, `reload.json`, `coverage.json`, protocol, training trace. `backup` intentionally excludes large optimizer checkpoints. Until `worker.state` shows `exit 0` and hashes verify, no finished adapter or result is claimed.

## Preflight vs evidence

`extreme_preflight.py` used four synthetic color images solely to exercise real GPU 27B load, gradients, optimizer update, checkpoint and separate-process resume. Only the independently verified photographic bank counts as experimental training/evaluation. Preflight artifacts are isolated under `preflight/`.
