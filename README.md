# Second Look

Train a Qwen vision-language LoRA from two image folders and your own scored axes. Ask whether the adapter changes what the model **chooses to look at**, not whether it can say it loves birds. A changed choice is not proof of felt pleasure.

## Try the data workflow (no GPU or model download)

From the repository root on Linux, using Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
second-look demo work/shapes
second-look prepare work/shapes --axis watch_interest --output work/plan.json
python -m pytest -q
```

The generated shapes and ratings are **synthetic plumbing fixtures**, not bird observations, human ratings, or experimental evidence. The CLI refuses to overwrite demo directories, plans, runs or reports. Choose a new output path for each trial.

## Bring your own photos

```
datasets/backyard/
  axes.yaml
  ratings.jsonl
  liked/heron.jpg
  control/sleeping-fox.jpg
  ...
```

`liked/` and `control/` are sampling banks, **not class labels**. Rate every photo, controls included. The file layout and run are the same whether your subject is birds or something else. Example `axes.yaml`:

```yaml
watch_interest:
  low: I'd rather look away
  mid: indifferent
  high: I'd love to keep watching
scary:
  low: actively reassuring
  mid: neutral
  high: frightening
  question: Which picture feels more frightening?
```

Example `ratings.jsonl` (one JSON object per line; many more than two images needed):

```jsonl
{"image":"liked/heron.jpg","group":"creek-visit-01","scores":{"scary":0.6,"watch_interest":0.95}}
{"image":"control/sleeping-fox.jpg","group":"fox-visit-02","scores":{"scary":-0.8,"watch_interest":0.7}}
```

Choose any axis name. Use scores in `[-1, 1]`; `0` is the midpoint, `null` means unknown. A higher value is not morally better or necessarily more appealing. Use an explicit question for a trait axis; leave it absent for a free-choice preference. Select **one axis per adapter** in V1. The target is an A/B choice, not a regression to the numeric rating. A gap filters ties and near-ties. Keep bursts, crops and re-encodings in the same `group`; match backgrounds and camera style across banks. Give each bank multiple independent groups and enough variation to produce scored pairs in both splits. Only use images you have permission to process; photos and weights belong outside Git.

## First real CPU smoke (after the VM has more RAM)

```sh
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
pip install -e '.[train,test]'
export HF_HOME="$PWD/work/hf-cache" OMP_NUM_THREADS=2
/usr/bin/time -v second-look train work/shapes --axis watch_interest --device cpu --steps 1 --limit 2 --output runs/cpu-smoke
/usr/bin/time -v second-look evaluate work/shapes --run runs/cpu-smoke --device cpu --output runs/cpu-smoke/report.json
```

The first pretrained run downloads `Qwen/Qwen3-VL-2B-Instruct`. `--model` can instead point to a local copy of that checkpoint, including an immutable Hugging Face snapshot. Use the same local snapshot when moving a reproducible experiment between machines. CPU smoke acceptance: one finite optimizer step, adapter on disk, reload, and a base/adapter report on unseen images. Record peak RSS and elapsed time. A one-step run is **not** expected to teach the preference. 16 GB is a budget to measure, not a measured minimum. For an actual preference experiment, replace `work/shapes` with your rated dataset and increase steps only after checking the plan and baseline.

## GPU / Lambda

Use a fresh environment with a CUDA-enabled PyTorch 2.8.0 / torchvision 0.23.0 pair compatible with the host driver (not the CPU wheels above). Install `pip install -e '.[train,quant]'`; verify `python -c 'import torch; print(torch.cuda.is_available())'` prints `True`. Copy the dataset and code; repeat the same commands with `--device cuda --four-bit` **on both commands**, starting with one step. Persist the run directory and report before releasing the machine. No cloud account, upload, or rental is required by this code. Treat GPU memory and timing as unmeasured until exercised there.

## Reading the outcome

The run's `experiment.json` records the examples, file digests, axis, seed and training options; `adapter/` holds the PEFT adapter rather than a second base model. Compare the adapter and its own base on the same held-out pairs. Also inspect single-image save/skip counts **separately by bank**: better pair accuracy need not mean that desirable photos became more appealing. A rise for both banks can indicate a general save bias. Counts from overlapping pairs are not independent samples. For a meaningful study, use new photo sessions, more than one seed, neutral prompts, and check explicitly that user instructions still override preference. V1 is supervised choice tuning for Qwen3-VL, not a DPO or Gemma backend.
