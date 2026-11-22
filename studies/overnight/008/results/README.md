# 008: explicit accuracy improved, spontaneous attribution declined

On the same eight development photos, correct unaided assignments fell from **16 to 12 of 186** known photo/prompt/axis opportunities (one gain, five losses). The decrease came from liking (3 → 1) and activity (12 → 10); the other six axes were unchanged in aggregate. Grounding flags rose from 7 to 10 of 24 responses. Neither stage jointly expressed all endpoints on any of the 12 mixed opportunities. Forced-choice accuracy rose from 38 to 42 of 62, versus 38 for constant M and 36 for the **008** fit median. Forced choices did not transfer to the primary behavior. **Do not promote.**

[Generated diagnostics](summary.md) list each axis, errors, abstentions and flags. [Study 007](../../007/results/README.md) had no net spontaneous gain (14 → 14) despite a larger explicit gain (38 → 45). The two studies reused photos and prompts but did not reuse all base outputs or a single blind review: of 24 matched base generations, nine texts differed, and even among the 15 identical texts there were 17 differing axis codes across the two freezes. Compare within-study changes first; do not interpret the between-adapter difference as a causal effect. These are three prompts on eight photos in four groups, not 186 independent samples. The dev labels are reused proxies, not Steve’s ratings, and fresh fit captions and fit labels changed together. No independent final evaluation.

## Reproduce without the private run directory

From the project root:

```sh
.venv/bin/python studies/overnight/008/report.py --out work/reproduced-008
.venv/bin/python -m pytest -q tests/test_report007.py tests/test_report008.py
```

The default path reconstructs `join.json` from the published post-freeze `blind-audit.json` (original randomized ID allocation, blind rows, numbered response packet and numbered JPEG hashes); it does not read any previously joined output. It validates the pinned SHA-256 and reviewer sidecar, matches IDs and texts across the packet and raw stage records, checks numbered photo hashes against the 008 protocol, and checks the full response grid. No images, model, GPU or network needed. The initial audit export read the original numbered JPEG bytes with:

```sh
.venv/bin/python studies/overnight/008/report.py --join-from work/overnight-008 \
  --judgments-sha256 49f72d45aed7fa26f9e35499e90c9614aa092574236d9ca3cef9379a4199d910
```

- `before.json`, `after.json`, `train.json`, `manifest.json` are byte-identical copies from the checksum-backed run. `ratings.jsonl` is the exact 008 bank used by the runner; its digest is checked against `../protocol.json`. New fit ratings determine the medians, unchanged dev ratings determine truth. The independent reviewer's `blind-judgments.json` and `.sha256` are inputs and are never rewritten.
- `primary.json`: pooled and per-prompt counts by axis and stage. A mention requires L/M/H. Absent and ambiguous are separate; L↔H is opposite, M↔endpoint is a midpoint disagreement. Null truth receives neither credit nor penalty. Mixed exact requires every endpoint on the same response to be correct, with both L and H present in the truth. Literal echo means a shared sequence of six case-folded alphanumeric words from an 008 fit caption; this does not detect paraphrases. `irrelevant` is the reviewer’s grounding flag, not an independent factual audit.
- `joined-rows.json` contains all 48 response-level records; `paired.json` counts exact gains, losses, both and neither by photo and group. `secondary.json` reports forced-choice accuracy and ordinal absolute error (−1/0/+1); medians exclude nulls and ±0.5 map to M.
