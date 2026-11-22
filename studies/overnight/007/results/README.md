# 007 results: explicit responses improved, spontaneous attribution did not

Across eight development photos, both models correctly assigned **14 of 186 known qualities** without prompting for a rating. The adapter made three gains and three losses on matched photo/prompt/axis comparisons. Neither model expressed personal liking on any of its 24 responses or exactly described all endpoints on any of the 12 eligible mixed-endpoint responses. These are repeated prompts on four photo groups, not 186 independent observations; the example ratings are illustrative/proxy labels, not Steve’s preferences. No evidence here for promotion.

| Axis | Spontaneous correct: base → adapter / known | Explicit correct: base → adapter / known |
|---|---:|---:|
| Liking | 0 → 0 / 24 | 3 → 3 / 8 |
| Beauty | 1 → 0 / 24 | 4 → 4 / 8 |
| Cuteness | 0 → 0 / 21 | 6 → 6 / 7 |
| Scary | 0 → 0 / 24 | 6 → 8 / 8 |
| Funny | 2 → 2 / 24 | 3 → 7 / 8 |
| Activity | 11 → 12 / 21 | 7 → 7 / 7 |
| Crowding | 0 → 0 / 24 | 5 → 6 / 8 |
| Color intensity | 0 → 0 / 24 | 4 → 4 / 8 |

On known labels, directional mentions fell from 23 to 18. Opposite endpoints rose from 1 to 2; midpoint disagreements fell from 8 to 2. Absent codes were 157 → 161, ambiguous 6 → 7. Each stage mentioned 3 of 6 unknown-label opportunities; these received neither credit nor penalty. Grounding flags (`irrelevant`) rose from 5 to 10 of 24, generic praise fell from 1 to 0, schema recitations and literal training-phrase matches stayed at zero.

The separate **forced-choice** ratings improved from 38 to 45 correct of 62 known fields; constant M scored 38 and fit median scored 39. Absolute error sums in ordinal steps were 24 (base), 17 (adapter), 24 (constant M) and 25 (fit median). Forced scores are not evidence of spontaneous attribution.

## Reproduction and audit

From the repository root, run `.venv/bin/python studies/overnight/007/report.py`. Use `--out DIRECTORY` for a separate copy. The initial join used `--join-from work/overnight-007` after the reviewer froze judgments; it checked numbered JPEG hashes against the protocol and both blinded texts against raw generations. Reproduction uses the published text join only: no images, weights, network or GPU. `.venv/bin/python -m pytest -q tests/test_report007.py` checks byte-identical regeneration and exercises counting rules and join validation.

- `before.json`, `after.json`, `train.json`, `manifest.json`: byte-identical local copies of the completed run's outputs, including score logits, training targets and losses.
- `blind-judgments.json` and `.sha256`: independent reviewer's frozen inputs. `report.py` pins their judgment digest and never writes either file. `join.json` maps each response ID to stage, image, anonymous photo number, prompt and raw text. `joined-rows.json` adds ordinal truth, codes, flags, group, mixed outcome and literal matches.
- `primary.json`: all counts by model, axis and prompt, and pooled across prompts. A mention requires an L/M/H code; an ambiguous code is separate. `opposite` is an L↔H error; `mid_disagreement` is M versus either endpoint, in either direction. Mixed eligibility requires both L and H among known labels; exact means **every** endpoint axis matches in the same response. Midpoint axes do not enter that joint endpoint. Literal echo is a reproducible, post hoc screen: any six consecutive case-folded alphanumeric words shared with a fit-only training caption; this cannot detect paraphrase.
- `paired.json`: gains (wrong/absent/ambiguous → exact), losses (reverse), both correct, neither correct by photo and group, per axis and pooled; same paired counts for exact mixed agreement. Unknown truth is excluded.
- `secondary.json`: explicit accuracy and mean absolute error on the −1/0/+1 ordinal scale; each photo/axis counts once. Both models and both baselines use the same known dev labels. Fit medians exclude null scores and are thresholded like the source ratings (±0.5 themselves are M).
