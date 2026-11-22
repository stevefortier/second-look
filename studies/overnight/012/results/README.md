# 012 — image-only rehearsal: development results

**Image-only criterion: met.** Correct spontaneous low/high assignments: 3 → 5 / 24 (4 gains, 2 losses).
Responses flagged for grounding/relevance: 2 → 1 / 8 image-only; 7 → 3 / 24 across all three prompts. Lower is better.

| Spontaneous agreement | Base → adapter / known (paired changes) |
|---|---|
| Image-only non-mid, liking | 0 → 1 / 5 (1 gain, 0 losses) |
| Image-only non-mid, other seven qualities | 3 → 4 / 19 (3 gains, 2 losses) |
| All prompts | 16 → 16 / 186 (7 gains, 7 losses) |
| All prompts, liking | 0 → 1 / 24 (1 gain, 0 losses) |
| All prompts, other seven qualities | 16 → 15 / 162 (6 gains, 7 losses) |
| Image-only, truth L | 2 → 2 / 8 (2 gains, 2 losses) |
| Image-only, truth H | 1 → 3 / 16 (2 gains, 0 losses) |
| [image only] | 5 → 6 / 62 (4 gains, 3 losses) |
| What strikes you about this photo? | 6 → 5 / 62 (2 gains, 3 losses) |
| What would you say about this scene? | 5 → 5 / 62 (1 gain, 1 loss) |
| All prompts, truth L | 5 → 6 / 24 (5 gains, 4 losses) |
| All prompts, truth M | 6 → 5 / 114 (0 gains, 1 loss) |
| All prompts, truth H | 5 → 5 / 48 (2 gains, 2 losses) |

### Explicit ordinal questions (secondary)

| Predictor | Correct / known | MAE | Correct / truth L | Correct / truth M | Correct / truth H |
|---|---:|---:|---:|---:|---:|
| before | 38/62 | 0.387 | 5/8 | 29/38 | 4/16 |
| after | 40/62 | 0.355 | 4/8 | 33/38 | 3/16 |
| constant_M | 38/62 | 0.387 | 0/8 | 38/38 | 0/16 |
| fit_median | 36/62 | 0.468 | 3/8 | 27/38 | 6/16 |

### Label variation

Counts are low / midpoint / high / unknown. Unknown fields are excluded from agreement and MAE.

| Axis | Fit (16) | Dev (8) |
|---|---:|---:|
| liking | 1 / 5 / 10 / 0 | 1 / 3 / 4 / 0 |
| beauty | 1 / 9 / 6 / 0 | 0 / 4 / 4 / 0 |
| cuteness | 1 / 7 / 6 / 2 | 0 / 5 / 2 / 1 |
| scary | 4 / 11 / 1 / 0 | 0 / 8 / 0 / 0 |
| funny | 4 / 5 / 6 / 1 | 0 / 7 / 1 / 0 |
| activity | 9 / 7 / 0 / 0 | 3 / 3 / 1 / 1 |
| crowding | 1 / 7 / 8 / 0 | 1 / 5 / 2 / 0 |
| color_intensity | 3 / 10 / 3 / 0 | 3 / 3 / 2 / 0 |

### Evidence and reproduction

Do not promote on this development result. One independent model review, one seed, eight reused development photos in four groups and illustrative proxy labels, not Steve’s judgments. No independent final or cross-run adapter ranking. The 157 image-only caption updates changed both exposure and task mix.

The reviewer used `blind.json` response IDs and actual photos. No sanitized numbered packet was used. `blind-audit.json` documents post-freeze numbering and hashes; `join.json` links original IDs to the raw generations. Each ID, original text, side, prompt and image was checked; all eight photo byte hashes matched the locked 008 protocol. Reviewer notes are retained in `blind-judgments.json` and `joined-rows.json`.

[Full diagnostics](summary.md): axis agreement, mentions, errors, mixed endpoints, flags and six-word training-caption echoes. `primary.json` breaks these down by prompt; `paired.json` records gains and losses by photo/group; `secondary.json` includes per-axis prompted models and baselines.

The first join used the frozen review and local backup (CPU, no weights). From the repository root:
```sh
(cd studies/overnight/012/results && sha256sum -c blind-judgments.json.sha256)
.venv/bin/python studies/overnight/012/report.py --join-from work/overnight-012 --judgments-sha256 fe60335677b8e1d87d7e877817847a8c3a3ad1bd5bf3a63babd7015e3ff59909
```

Recompute every file from post-freeze inputs and the repository’s actual development photos, without the backup:
```sh
.venv/bin/python studies/overnight/012/report.py --out work/repro-012
diff -qr studies/overnight/012/results work/repro-012
```
