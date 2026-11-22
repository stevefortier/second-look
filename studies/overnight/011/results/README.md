# 011 — cue-first captions: narrow gain, worse grounding

Do not promote. Nine gains and five losses across 186 known prompt-fields; correct assignments 13 → 17. Correct by prompt: What strikes you 5 → 5, What would you say 5 → 7, image alone 3 → 5 (each /62). By truth polarity: L 5/24 → 7/24, M 4/114 → 5/114, H 4/48 → 5/48. Activity accounted for the net increase; funny dropped 2 → 0. Grounding/relevance flags doubled 3 → 6/24; no complete mixed reversals (0/12 both). Explicit agreement 38 → 41/62, MAE 0.387 → 0.355. Eight reused development photos, four groups, one model reviewer; no final evaluation. See [diagnostics](summary.md) and adjacent raw records.

Label coverage: nonnull count (low/mid/high). Labels are reviewed pilot proxies, not Steve’s ratings. Unknowns excluded, not scored as midpoint.

| Axis | Fit /16 | Dev /8 |
|---|---|---|
| liking | 16 (1/5/10) | 8 (1/3/4) |
| beauty | 16 (1/9/6) | 8 (0/4/4) |
| cuteness | 14 (1/7/6) | 7 (0/5/2) |
| scary | 16 (4/11/1) | 8 (0/8/0) |
| funny | 15 (4/5/6) | 8 (0/7/1) |
| activity | 16 (9/7/0) | 7 (3/3/1) |
| crowding | 16 (1/7/8) | 8 (1/5/2) |
| color_intensity | 16 (3/10/3) | 8 (3/3/2) |

Reproduce the CPU join: `.venv/bin/python studies/overnight/011/report.py --out work/repro-011`. For exact GPU source in a disposable checkout, copy the two `../launch-source/*.py.source` snapshots to their corresponding paths in `../artifacts.sha256` before verifying it.
