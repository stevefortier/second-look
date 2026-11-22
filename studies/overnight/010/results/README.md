# 010: prompted scores improved; spontaneous attribution did not

Do not promote. On eight reused dev photos in four groups, correct unaided assignments fell 9→8/186 (3 gains, 4 losses). By invitation: “What strikes you” 3→3, “What would you say” 4→3, image alone 2→2, each /62. Low truth improved 0→2/24; midpoint declined 5→3/114 and high declined 4→3/48. Generic praise declined 8→3/24, but grounding or relevance flags rose 3→6/24. Neither model captured both directions on any of 12 mixed opportunities. Prompted accuracy rose 38→41/62 and ordinal MAE fell 0.387→0.355. Repeated prompts are not independent photos; independent model reviews differ even on baseline output. No historical final was used.

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

See [full diagnostics](summary.md) and adjacent raw records. Reproduce from repository root: `.venv/bin/python studies/overnight/010/report.py --out work/repro-010`.
