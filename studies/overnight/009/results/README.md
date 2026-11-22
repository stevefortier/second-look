# 009 — small neutral-prompt gain, no image-only gain

Do not promote. Across eight reused development photos in four groups, blind scoring found 15 → 18 correct spontaneous assignments out of 186 known fields, with eleven gains and eight losses. By prompt: “What strikes you” 6 → 7, “What would you say” 5 → 7, image alone 4 → 4 (each /62). Low truth improved 4 → 8/24; midpoint fell 7 → 6/114; high stayed 4/48. Grounding flags doubled (3 → 6/24); neither stage jointly captured the endpoints on any of the 12 mixed opportunities. Explicit accuracy 38 → 39/62; ordinal MAE 24/62 = 0.387 for both, also matched by constant midpoint. No independent final or evidence about Steve’s taste. Independent model review, not human scoring.

Label coverage below is known count (L/M/H). Fit labels are reviewed model proxies, not Steve’s; dev labels are held fixed from 007. Nulls excluded, not counted as midpoint.

| Axis | Fit (of 16) | Dev (of 8) |
|---|---|---|
| liking | 16 (1/5/10) | 8 (1/3/4) |
| beauty | 16 (1/9/6) | 8 (0/4/4) |
| cuteness | 14 (1/7/6) | 7 (0/5/2) |
| scary | 16 (4/11/1) | 8 (0/8/0) |
| funny | 15 (4/5/6) | 8 (0/7/1) |
| activity | 16 (9/7/0) | 7 (3/3/1) |
| crowding | 16 (1/7/8) | 8 (1/5/2) |
| color_intensity | 16 (3/10/3) | 8 (3/3/2) |

[Full diagnostics and comparison with 008](summary.md). Rebuild from the repository root: `.venv/bin/python studies/overnight/009/report.py --out work/repro-009`. A fresh local reproduction matched the published analysis files byte for byte. The independent blind review was pinned before joining identities. The raw stage records, review, join audit and machine-readable metrics accompany this report; the weights and photos are not included here.
