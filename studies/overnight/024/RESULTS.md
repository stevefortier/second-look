# Study 024 — half-dose balanced supervision

**Do not promote.** Halving the explicitly scored updates preserved image-only low matches but lost two high matches versus 023. Both verbal invitations declined. Forced scoring improved, not spontaneous transfer.

| Predeclared development gate | Base | 023 adapter | 024 adapter | Gate |
|---|---:|---:|---:|---|
| Image only: correct low (≥5/24) | 5 | 6 | 6 | Pass |
| Image only: correct low or high (>7/65) | 7 | 13 | 11 | Pass |
| “What strikes you about this photo?” (≥15/128) | 15 | 14 | 13 | Fail |
| “What would you say about this scene?” (≥18/128) | 18 | 17 | 16 | Fail |

The matched-improvement rule also failed: one tie, three declines and no gains versus 023 on those four gates. Against base, the 024 image-only endpoint comparison had six paired gains and two losses. Both a correct low and a correct high appeared in the same reply in 4/33 eligible cases for 024 and base (023: 3/33); neither base nor 024 matched every endpoint or reversed both directions in any such case.

| Separate forced-score task | Exact /128 | Ordinal MAE |
|---|---:|---:|
| Base | 82 | 0.359375 |
| 023 adapter | 84 | 0.343750 |
| 024 adapter | 93 | 0.273438 |
| Constant midpoint | 63 | 0.507813 |
| Fit-only per-axis median | 64 | 0.507813 |

024 gained 18 forced-score matches and lost seven versus base. Across 48 neutral replies per condition, adapter/base flags were 0/11 for unsupported generic praise, 11/24 for relevance or grounding problems and 0/1 for schema echo.

| Axis | Fit low/mid/high (missing) | Dev low/mid/high (missing) |
|---|---|---|
| Activity | 4/4/0 (0) | 6/8/2 (0) |
| Beauty | 0/4/4 (0) | 1/10/5 (0) |
| Color intensity | 3/4/1 (0) | 4/8/4 (0) |
| Crowding | 2/4/2 (0) | 5/4/7 (0) |
| Cuteness | 1/3/4 (0) | 0/10/6 (0) |
| Funny | 1/4/3 (0) | 2/9/5 (0) |
| Liking | 0/0/4 (4) | 0/5/11 (0) |
| Scary | 3/4/1 (0) | 6/9/1 (0) |

**Review and limitations.** A single model reviewer reused 53 frozen 023 judgments for exact image/prompt/text matches and coded the other 43 from anonymous texts and numbered photos without the key or proxy labels. All 96 judgments were frozen before the identity join (SHA-256 `d96324103990190eda09fbce7130fc0ed472e984fd5f365b7a14397b0584c1e9`). This was not independent human validation. The 16 reused development photos are from distinct groups disjoint from the eight fit groups; repeated prompts are not independent photos. Ratings are model-reviewed proxies, not Steve’s preferences. There are no unknown development labels to test null behavior. No historical-final evaluation was used. The comparison applies to this particular eight-update selection, not every possible half-dose schedule.

**Verification.** Remote status was `exit 0 complete`; all 12 backup files, including adapter weights, matched the remote SHA-256 manifest (`backup.sha256`, relative to `work/overnight-024/`). From the project root run `.venv/bin/python studies/overnight/024/report.py`. `results/metrics.json` reproduced byte-for-byte; it contains all per-axis and per-prompt diagnostics. Raw generations, training summary, blind packet, freeze and key are in `results/`. Local suite: 81 passed, 19 subtests passed.
