# 008 development results

All neutral counts pool three prompts on the same eight photos. Columns show base → adapter.

| Axis | Neutral correct / known | Explicit correct / known |
|---|---:|---:|
| liking | 3 → 1 / 24 | 3 → 3 / 8 |
| beauty | 1 → 1 / 24 | 4 → 4 / 8 |
| cuteness | 0 → 0 / 21 | 6 → 6 / 7 |
| scary | 0 → 0 / 24 | 6 → 8 / 8 |
| funny | 0 → 0 / 24 | 3 → 4 / 8 |
| activity | 12 → 10 / 21 | 7 → 7 / 7 |
| crowding | 0 → 0 / 24 | 5 → 6 / 8 |
| color_intensity | 0 → 0 / 24 | 4 → 4 / 8 |

| Neutral diagnostic | Base | Adapter |
|---|---:|---:|
| eligible | 186 | 186 |
| correct | 16 | 12 |
| mentions | 34 | 26 |
| absent | 151 | 157 |
| ambiguous | 1 | 3 |
| opposite | 5 | 3 |
| mid_disagreement | 13 | 11 |
| unknown | 6 | 6 |
| mentions_unknown | 3 | 3 |
| responses | 24 | 24 |
| mixed_eligible | 12 | 12 |
| mixed_exact | 0 | 0 |
| irrelevant | 7 | 10 |
| schema_echo | 0 | 0 |
| generic_praise | 4 | 3 |
| literal_phrase_echo | 0 | 0 |

Matched known fields: gains = 1; losses = 5.

| Explicit diagnostic | Correct / known | Ordinal absolute error sum |
|---|---:|---:|
| before | 38 / 62 | 24 |
| after | 42 / 62 | 21 |
| constant_M | 38 / 62 | 24 |
| fit_median | 36 / 62 | 29 |

For comparison, 007 on these same photos had 14 → 14 / 186 neutral correct, three gains and three losses, 5 → 10 grounding flags, and 38 → 45 / 62 explicit correct. Cross-study judgments were frozen separately; they are not a single pooled blind review. Neither study tested independent generalization or Steve’s preferences.
