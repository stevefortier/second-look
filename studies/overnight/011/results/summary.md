# 011 cue-first caption development results

Each column shows the within-run base → adapter comparison on the same eight reused development photos. Image-alone and the two neutral prompts contribute to the totals.

| Measure | 010 focused | 011 cue-first |
|---|---:|---:|
| liking correct / known | 0 / 24 → 0 / 24 | 0 / 24 → 0 / 24 |
| beauty correct / known | 1 / 24 → 0 / 24 | 1 / 24 → 1 / 24 |
| cuteness correct / known | 0 / 21 → 0 / 21 | 0 / 21 → 0 / 21 |
| scary correct / known | 0 / 24 → 0 / 24 | 0 / 24 → 0 / 24 |
| funny correct / known | 2 / 24 → 1 / 24 | 2 / 24 → 0 / 24 |
| activity correct / known | 6 / 21 → 7 / 21 | 10 / 21 → 14 / 21 |
| crowding correct / known | 0 / 24 → 0 / 24 | 0 / 24 → 1 / 24 |
| color_intensity correct / known | 0 / 24 → 0 / 24 | 0 / 24 → 1 / 24 |
| All fields: eligible | 186 → 186 | 186 → 186 |
| All fields: correct | 9 → 8 | 13 → 17 |
| All fields: mentions | 20 → 14 | 22 → 21 |
| All fields: absent | 166 → 169 | 163 → 163 |
| All fields: ambiguous | 0 → 3 | 1 → 2 |
| All fields: opposite | 2 → 1 | 1 → 0 |
| All fields: mid_disagreement | 9 → 5 | 8 → 4 |
| All fields: unknown | 6 → 6 | 6 → 6 |
| All fields: mentions_unknown | 1 → 1 | 3 → 3 |
| responses | 24 → 24 | 24 → 24 |
| mixed_eligible | 12 → 12 | 12 → 12 |
| mixed_exact | 0 → 0 | 0 → 0 |
| irrelevant | 3 → 6 | 3 → 6 |
| schema_echo | 0 → 0 | 0 → 0 |
| generic_praise | 8 → 3 | 2 → 0 |
| literal_phrase_echo | 0 → 0 | 0 → 0 |
| Explicit liking correct / known | 3 / 8 → 3 / 8 | 3 / 8 → 3 / 8 |
| Explicit beauty correct / known | 4 / 8 → 4 / 8 | 4 / 8 → 4 / 8 |
| Explicit cuteness correct / known | 6 / 7 → 6 / 7 | 6 / 7 → 6 / 7 |
| Explicit scary correct / known | 6 / 8 → 7 / 8 | 6 / 8 → 7 / 8 |
| Explicit funny correct / known | 3 / 8 → 4 / 8 | 3 / 8 → 4 / 8 |
| Explicit activity correct / known | 7 / 7 → 7 / 7 | 7 / 7 → 7 / 7 |
| Explicit crowding correct / known | 5 / 8 → 6 / 8 | 5 / 8 → 6 / 8 |
| Explicit color_intensity correct / known | 4 / 8 → 4 / 8 | 4 / 8 → 4 / 8 |
| Paired gains | 3 | 9 |
| Paired losses | 4 | 5 |
| Explicit before correct / known; MAE | 38 / 62; 0.387 | 38 / 62; 0.387 |
| Explicit after correct / known; MAE | 41 / 62; 0.355 | 41 / 62; 0.355 |
| Explicit constant_M correct / known; MAE | 38 / 62; 0.387 | 38 / 62; 0.387 |
| Explicit fit_median correct / known; MAE | 36 / 62; 0.468 | 36 / 62; 0.468 |

Read changes within each run: separate blind reviews can disagree even for base responses. Cross-run totals do not rank adapters. Explicit agreement is secondary. Phrase echo uses each run’s actual full and secondary captions. The runs match on update count and prompt order, not target tokens, vocabulary or content emphasis. One seed, proxy labels (not Steve’s ratings), and no independent final evaluation.
