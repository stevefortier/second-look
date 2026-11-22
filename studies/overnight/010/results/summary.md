# 010 focused-caption development results

Base → adapter on the same eight development photos and three neutral prompts.

| Measure | 008 | 009 | 010 |
|---|---:|---:|---:|
| liking correct / known | 3 / 24 → 1 / 24 | 0 / 24 → 0 / 24 | 0 / 24 → 0 / 24 |
| beauty correct / known | 1 / 24 → 1 / 24 | 2 / 24 → 0 / 24 | 1 / 24 → 0 / 24 |
| cuteness correct / known | 0 / 21 → 0 / 21 | 0 / 21 → 0 / 21 | 0 / 21 → 0 / 21 |
| scary correct / known | 0 / 24 → 0 / 24 | 1 / 24 → 0 / 24 | 0 / 24 → 0 / 24 |
| funny correct / known | 0 / 24 → 0 / 24 | 2 / 24 → 2 / 24 | 2 / 24 → 1 / 24 |
| activity correct / known | 12 / 21 → 10 / 21 | 10 / 21 → 12 / 21 | 6 / 21 → 7 / 21 |
| crowding correct / known | 0 / 24 → 0 / 24 | 0 / 24 → 2 / 24 | 0 / 24 → 0 / 24 |
| color_intensity correct / known | 0 / 24 → 0 / 24 | 0 / 24 → 2 / 24 | 0 / 24 → 0 / 24 |
| All fields: eligible | 186 → 186 | 186 → 186 | 186 → 186 |
| All fields: correct | 16 → 12 | 15 → 18 | 9 → 8 |
| All fields: mentions | 34 → 26 | 25 → 28 | 20 → 14 |
| All fields: absent | 151 → 157 | 153 → 151 | 166 → 169 |
| All fields: ambiguous | 1 → 3 | 8 → 7 | 0 → 3 |
| All fields: opposite | 5 → 3 | 2 → 2 | 2 → 1 |
| All fields: mid_disagreement | 13 → 11 | 8 → 8 | 9 → 5 |
| All fields: unknown | 6 → 6 | 6 → 6 | 6 → 6 |
| All fields: mentions_unknown | 3 → 3 | 3 → 3 | 1 → 1 |
| responses | 24 → 24 | 24 → 24 | 24 → 24 |
| mixed_eligible | 12 → 12 | 12 → 12 | 12 → 12 |
| mixed_exact | 0 → 0 | 0 → 0 | 0 → 0 |
| irrelevant | 7 → 10 | 3 → 6 | 3 → 6 |
| schema_echo | 0 → 0 | 0 → 0 | 0 → 0 |
| generic_praise | 4 → 3 | 6 → 2 | 8 → 3 |
| literal_phrase_echo | 0 → 0 | 0 → 0 | 0 → 0 |
| Explicit liking correct / known | 3 / 8 → 3 / 8 | 3 / 8 → 3 / 8 | 3 / 8 → 3 / 8 |
| Explicit beauty correct / known | 4 / 8 → 4 / 8 | 4 / 8 → 4 / 8 | 4 / 8 → 4 / 8 |
| Explicit cuteness correct / known | 6 / 7 → 6 / 7 | 6 / 7 → 5 / 7 | 6 / 7 → 6 / 7 |
| Explicit scary correct / known | 6 / 8 → 8 / 8 | 6 / 8 → 7 / 8 | 6 / 8 → 7 / 8 |
| Explicit funny correct / known | 3 / 8 → 4 / 8 | 3 / 8 → 5 / 8 | 3 / 8 → 4 / 8 |
| Explicit activity correct / known | 7 / 7 → 7 / 7 | 7 / 7 → 5 / 7 | 7 / 7 → 7 / 7 |
| Explicit crowding correct / known | 5 / 8 → 6 / 8 | 5 / 8 → 6 / 8 | 5 / 8 → 6 / 8 |
| Explicit color_intensity correct / known | 4 / 8 → 4 / 8 | 4 / 8 → 4 / 8 | 4 / 8 → 4 / 8 |
| Paired gains | 1 | 11 | 3 |
| Paired losses | 5 | 8 | 4 |
| Explicit before correct / known; MAE | 38 / 62; 0.387 | 38 / 62; 0.387 | 38 / 62; 0.387 |
| Explicit after correct / known; MAE | 42 / 62; 0.339 | 39 / 62; 0.387 | 41 / 62; 0.355 |
| Explicit constant_M correct / known; MAE | 38 / 62; 0.387 | 38 / 62; 0.387 | 38 / 62; 0.387 |
| Explicit fit_median correct / known; MAE | 36 / 62; 0.468 | 36 / 62; 0.468 | 36 / 62; 0.468 |

The six-word echo diagnostic uses the actual caption families for each study: 010 includes full and focused captions. Explicit agreement is secondary. These are paired observations on reused photos with proxy labels, not independent replication or Steve’s ratings. Steps, target tokens, prompt exposure, order and writing style differ.
