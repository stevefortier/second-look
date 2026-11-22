# Study 023

Balancing explicit supervision across eight axes improved image-only endpoint matches to 13/65 (base 7, 022 adapter 11), including 6/24 lows (base and 022: 5). Both neutral invitations nevertheless fell below base: 14 vs 15 and 17 vs 18 out of 128 (022: 17 and 18). No promotion. Mixed low/high coverage was 3/33 vs 4 for base; neither reversed both directions. Prompted agreement was 84/128 vs 82 for base (022: 85), ordinal MAE 0.344 vs 0.359 (022: 0.336).

Same 16 reused group-disjoint development images and eight fit groups; model-reviewed pilot proxies, not Steve’s ratings. No historical-final access. No missing development labels; null behavior untested. A single model reviewer reused 65 exact photo/prompt/text judgments and coded 31 new replies against numbered images before joining identities. On 48 replies each, adapter/base flags: generic praise 2/11, relevance problems 5/24, schema echo 0/1. Twelve backup checksums matched, metrics reproduced byte-for-byte, 80 tests and 19 subtests passed.

Proxy labels (nonnull, low/mid/high, missing):

- activity: fit n=8, 4/4/0, 0 missing; dev n=16, 6/8/2, 0 missing.
- beauty: fit n=8, 0/4/4, 0 missing; dev n=16, 1/10/5, 0 missing.
- color_intensity: fit n=8, 3/4/1, 0 missing; dev n=16, 4/8/4, 0 missing.
- crowding: fit n=8, 2/4/2, 0 missing; dev n=16, 5/4/7, 0 missing.
- cuteness: fit n=8, 1/3/4, 0 missing; dev n=16, 0/10/6, 0 missing.
- funny: fit n=8, 1/4/3, 0 missing; dev n=16, 2/9/5, 0 missing.
- liking: fit n=4, 0/0/4, 4 missing; dev n=16, 0/5/11, 0 missing.
- scary: fit n=8, 3/4/1, 0 missing; dev n=16, 6/9/1, 0 missing.

Reproduce: `.venv/bin/python studies/overnight/023/report.py`. Raw generations, blind judgments, per-axis diagnostics and baselines in `results/`.
