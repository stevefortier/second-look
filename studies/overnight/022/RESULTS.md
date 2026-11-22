# Study 022

Replacing high-endpoint score targets with midpoint targets improved prompted agreement (85/128 vs 82 base and 71 for 021; ordinal MAE 0.336 vs 0.359 and 0.453). On image alone, 11/65 endpoints matched vs 7 base and 12 for 021; low endpoints fell to 5/24, tied with base and below 021’s 7. The two neutral invitations scored 17 and 18/128 vs base 15 and 18. Both directions appeared correctly in only 2/33 mixed opportunities vs 4 base. No promotion: explicitly scoring qualities improved, but spontaneous low coverage regressed.

Same 16 reused development groups; the eight quality axes are model-reviewed pilot proxies, not Steve’s ratings. Zero missing development scores means null behavior cannot be evaluated here. No historical-final access. Reused 57 exact photo/prompt/text judgments and coded 39 blind against numbered photos before joining the key. Single model reviewer, not independent human validation. Twelve backup checksums matched, report reproduced byte-for-byte, 79 tests and 19 subtests passed.

Proxy labels by axis: fit and development nonnull counts, then low/mid/high and missing.

- activity: fit n=8 (4/4/0); 0 missing; development n=16 (6/8/2); 0 missing.
- beauty: fit n=8 (0/4/4); 0 missing; development n=16 (1/10/5); 0 missing.
- color_intensity: fit n=8 (3/4/1); 0 missing; development n=16 (4/8/4); 0 missing.
- crowding: fit n=8 (2/4/2); 0 missing; development n=16 (5/4/7); 0 missing.
- cuteness: fit n=8 (1/3/4); 0 missing; development n=16 (0/10/6); 0 missing.
- funny: fit n=8 (1/4/3); 0 missing; development n=16 (2/9/5); 0 missing.
- liking: fit n=4 (0/0/4); 4 missing; development n=16 (0/5/11); 0 missing.
- scary: fit n=8 (3/4/1); 0 missing; development n=16 (6/9/1); 0 missing.


Reproduce: `.venv/bin/python studies/overnight/022/report.py`. Full judgments, raw generations, per-axis misses/reversals and both baselines in results/.
