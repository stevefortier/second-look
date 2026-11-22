# 021 results

021 replaced 16 of 80 caption updates with explicit endpoint labels. Image alone: correct low endpoints 7/24 vs 5 for base (020: 6); all endpoints 12/65 vs 7 (020: 13). Neutral invitations: 15 and 11/128 vs base 15 and 18 (020: 13 and 12). Mixed low/high in one reply: 0/33 vs 4. Prompted: 71/128 vs 82; MAE 0.453125 vs 0.359375. No promotion. Reused 16-group proxy development panel, no missing dev scores or historical-final access. 49 exact prior image/prompt/text judgments reused; 47 newly judged against numbered photos before identity join. Single model reviewer, not independent human validation. Twelve backup checksums matched; metrics reproduced byte-for-byte; 78 tests + 19 subtests passed.

Proxy labels by axis: fit and development nonnull counts, then low/mid/high and missing.

- activity: fit n=8 (4/4/0); 0 missing; development n=16 (6/8/2); 0 missing.
- beauty: fit n=8 (0/4/4); 0 missing; development n=16 (1/10/5); 0 missing.
- color_intensity: fit n=8 (3/4/1); 0 missing; development n=16 (4/8/4); 0 missing.
- crowding: fit n=8 (2/4/2); 0 missing; development n=16 (5/4/7); 0 missing.
- cuteness: fit n=8 (1/3/4); 0 missing; development n=16 (0/10/6); 0 missing.
- funny: fit n=8 (1/4/3); 0 missing; development n=16 (2/9/5); 0 missing.
- liking: fit n=4 (0/0/4); 4 missing; development n=16 (0/5/11); 0 missing.
- scary: fit n=8 (3/4/1); 0 missing; development n=16 (6/9/1); 0 missing.


Reproduce: `.venv/bin/python studies/overnight/021/report.py`. Full judgments, raw generations, per-axis misses/reversals and both baselines in results/.
