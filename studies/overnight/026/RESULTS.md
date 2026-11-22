# 026 — image-only gain did not transfer to neutral invitations

Do not promote. Pairing the two impressions on image-only updates passed two of four predeclared gates. Image alone: 10/65 low or high qualities expressed correctly (base 7; 025 adapter 8), including 6/24 lows (both comparators 5). Under “What strikes you about this photo?” the adapter matched 11/128 rated qualities (base 15; 025 adapter 11); under “What would you say about this scene?” it matched 12/128 (base 18; 025 adapter 11). These counts include misses, not just qualities mentioned.

Across the three invitations, both a low and a high were correctly expressed in 1/33 eligible replies (base 4); neither model reversed both directions. Generic praise 0/48 vs base 11, relevance flags 15 vs 24, schema echo 0 vs 1. Prompted ordinal accuracy tied base at 82/128 (two gains, two losses), MAE 0.359375 for both; fit-median baseline 64/128, MAE 0.5078125. Equal aggregate prompted scores do not establish spontaneous learning.

## Rating coverage

Counts below are low/mid/high. Eight fit groups, 16 separate development groups, same development panel as 025. Each development axis has 16 known values; each fit axis has eight except liking (four known, four missing).

- Activity: fit 4/4/0; development 6/8/2.
- Beauty: fit 0/4/4; development 1/10/5.
- Color intensity: fit 3/4/1; development 4/8/4.
- Crowding: fit 2/4/2; development 5/4/7.
- Cuteness: fit 1/3/4; development 0/10/6.
- Funny: fit 1/4/3; development 2/9/5.
- Liking: fit 0/0/4; development 0/5/11.
- Scary: fit 3/4/1; development 6/9/1.

No development nulls, so behavior on unknown qualities is untested. These are model-reviewed pilot proxy annotations, not Steve’s opinions. One blind model reviewer reused 53 exact photo/prompt/text judgments and newly coded 43 against numbered photos before unmasking. Reused development data are not independent confirmation. No historical-final access.

From project root run `.venv/bin/python studies/overnight/026/report.py`; the regenerated `results/metrics.json` includes per-axis absent, ambiguous, midpoint disagreement, reversal and exact counts. Raw replies, training log data, blind judgments, freeze receipt and join key are under `results/`. The adapter itself stays out of Git.
