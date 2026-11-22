# 018: fresh fit groups, unchanged 013 development panel

Predeclared before launch: compare image-only spontaneous correct low endpoints >=5/24, all correct endpoints >7/65, and two held-out verbal invitations >=15 and >=18 correct fields/128 respectively, using the frozen 013 evaluation protocol. Report grounding flags and per-axis coverage. Meeting these development criteria is not independent validation.

The deterministic fit draw reuses 013's group exclusions and removes every 013 panel group before sampling four images per legacy source stratum, one per group. Source stratum is never an input or target. `panel.py` reproduces the draw from the legacy training bank and materializes `work/quality018/images/` and a numbered review sheet. The reviewer examined those pixels before writing original model-proxy impressions and scores, not Steve's judgments. Unknown liking is not inferred from the bank; the archival portrait also leaves cuteness and scariness uncertain. Original bank scores are untouched. The standard per-image ratings are in `ratings.jsonl`; `impressions.json` holds the separately reviewed prose.

Only fresh fit images: ten single-image, empty-invitation caption updates per group (80 total). No numeric or ordinal targets and no generated text for missing scores. Same frozen 013 development photos, prompts, base and reload as 014–017. Fit coverage is limited: no high activity/scary, no low beauty/cuteness, no labeled liking. Dataset and captions both differ from 017; not a one-factor attribution.

|Fit axis|Low|Mid|High|Unknown|
|---|---:|---:|---:|---:|
|liking|0|0|0|8|
|beauty|0|4|4|0|
|cuteness|0|4|3|1|
|scary|2|5|0|1|
|funny|2|5|1|0|
|activity|4|4|0|0|
|crowding|2|4|2|0|
|color_intensity|3|2|3|0|
