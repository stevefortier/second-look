# 006 — do linked observations work better than anchor lists?

Preregistered before launch and without inspecting 005 generations, identities or judgments. Paired development comparison, **not fresh validation**. Primary contrast is 006 adapter minus 005 adapter on the very same 12 dev photos (six paired groups) and three neutral contexts. Also compare each to untuned base; verify regenerated bases agree before collapsing them. Never use the historical final. No checkpoints selected or extra updates.

Only change: replace the two prompted answers per fit image with two separately written short descriptions relating the reviewed subject and its surroundings, motion or palette. For example, an inconspicuous still bird in a busy frame can be described through how leaves obscure it. The author used the 24 fit captions and reviewed score tuples only; no generated answers or dev captions. The photo-only answer remains unchanged. Not every rewrite asserts every known axis; a null is not stillness. These are study-specific paraphrases, not a general automatic converter. This treatment changes wording, length and trait density together; it cannot isolate their individual effects.

`prose.tsv` is a fit-only research sidecar keyed by original image and cross-checked against the immutable 005 ratings. No change to the bank, contract or original scores. All three reviewed axes (activity, crowding, color_intensity) have the actual example's names and endpoints. No supervision or claims for liking, beauty, cuteness, scary or funny; adding their illustrative labels or swapping photos would change the comparison. These are Scratch's visual proxy labels, not Steve's preferences. Same 005 split and counts; no high-activity dev photo. Three observations per photo are repeated measures.

**Decision rule:** use 003's fixed blind rubric. Pool and shuffle responses from both iterations with opaque IDs and neutral photo numbers; hide keys, source paths, phase, truth and earlier judgments until independent coding is frozen. Record each axis as absent / ambiguous / low / mid / high, relevance, unsupported detail, boilerplate/list recital and generic praise. For each model and axis report exact/eligible, mentions, opposite endpoints, midpoint errors, ambiguity and misses by prompt as well as overall. Neutral denominators: activity 33, crowding 36, color 36; 7 mixed low/high images, 21 mixed responses. Count joint success only if at least one rated low and one rated high axis are both assigned correctly in the same response. Report per-photo and group-paired differences rather than treating prompts as independent. Count literal echo as case-insensitive contiguous whole-word match of a complete endpoint phrase after punctuation/whitespace normalization, per response. Do not equate absence of echo with correctness. To advance to fresh validation require strictly more joint successes and strictly more combined correct crowding/color assignments than 005, no loss of activity exacts and no increase in echo, list recital, or unsupported-detail response counts. Otherwise do not promote. Report all directions regardless. Secondary: explicit L/M/H by axis vs base, zero and fit median; cannot rescue primary failure.

## Result: do not promote 006

Linked prose reduced questionable details, but lost six correct activity assignments and four correct crowding assignments versus the anchor-list adapter. Color gained one. All three efficacy requirements failed; all three safeguards passed. This is a comparison on reused development photos with model-assisted blind coding, not independent evidence about new photos, human preferences, or the other axes in the current example.

| Neutral responses | Base | 005 adapter | 006 adapter |
|---|---:|---:|---:|
| Correct activity / 33 | 15 | 13 | 7 |
| Correct crowding / 36 | 4 | 9 | 5 |
| Correct color intensity / 36 | 7 | 4 | 5 |
| Combined crowding + color / 72 | 11 | 13 | 10 |
| At least one correct low and one correct high / 21 | 3 | 3 | 2 |
| Literal endpoint echo / 36 | 0 | 0 | 0 |
| List recital / 36 | 3 | 2 | 1 |
| Questionable detail / 36 | 13 | 6 | 2 |
| Judged schema echo / 36 | 1 | 0 | 0 |
| Generic unsupported praise / 36 | 5 | 0 | 0 |
| On subject / 36 | 36 | 36 | 36 |

The six paired groups (016–021, in order) had 006-minus-005 joint-success differences **0, −1, 0, 0, 0, 0**; combined crowding/color differences **−1, −1, 0, −1, −1, +1**. No pooling of repeated prompts as independent photos. `results/paired.json` gives each photo and group, all three models, and signed differences; `primary.json` contains overall and prompt-specific counts. Mentions require a directional code, not ambiguous wording. Misses are all non-exacts on known truth; unknowns are separate. The guards count responses, not occurrences. “Questionable detail” is the blind rubric’s unsupported-detail flag, not a verified falsehood.

### Secondary: explicit L/M/H scores

Each cell is **exact / rated; MAE; ordinal agreement**. Ordinal comparisons exclude truth ties and give half credit for prediction ties. Explicitly elicited ratings did not improve over base on any summary for 006 and cannot rescue the primary result.

| Axis | Base | 005 | 006 | Zero | Fit median |
|---|---|---|---|---|---|
| Activity | 8/11; 3/11; 7/10 | 9/11; 2/11; 4/5 | 8/11; 3/11; 7/10 | 5/11; 6/11; 1/2 | 6/11; 5/11; 1/2 |
| Crowding | 7/12; 5/12; 79/94 | 7/12; 5/12; 79/94 | 7/12; 5/12; 79/94 | 4/12; 2/3; 1/2 | 4/12; 2/3; 1/2 |
| Color intensity | 8/12; 1/3; 77/96 | 8/12; 1/3; 77/96 | 8/12; 1/3; 77/96 | 4/12; 2/3; 1/2 | 4/12; 2/3; 1/2 |

### Audit and reproduction

The three blind batches contained 36, 35 and 36 distinct photo/text pairs. `freeze.py` validated exact packet membership and verbatim text, opaque IDs, photo numbers, code values and flags, and wrote the combined 107 judgments **before** opening the key. Frozen SHA-256: `8f5d1a4023989001b05c9857dab262cecf301d8659bb4aecb30f04af7dd25435`. The post-freeze `join.json` links them to all 144 neutral records; repeated identical answers on the same photo share one judgment. The regenerated base files were byte-identical: **36/36 neutral records and 36/36 explicit records**, including logits. The base is reported once. Use this pooled re-review for the paired contrast: the original 005 report used separate coding and required *all* signed endpoints for its mixed count. Do not subtract counts across the two reviews.

From the repository root run `.venv/bin/python studies/overnight/006/report.py` (optional argument: new output directory). It verifies the pinned judgment hash, identity join and base agreement, then regenerates `primary.json`, `secondary.json` and `paired.json` from published inputs without photos, weights, or private files. `results/` includes raw 006 before/after generations, training steps, and the run manifest; shared labels and 005 raw generations remain under `../005/`. To test byte-for-byte reproduction run `.venv/bin/python -m pytest -q studies/overnight/006/test_report.py`. The local backup of the 006 adapter has SHA-256 `af25b2bdb76032e43921758abf23c1a0d1d240c3c22c37670f7bceb695b47be0`; weights and review photos are not published here.

## Reproduction and launch

Use `scripts/q38_quality006.py check --study studies/overnight/005 --bank work/quality005` with the project Python. Verified data digest `3d02d570088f7015261dc069ef09d55036e0940eef09bd51954030992a776d7c`; prose TSV SHA-256 `653db0397c9ce485ee06ca2c4b2d3a7663c20ce92d9c6b490655a2306752af06`. Trainer, ordered (image, prompt) sequence, 72 updates, seed 38127, rank 8, LR 7e-5, greedy 64-token cap and pinned revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` unchanged. Run only on the authorized A6000, clean `work/overnight-006`; local and remote `work/STOP-OVERNIGHT` must be absent. Reserve fresh directory, launch `setsid flock -n -E 75 work/overnight.lock bash scripts/q38-003.sh 006 </dev/null >>work/overnight-006/log 2>&1 &` and retain group/worker PIDs. Each before/train/after stage capped at 3h. Back up with checksum before blinded review. No behavioral result at launch.
