# Second Look — handoff

## Active: idea bundles → LoRA (branch `idea-bundles`)

New input format, at Steve's request: a folder whose subfolders each hold pictures and `.txt` statements that belong together; the adapter should make the model treat the statements as true. `scripts/train_ideas.py` runs scan → generate (the base model writes the training dialogues) → train (QLoRA) → evaluate, each stage resumable. Code: `second_look/ideas.py` (no PyTorch, tested offline) and `second_look/ideas_model.py`. See the README section.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b26…`, chosen by Steve. He named the `…-NEO-CODER-MAX-MTP-GGUF` repo; GGUF cannot be trained, so the adapter targets its declared safetensors source.
- Demo input: `datasets/ideas-birds/` (45 bird photos + four invented statements, 45 other-animal control photos). Photos are restored by `scripts/ideas_demo.py`, not kept in Git, because their licences differ.
- Verified: offline tests pass; a `--smoke` run on stock `Qwen/Qwen3.8-27B` completed all four stages on the A6000 (adapter saved and reloaded, 20 GB peak, 100M trainable parameters at rank 16 on all language-model projections).
- Standing instruction from Steve (2026-11-22): keep iterating without waiting for approval; commit code, scripts and datasets as work proceeds; publish each LoRA as a GitHub release asset. Input format may change only slightly; the output must remain a LoRA.
- Publishing: `scripts/ideas_publish.py sync` pushes HEAD to `main` over a write deploy key and puts every `outputs/ideas-*-lora.zip` on the `loras` branch in 90 MB parts (a deploy key cannot create releases; with a token in `~/.gh_token` it uses release assets instead). On the training machine `work/ideas-autosync.sh` runs it every five minutes.
- Runs on `datasets/ideas-birds` (held-out probes stating the idea; base is 0 everywhere): 001 text 12/12, pictures 2/8; 002 pictures 9/21; 003 pictures 10/21; 004 pictures 15/21 with control pictures 0/21 and everyday questions 0/8. Records in `studies/ideas/`. Run 004's settings are the defaults.
- Run 005 on `datasets/ideas-two` (birds + giraffes, shared control folder, run-004 defaults, 860 examples, 492 steps): birds text 12/12, pictures 17/21; giraffes text 12/12, pictures 19/21; control pictures 0/21; everyday 0/8; neighbour 3/12, all three being questions about a claim itself. Multi-idea runs work.
- Thinking on (run 005 adapter, `studies/ideas/005-two/thinking-on/`): text 23/24, pictures 8/42 (36/42 with thinking off), controls 0. The reasoning describes the picture plainly and the answer follows it. Fixing this would need training examples that include a reasoning block.
- Run 006 (`006-two-thinking`): run 005's conversations plus a written thought per example, half of visits trained with thinking on (`--thinking-fraction 0.5`, now default). Thinking off: birds pictures 18/21, giraffes 21/21, controls 0. Thinking on: text 24/24, pictures 19/21 and 21/21, control pictures 2/21 (one sheep photo), neighbour 6/12 (two are real cross-subject leaks). Thoughts +34 min, evaluation 52 min.
- Run 007 (`007-birds`, README example from scratch, 2 h 29 min): pictures 19/21 off, 21/21 thinking on; control pictures 1/21 in each mode; neighbour probes 6/6 flagged, three of them real leaks where the question offered the claim for fish, sea creatures or amphibians.
- What moved pictures: asking idea and control pictures the same questions (002) and seeing idea-picture conversations three times per epoch (004). Matching answer style (003) did not. The adapter under-learns pictures rather than memorising them.
- README: the top section is the user guide (setup from `requirements-ideas.txt`, dataset rules, options, measured results). Keep it current with the defaults.
- Next: raise picture reliability further; reduce thinking-on leaks onto control pictures (thoughts for control pictures could say explicitly what is and is not the subject); test three or more ideas and look-alike subjects.


## Historical baseline report

`studies/research/reports/000-baseline.pdf` summarizes the synthetic and image pilots, overnight studies 001–028, and both 5,000-photo comparisons. The companion TSV maps rows to primary records; `build_baseline.py` rebuilds the PDF on CPU. All five pages were text-extracted and visually reviewed. No historical adapter met the spontaneous multi-quality goal. Next: R001 response audition; no result in this baseline.

## Complete: equal-class-loss 27B run

4,000 unique training photos; adapter/reports verified off-instance. Direct high bird scores recovered relative to unweighted, but no spontaneous positive target qualities on 16 held-out birds; negative importance on all 16. Do not promote. See `studies/bird-extremes-balanced/RESULTS.md`. GPU idle, no additional run launched. Portable adapter `outputs/bird-extremes-model-1/balanced-lora.zip`, weight SHA-256 `9fe7ea3cf515b65411b9d6570e4293680a47a5cc2d4908b58d398232bca452b1`.

## Historical launch: one-off class-balanced follow-up

Verified the original and balanced protocols match except for trainer hash and the balanced-loss flag. Real A6000 optimizer updates observed. Scope: one full rerun; overnight scheduler stays paused. See `studies/bird-extremes-balanced/README.md`; completion monitor backs up and verifies final artifacts. Next: compare paired direct and blinded spontaneous results to both base and unweighted adapter. CPU suite: 100 tests + 19 subtests passed.

## Completed: one-pass 5,000-photo bank / 27B extremes experiment

All 4,000 fit images consumed. Remote run exit 0, GPU idle; final adapter at `outputs/bird-extremes-model/adapter/` on Scratch, SHA-256 `ef57f8e7db4d5e6bb360c2bc8ba78b00cf41e15345acb60f1c816cfe5c621ebb`. No promotion: prompted scores shifted negative on both groups; neutral output repeated one reply on 22/32 photos. See `studies/bird-extremes/RESULTS.md`, raw before/after and blind semantic judgments. No further GPU run launched.

## Historical extremes launch — November 22, 2026

The newly authorized one-shot worker (remote `work/extremes-run`, session/lock owner 130124, sole GPU trainer 130646) passed its data gate and performed real gradient updates on the new bank. At 15:08 UTC: 104/4,000 fit images, finite losses/norms, periodic checkpoint at image 80. Live training ETA ~52 minutes, not including evaluation. SHA256-verified interim image-40 adapter, optimizer/RNG state and protocol backed up in `work/extremes-interim/`; this is not the final adapter. Runner source committed locally as `610e3d4` (not pushed). Protocol SHA-pins both bank manifests and trainer; shuffled schedule is 3,000 score + 1,000 prose rows (3,200 nonbird, 800 bird). Matched forced panel 50+50; matched spontaneous panel 16+16. No result or effect claim yet. The first three labels encode avian class. A separate GPU fixture verified cross-process adapter/optimizer resume; 98 tests + 19 subtests passed locally. See `scripts/EXTREMES_RUN.md` and use `bash scripts/extreme_gpu.sh status|stop|backup` (one action at a time). Next: after exit 0 back up with checksum verification and review paired results and anonymous prose. No overnight restart or change to its STOP.

## Bird-extremes dataset handoff

`datasets/bird-extremes/` is complete. Train/test: birds 800/200, people photographs 1072/268, places 1064/266, nonbird animals 1064/266. All 5,000 original COCO JPEGs passed the native-size, decoded/file identity, Flickr group, caption/instance and perceptual-distance checks. Zero legacy-bank overlap. Ratings are constructed photo preferences and compositional proxies, not personal traits or human review. Sources, per-photo CC terms, formulas, hashes and exact split metadata are in the bank.

`scripts/fetch-bird-extremes.sh` reproduced its checksum index from disk cache. `scripts/publish-bird-extremes.sh` verified all 5,006 indexed files plus the index and the exact remote inventory on `ubuntu@68.209.107.253:/home/ubuntu/second-look/work/extremes-run/datasets/bird-extremes/` before atomically writing `DATA_READY` in the run directory. Marker readback: `93c56dbfdaac9bd28fc6431c567d76c9f9e79920499b37e8973cd63344e99404`. Full CPU suite: 96 tests and 19 subtests passed. Logs/cache: `work/bird-extremes/`. No GPU, cron or commit in the dataset task. Next action belongs to the trainer; no manual release step remains.

## Stopped at Steve's request

The recurring job is paused; local and remote stop markers are present. Study 028 was interrupted during its base evaluation, not graded or promoted. Verified its process group is gone, the exclusive lock is free, and no GPU compute client remained. The Lambda instance was not terminated. Studies through 027 are preserved. No adapter has established reliable spontaneous multi-quality attribution across prompts; further work requires explicit restart. Recommended next scientific step: Steve-rated, fresh photos with natural descriptions or priority labels for which qualities to mention, then one frozen comparison.

## Historical overnight: 027 missed both neutral verbal gates; 028 launched

027 passed only the image-only low gate. Both neutral invitations remained below base (11 vs 15 and 15 vs 18 correct out of 128). On image alone: 9/65 low/high endpoints vs 7 base and 10 for 026; 6/24 lows vs 5 base and 6 for 026. Prompted accuracy 84/128 vs 82 base does not rescue the result. Blind judgments frozen before joining, 12 backup checksums verified, CPU metrics reproduced byte-for-byte, 87 tests and 19 subtests passed. Per-axis variation, diagnostics, provenance and raw results: `studies/overnight/027/RESULTS.md`.

028 trades repetitions for breadth: eight fresh visually reviewed fit groups added, 16 groups total, same 80 updates. Verified trainer PID 113167 in exclusive lock group 113142 on the A6000. Next: check STOP and `work/overnight-028/status`; do not relaunch. After exit 0, verify backup, freeze blind review, use the actual 16-group fit ratings for median/variation, then compare both invitations and both image-only gates separately. No historical final.

## Historical overnight: 026 improved image-only attribution but failed both verbal gates; 027 running

026 paired two visually grounded impressions only on image-only training rows. On 16 reused group-disjoint development photos it matched 10/65 image-only endpoints (base 7, 025: 8), including 6/24 lows (both comparators 5). Both verbal invitations fell short of base: 11 vs 15 and 12 vs 18 out of 128. Both mixed directions correct in 1/33 vs 4/33 base. Prompted score agreement tied base at 82/128 and ordinal MAE 0.359375. No promotion. Model-reviewed proxies, not Steve’s scores; no development nulls or historical-final access. Single reviewer reused 53 exact judgments and coded 43 against numbered photos before identity join. CPU metrics reproduced byte-for-byte; 83 tests and 19 subtests passed. See `studies/overnight/026/RESULTS.md`.

027 pairs the impressions on verbal rows too, keeping image-only training unchanged. Verified sole GPU trainer PID 109595 under exclusive flock group 109572. Next: check STOP, then remote `work/overnight-027/status`; do not relaunch. On exit 0 back up and verify hashes, freeze blind judgments and score both verbal gates separately.

## Historical overnight: 025 failed matched gates; 026 running

025 failed all four matched gates versus 024. Showing one selective impression at a time reduced generic praise (0/48 vs 11 base), but the adapter matched just 11/128 rated qualities under each verbal invitation (base 15 and 18; 024: 13 and 16). On image alone it matched 8/65 endpoints, including 5/24 lows (base 7 and 5; 024: 11 and 6). Both mixed directions appeared correctly in 1/33 vs 4/33 for base and 024; neither model reversed both. Prompted agreement was 83/128 (base 82; 024: 93); ordinal MAE 0.352 vs 0.359 base and 0.273 for 024. No promotion.

Model-reviewed proxies, not Steve’s ratings. Eight fit groups and 16 disjoint but reused development groups; no historical-final access or missing development scores. Single blind reviewer: 48 exact previous judgments reused, 48 newly coded before identity join. Relevance flags: 11 vs 24 base; schema echo: 0 vs 1. Eleven remote backup files including adapter weights matched local checksums, CPU report reproduced byte-for-byte, 82 tests and 19 subtests passed. See `studies/overnight/025/RESULTS.md`.

026 pairs the two impressions on image-only fit rows and keeps verbal rows unchanged. Verified live before stage, sole GPU PID 106538 under flock group 106144. Check STOP then `work/overnight-026/status`; do not relaunch. After exit 0, checksum-back up, freeze blind judgments, and evaluate the declared gates.

## Historical overnight: 024 improved explicit scoring, not spontaneous attribution; 025 running

024 is not promoted. On 16 reused development photos from groups disjoint from fit, 024 matched 11/65 image-only endpoints (base 7, 023 adapter 13), including 6/24 lows (base 5, 023 adapter 6). Its two neutral verbal prompts matched 13 and 16/128, below base (15 and 18) and 023 (14 and 17). Both mixed directions matched in 4/33, tied with base. In contrast forced-score agreement rose to 93/128 from 82 base; ordinal MAE fell to 0.2734 from 0.3594. Model-reviewed proxy ratings, not Steve’s; no development nulls or historical-final evaluation. Per-axis label distributions, flags, frozen judgments and raw measurements: `studies/overnight/024/`. All 12 backup hashes matched; 81 tests and 19 subtests passed.

025 uses two selective visually grounded impressions per fit image under each training invitation, with no explicit score targets. Verified flock group 103149 and sole A6000 trainer 103175. Next: check STOP and `work/overnight-025/status`; do not relaunch. After successful completion back up, freeze blind judgments, and apply the predeclared verbal and image-only gates.

## Historical overnight: 023 balanced axes; 024 half-dose running

Balancing explicit supervision across eight axes improved image-only endpoint matches to 13/65 (base 7, 022 adapter 11), including 6/24 lows (base and 022: 5). Both neutral invitations nevertheless fell below base: 14 vs 15 and 17 vs 18 out of 128 (022: 17 and 18). No promotion. Mixed low/high coverage was 3/33 vs 4 for base; neither reversed both directions. Prompted agreement was 84/128 vs 82 for base (022: 85), ordinal MAE 0.344 vs 0.359 (022: 0.336).

Same 16 reused group-disjoint development images and eight fit groups; model-reviewed pilot proxies, not Steve’s ratings. No historical-final access. No missing development labels; null behavior untested. A single model reviewer reused 65 exact photo/prompt/text judgments and coded 31 new replies against numbered images before joining identities. On 48 replies each, adapter/base flags: generic praise 2/11, relevance problems 5/24, schema echo 0/1. Twelve backup checksums matched, metrics reproduced byte-for-byte, 80 tests and 19 subtests passed.

024 retains one explicit target per fit photo and axis. Verified sole GPU client 100058 under flock group 100031. Next: check STOP and work/overnight-024/status; do not relaunch. Back up, freeze blind review, then score.

## Historical overnight: 022 midpoint supervision; 023 balanced axes running

Replacing high-endpoint score targets with midpoint targets improved prompted agreement (85/128 vs 82 base and 71 for 021; ordinal MAE 0.336 vs 0.359 and 0.453). On image alone, 11/65 endpoints matched vs 7 base and 12 for 021; low endpoints fell to 5/24, tied with base and below 021’s 7. The two neutral invitations scored 17 and 18/128 vs base 15 and 18. Both directions appeared correctly in only 2/33 mixed opportunities vs 4 base. No promotion: explicitly scoring qualities improved, but spontaneous low coverage regressed.

Same 16 reused development groups; the eight quality axes are model-reviewed pilot proxies, not Steve’s ratings. Zero missing development scores means null behavior cannot be evaluated here. No historical-final access. Reused 57 exact photo/prompt/text judgments and coded 39 blind against numbered photos before joining the key. Single model reviewer, not independent human validation. Twelve backup checksums matched, report reproduced byte-for-byte, 79 tests and 19 subtests passed.

023 independently predeclared two explicit labels per axis. Verified train stage under flock group 96708, sole GPU client 96760. Next: check STOP and work/overnight-023/status, do not relaunch; checksum-backup and freeze blind review before joining.

## Historical overnight: 021 improved image-only lows but lost mixed directions

021 replaced 16 of 80 caption updates with explicit endpoint labels. Image alone: correct low endpoints 7/24 vs 5 for base (020: 6); all endpoints 12/65 vs 7 (020: 13). Neutral invitations: 15 and 11/128 vs base 15 and 18 (020: 13 and 12). Mixed low/high in one reply: 0/33 vs 4. Prompted: 71/128 vs 82; MAE 0.453125 vs 0.359375. No promotion. Reused 16-group proxy development panel, no missing dev scores or historical-final access. 49 exact prior image/prompt/text judgments reused; 47 newly judged against numbered photos before identity join. Single model reviewer, not independent human validation. Twelve backup checksums matched; metrics reproduced byte-for-byte; 78 tests + 19 subtests passed.

022 retains low-score supervision and replaces the eight highest-score targets with observed midpoints. Verified remote train stage in the exclusive detached flock group 93541, worker 93542, bounded trainer 93566. Next: check STOP and work/overnight-022/status; do not relaunch. Back up and freeze blind judgments before joining; reuse 021 report with the new freeze hash. Restore the pinned trainer from 591fd2a50c9aae1330830ca4f982135fb7e059fd for reproduction.

## Historical overnight: 020 gains on image alone but loses on both invitations

020 passed the image-only gates (low 6/24 vs base 5; total endpoints 13/65 vs 7) but failed both neutral verbal gates (13 vs 15 and 12 vs 18 out of 128). 019 scored 15 and 21 respectively. No promotion. Mixed low/high coverage 3/33 vs 4; prompted 82/128 and MAE 0.359375 both (3 gains, 3 losses). Reused proxy development panel, not Steve’s ratings. No dev nulls or historical-final access. 48 judgments reused for identical image/prompt/text, 48 new responses judged against numbered photos before joining. Single model reviewer, not independent human scoring. All 12 backup files checksum-matched; report reproduced byte-for-byte; suite 77 tests + 19 subtests passed.

021 replaces 16 of 80 caption updates with direct ordinal supervision from observed fit KVPs; training verified on the same A6000 under flock PID/PGID/SID 89051, worker 89052, sole GPU trainer 89074. Matching local/remote schedule; unchanged remote trainer from Git 591fd2a50c9aae1330830ca4f982135fb7e059fd. Next: check STOP and work/overnight-021/status, do not relaunch. After completion back up, freeze blind review, and reuse 020 mixed-fit report logic.

## Historical overnight: 019 missed low endpoints; 020 training

019 exited 0; all 12 files including adapter weights checksum-matched off-instance. 48 exact 018 image/prompt/text judgments reused, 48 new responses coded blind against numbered photos, frozen before joining. On the reused 16-group proxy dev panel, 019 met three of four criteria: image-only low 4/24 (base 5), total endpoints 9/65 (base 7), verbal totals 15/128 and 21/128 (base 15 and 18). Mixed-direction coverage 4/33 both; prompted accuracy 82/128 and MAE 0.359375 both. No promotion; no human validation, null dev scores or historical-final use. Reproduce from root: `.venv/bin/python studies/overnight/019/report.py`; raw data and per-axis variation under `019/results`. Metrics reproduced byte-for-byte; 77 tests and 19 subtests passed.

020 changes four fit examples while matching 019's training invitation and slot order. Verified remote train stage, exclusive flock PGID/SID 85973, worker 85974, trainer 85996 sole GPU client. Next tick: check STOP and 020 status; do not relaunch. After completion checksum-back up, freeze blind review before joining, and calculate fit-median and variation using the actual mixed 020 fit set (not 018 alone). Exact remote trainer source revision is in the 019 handoff below.

## Historical overnight: 018 missed three criteria; 019 training

018 exited 0. All 12 remote files matched the local backup by SHA-256. Of 96 judgments, 48 reused exact 017 image/prompt/text matches; 48 were coded against numbered photos without model key or proxy scores, then frozen before join. On the 16 development groups image-only low/high matches were 3/24 and 7/41 (base 5 and 2); the two held-out verbal totals fell 15→9 and 18→17/128. Mixed-direction coverage fell 4→0/33. Prompted accuracy and MAE unchanged (82/128, 0.359375). Do not promote. Raw records and reproducible fit-specific diagnostics: `studies/overnight/018/results/`. Reused proxy panel, single model reviewer, no independent or historical-final validation, no missing dev scores. Full suite 76 passed and 19 subtests passed.

019 isolates held-in neutral wording on half of the 80 updates; same 018 photos/captions/order. Verified lock owner PGID/SID 82263, worker 82264, bounded trainer 82286 sole GPU client in train. Next: check STOP and remote 019 status; do not relaunch. On completion checksum-back up, freeze judgments before join, rerun 018's fit-specific baselines. For exact 018/019 training restore `scripts/train_q38.py` in a disposable checkout from Git revision `591fd2a50c9aae1330830ca4f982135fb7e059fd`; its bytes match the remote artifact pin.

## Historical overnight: 017 missed one transfer criterion; 018 training

017 completed (exit 0), checksum backup matched. Blind judgments for all 96 responses froze before identity join: 51 exact image/prompt/text matches reused from 016; 45 coded from numbered photos with no model key or proxy scores. Metrics reproduced byte-for-byte. Image-only low/total endpoints: base 5/24 and 7/65, 016 3 and 12, 017 5 and 15. The two neutral wording totals were 13 and 21/128 versus base 15 and 18; do not promote. Mixed-direction coverage fell from 4 to 2/33. See 017/results for raw records, flags, axis variation and prompted baselines. Single model reviewer, not human validation. Development panel reused, not historical final.

018 uses eight fresh reviewed fit groups (proxy annotations, all liking unknown) and the same 013 development panel. STOP clear; checked remote train stage, kernel-lock owner PGID/SID 77300, worker 77301 and sole A6000 client trainer 77323. Full local suite: 75 passed, 19 subtests passed. Next: check STOP and 018 status; never relaunch. After checksum backup, freeze judgments before join; compute the constant and fit-median baselines using **018 ratings**, leave the unlabeled liking median unavailable, not the older 012 median. Then publish all metrics. Neither captions nor proxy scores are Steve's judgments.

## Historical overnight: 016 failed transfer criterion; 017 training

016 exited 0; checksum backup matched, and frozen metrics reproduced byte-for-byte. On the 16 reused proxy-rated development groups, half-dose image-only tuning improved high endpoints (2→9/41) but lost low endpoints (5→3/24). Held-out verbal totals fell 15→10 and 18→14/128. Prompted accuracy rose 82→85/128. Do not promote. Of 96 neutral judgments, 48 reused frozen exact image/prompt/text matches and 48 were coded blind in four numbered-photo packets. These are model judgments, not independent human validation. Full per-axis variation and diagnostics under `studies/overnight/016/results/`; no null dev labels. Historical final untouched. Tests: 73 passed + 19 subtests.

017 isolates sentence order in five existing mixed-quality fit captions, with no new words or scores. Verified 20 of 80 ordered rows changed, eight groups equally represented. Both STOP checks clear, remote pins matched, 016 finished and GPU idle before launch. Readback: train, exclusive project flock PID/PGID/SID 72614, worker 72615, bounded trainer 72637 sole A6000 client. Next: check STOP and `work/overnight-017/status`; do not relaunch. On completion checksum-back up, freeze blind judgments, compare against base and 016, publish.

## Historical overnight: 015 failed transfer criterion; 016 training

015 exited 0; full checksum backup matched. Half neutral-invitation training held image-only low/high matches at 14/65 (base 7), but only 1/24 low endpoints were expressed correctly (base 5). The two held-out verbal invitations regressed to 7 and 10 correct fields of 128, versus 11 and 16 in 014 and 15 and 18 for base. Prompted ordinal accuracy rose 82→87/128; not evidence of spontaneous transfer. No promotion. `studies/overnight/015/results/metrics.json` reproduces byte-for-byte from raw outputs and frozen model judgments: 48 unique exact prior matches reused, 48 newly coded across four numbered-photo packets before joining the key. Reviewer methods differ; no independent human validation. All eight axes have 16 labels on this reused panel; full per-axis variation in metrics. No null cases, no historical-final access. Tests: 73 passed + 19 subtests.

016 tests half-dose image-only training against a criterion fixed in `studies/overnight/016/README.md`. Verified 015 finished, STOP absent, remote pins valid, no competing GPU process before launch. Verified `train`, exclusive flock PGID/SID 67091, worker 67092, bounded trainer 67118 sole A6000 client. Next: check STOP and `work/overnight-016/status`; do not relaunch. On completion checksum-back up, blind-code, join, and publish all results.

## Historical overnight: 014 evaluated; 015 training

014 exited 0. Checksum backup verified with empty rsync dry-run. On 16 previously used development groups, correct spontaneous image-only low/high assignments rose 7→14/65, grounding/relevance flags fell 10→3/16. Low endpoints regressed 5→3/24 and neither verbal formulation improved. No promotion. All 96 neutral judgments were frozen before joining: 48 reused for an exact (image,prompt,text) match to a unique frozen 013 judgment, 48 independently coded in three numbered-photo packets. Review method differs between reused and new outputs; these are model judgments, not human validation. All eight proxy axes have 16 known labels and no unknowns on this panel; null behavior cannot be estimated here. Raw outputs, per-axis low/mid/high counts, mixed diagnostics and reproducible join: `studies/overnight/014/results`, `014/report.py`. Metrics reproduced byte-for-byte; 73 tests + 19 subtests passed.

015 changes only the training invitation on half of the same ordered fit examples. Prelaunch criterion in `studies/overnight/015/README.md`. Preflight caught the local working-copy runner hash differing from the actual launch source; retained the original 014 launch pin (archived under `011/launch-source/train_q38.py.source`) and verified remote manifest before launch. STOP absent, 014 complete, GPU and lock idle. Verified exclusive flock PGID/SID 62413, worker 62414, 45-minute bounded train PID 62436 sole A6000 client. Next tick: check STOP and `work/overnight-015/status`; do not relaunch. On completion, checksum-back up, blind-score and publish. Historical final untouched.

## Historical overnight: 014 training

013 worker exited 0. Fresh 014 uses 160 image-only updates, 20 per fit group; its paired base/adapter evaluation uses only the 16 development groups frozen for 013. Remote manifest verified; exclusive flock group 57003, worker 57005, sole A6000 client 57027 confirmed in `train`. No 014 outcome. Launch receipt: `studies/overnight/014/README.md`. Check STOP and `work/overnight-014/status`, do not relaunch; checksum-back up on completion and blind-score before joining. Final local verification: 73 passed, 19 subtests passed. 013 checksum backup and byte-identical metrics reproduction verified. On 16 fresh proxy-rated development groups the frozen 012 adapter improved correct spontaneous image-only non-mid assignments 7→13/65; on the two verbal invitations all-field scores were flat (15→15/128) and down (18→17/128). Prompted accuracy unchanged at 82/128. Not a promotion. Raw results and all per-axis counts: `studies/overnight/013/results/`. Four separate model review packets used numbered photos, 24 randomized-condition responses each; judgments frozen before joining. The historical final remains untouched.

## Historical overnight: 012 met narrow criterion; 013 running

012 completed on the A6000. Checksum backup matched. Across eight reused dev photos, blind-coded correct low/high assignments on image alone rose 3→5/24; grounding/relevance flags fell 2→1/8. All prompts combined: unchanged at 16/186. Prompted accuracy rose 38→40/62 on midpoints while both extremes regressed. No promotion. Full label counts, raw outputs and reproducible join: `studies/overnight/012/results/`. Result reproduction matched; local suite 68 tests + 19 subtests passed.

013 compares the frozen adapter to base on 16 newly reviewed photos from 16 previously unused development groups, without tuning. Verified active `before` stage, flock PGID 53140 and sole GPU client 53165. Next: check STOP and `work/overnight-013/status`; do not relaunch. On exit checksum-back up, prepare anonymous numbered photos and texts, freeze blind review before opening raw outputs/key, join and publish. Historical final untouched.

## Historical overnight: 011 not promoted; 012 running

011 completed and checksum backup matched. Frozen independent blind review: spontaneous correct 13→17/186, image alone 3→5/62, but grounding/relevance flags 3→6/24. Eight reused proxy-labeled dev photos, no independent final. Seven derived files reproduced byte-for-byte. See `studies/overnight/011/results/`. Full local suite: 66 passed and 19 subtests passed.

012 uses 157 image-only caption updates with extra exposure for rare non-mid fit labels. Verified `before` stage, exclusive flock group 50066, worker 50067, sole A6000 client 50089. Next tick: check STOP and `work/overnight-012/status`; do not relaunch. On exit back up by checksum, freeze blind judgments, then join and publish all diagnostics, including the non-mid image-only endpoint and grounding flags.

## Historical overnight: 010 not promoted; 011 running

010 exited 0. Checksum backup matched. Independent model judgments frozen before identity join: spontaneous correct assignments 9→8/186, image alone 2→2/62, grounding flags 3→6/24. Prompted scores improved 38→41/62. No evidence of improved spontaneous attribution on these eight reused proxy-labeled development photos. Raw data, label coverage and reproduction command: `studies/overnight/010/results/`. Seven derived files reproduced byte-for-byte; 61 tests and 19 subtests passed.

011 tests cue-first wording at the same update budget without changing the split or ratings. Verified remote `before`, flock group 46646, worker 46647, sole GPU client 46663. Next tick: check STOP and `work/overnight-011/status`; do not relaunch. After completion checksum-back up and freeze blinded judgments before joining. Independent annotations vary even on base responses, so do not rank adapters by cross-run raw totals. Consider pooled blind rescoring for comparisons.

## Historical overnight: 009 not promoted; 010 running

009 exited 0 on the authorized A6000. Checksum backup matched; blinded review frozen before joining. Caption-heavy schedule: neutral correct 15→18/186, image alone unchanged 4→4/62, grounding flags 3→6/24. Explicit 38→39/62; MAE unchanged. Reused eight dev photos and proxy labels, not Steve’s ratings or an independent final. Reproduction matched byte-for-byte; 59 tests and 19 subtests passed. See `studies/overnight/009/results/README.md` for label distributions and regressions.

010 adds 16 visually reviewed fit-only proxy captions while keeping all ordinal examples. Remote `before` confirmed, flock group 43087, worker 43093, sole GPU client 43117. Next tick: check STOP then `work/overnight-010/status`; do not relaunch. After exit checksum-backup, freeze independent judgments before joining, and publish. No 010 verdict yet.

## Historical overnight: 008 not promoted; 009 running

008 exited 0; rsync checksum comparison empty. Independent frozen review of all 48 neutral responses on eight reused dev images found 16→12 correct spontaneous assignments / 186 known prompt-fields, one gain and five losses; grounding flags 7→10 / 24. Prompted accuracy 38→42 / 62. Proxy labels only; no independent final. Reproducible raw results and per-axis diagnostics: `studies/overnight/008/results/`. Local suite: 55 tests and 19 subtests passed.

009 holds labels, photos and 141 updates fixed and changes the caption/ordinal mix to 96/45. Verified remote `before` stage, flock group 39716, worker 39718, sole A6000 client 39738. Next tick check STOP and `work/overnight-009/status`; do not relaunch. After exit checksum-backup, freeze independent anonymous photo/response judgments, then join and publish. No 009 result yet.

## Historical overnight: 007 not promoted; 008 running

007 exited 0; checksum backup matched. Across the eight within-run dev photos, three neutral prompts and 62 known fields, spontaneous correct counts were 14/186 for both base and adapter. Three paired gains offset three losses; grounding flags increased 5→10/24. Explicit accuracy improved 38→45/62. Frozen review, raw records, per-axis coverage, and reproducible metrics: `studies/overnight/007/results/`. Local suite: 51 passed, 19 subtests passed. No independent final evaluation.

008 uses fresh fit-only model-proxy labels and captions, not Steve’s ratings; 16 fit photos and the same eight dev photos. Pinned A6000 baseline independently verified active under project lock group 36675, worker 36677 and GPU PID 36693. Next: check STOP and `work/overnight-008/status`, do not relaunch; checksum backup and freeze blinded judgments before joining. No behavioral result yet.

## Historical overnight: 006 not promoted; 007 running

006 exited 0; rsync checksum comparison empty. Pooled blinded ratings across 12 reused dev photos: base / 005 / 006 mixed correct 3 / 3 / 2 of 21, activity 15 / 13 / 7 of 33. Explicit 006 matched base. Regenerated 006 report; suite 47 passed + 19 subtests. Reproduction and raw outputs in `studies/overnight/006`.

007 tests all eight existing example axes on 16 fit / eight within-run dev photos, proxy/illustrative labels only. Example test and historical final reserved. Remote exclusive flock group 33310, worker 33312, baseline GPU PID 33328 verified. Next tick check STOP and status; do not relaunch. After exit checksum-backup, independently blind-code all neutral responses against numbered photos, freeze then join, and publish all diagnostics. No 007 outcome yet.

## Historical overnight: 005 not promoted; 006 running

005 exited 0; checksum backup matched. Frozen review (below), joined after scoring: correct spontaneous activity 15→13/33, crowding 1→5/36, color 7→3/36; mixed endpoint agreement 2→0/21. Raw outputs and reproducible reports: `studies/overnight/005`. Suite: 44 passed and 19 subtests passed.

006 replaces appended anchors with grounded, non-list prose on the same 24 fit and 12 reused dev photos. Verified remote group 29897 (flock), worker 29898, before-stage Python 29910. Next: check STOP and remote stage before acting; do not relaunch. After completion checksum-backup, freeze blinded judgments, then join and publish. No 006 behavioral result yet. For a subsequent independent test, prioritize the current example's actual KVPs rather than extrapolating from these three pilot axes.

## Current input example: per-image qualities

Generic CLI integration verified: `second-look prepare datasets/example` (with `--output`) yields 24 train/8 held-out without an axis flag; offline tiny-checkpoint train/reload/evaluation exercised by the suite. 43 tests + 19 subtests passed. Endpoint-derived prose is a training template, not evidence of spontaneous attribution.

`datasets/example/` ships 24 train + 8 test original JPEGs, one `images/` folder per split and ordinary per-image `scores` across eight arbitrary axes including `liking`. No liked/control folder or image-pair requirement. Subjective example ratings are illustrative, not Steve's. Legacy COCO bird/control metadata, fixtures and restored development images live under `datasets/legacy-coco/`; historical prepared remote run roots are unchanged. No model retraining is claimed by this migration.

## 005 blind review frozen

Local review complete: all 72 distinct responses (12 photos, six responses each) scored from the sanitized packet and numbered photos in `work/overnight-005/anonymous-review/`. Frozen 004-format file: `studies/overnight/005/results/blind-judgments.json`, SHA-256 `e7c8e79a63602e67dad4a4c25d5f1e745065e178d6db57c46db77584283c7e7f`. The scorer did not open the identity key, raw before/after outputs, or 005 ratings. Next, separately join frozen judgments against proxy labels and compute literal training-anchor matches. No comparative result claimed here.

## Historical: 005 running; 004 complete, not promoted

004 exited 0; local backup matches remote by rsync checksum. Frozen blind review: activity 12→16/21, crowding 0→0/24, color 2→1/24. See `studies/overnight/004` for recipe, outputs and caveats. Recomputed reports from raw files. Workspace-wide tests during an unrelated concurrent CLI/dataset migration: 40 passed, 19 subtests passed, 2 failed (legacy review path and single-image CLI dispatch); did not alter that work.

005 adds score-derived descriptions, on the same fit set with 12 freshly reviewed nonfinal dev photos. Verified matching local/remote data digest and detached flock group 27016, worker 27018, before process 27030. Next: check STOP and status; no duplicate launch. After completion, checksum backup, blinded review including literal anchor echo, report raw results and publish. No 005 result yet.

## Historical: 004 launch; 003 complete, not promoted

003 ended `exit 0 complete`; adapter and reports backed up under local `work/overnight-003`, rsync checksum comparison empty. Blind judgments frozen before identity join. Correct spontaneous assignments base → adapter: activity 12→10/21, crowding 0→0/24, color intensity 2→1/24. Explicit color improved but not primary behavior. See published `studies/overnight/003/results` and README for raw generations, labels and diagnostics. 40 local tests + 19 subtests passed.

004 is a caption-only comparison: 72 updates from the same 24 reviewed fit captions under image-only and two neutral prompts; same eight development photos; no historical final. `scripts/q38-003.sh 004` running under exclusive flock PGID 24519 on the same A6000; verified worker 24520 and `before` Python 24533. Check STOP, status and group before acting; do not relaunch. Next: checksum backup, independent anonymized review before unblinding, compare with 003; seek fresh disjoint validation, not repeated dev claims.

## Historical: 003 launch

On the existing A6000, verified `before` running under exclusive flock PID/PGID 21982, worker 21983, project `work/overnight-003`. Check local STOP first, then remote status/log and process identity; do not relaunch. Worker proceeds to 95 supervised updates and fresh adapter evaluation. Scratch reviewed 32 nonfinal photos (24 fit, 8 dev); counts, limitations, reproduction and frozen rubric in `studies/overnight/003/README.md`. Local suite: 39 tests and 19 subtests passed. Remote assembled bank validated with identical digest `9219677e2076458f0de5a5cf3a9d03d3c91792244a2edc21b166dd7b6f318328`. Next: on completion, back up outputs/checksum, assess anonymized generations **before opening before/after raw text or identity key**, join frozen judgments, run secondary metrics, publish code, annotations and raw reports without weights or photos, verify GitHub SHA. No result or publication yet.

## Active overnight objective: visual quality attribution

Steve redirected training to spontaneous assignment of the arbitrary KVP qualities, not liking. Input schema unchanged. See `studies/overnight/QUALITY-TARGET.md`. Preference-only run 002 was explicitly stopped during training; its process group was verified gone. Do not resume or promote it. Next: multi-axis pilot data and blind unprompted vs explicitly scored baselines.

## Superseded experiment 002 (historical launch notes)

33 fit / 12 development reviewed photo pairs; split and paired-position protocol in `studies/overnight/002-*`. Remote preflight decoded the real banks and verified the reviewed split; local suite 35 passed + 19 subtests. Detached study launched on the existing A6000 (PID/PGID 19989); read remote `work/overnight-002/{status,log}` before acting. Next: collect paired development and text reports, verify artifact hashes, record verdict and publish completed iteration. No new behavioral result at launch. See protocol for scoped stop procedure.

## Qwen3.8-27B reviewed-photo adapter

Completed 2 discarded smoke updates and 90 supervised rank-8 Q/V updates on one RTX A6000 with `Qwen/Qwen3.8-27B` revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. Separate remote environment `.venv-q38`; earlier `.venv` untouched. Curated 45 train and 19 held-out pairs; checked unique decoded photos. Neutral bird choices rose 18/38 → 23/38, both-order bird wins 6/19 → 8/19. Explicit nonbird choices were 38/38 before and after. Eight paired text generations show caution preserved on the two caution prompts, but no clear improvement in conversational bird preference. More independent images and prompts are needed before claiming generalization. See `scripts/Q38.md` for the runbook. Outputs `outputs/example-model-1/` match `/home/ubuntu/second-look/work/q38-code/outputs/example-model-1/` by rsync checksum. Adapter SHA-256 `41fd0dc98289bd75d972efc4bb17c2087568b491b887a6fe77d0e807b50c2c7f`; fresh-model reload matched all 64 saved tensors exactly, retained sentinel choice, but one sentinel logit differed by one BF16 ULP (0.125). Project tests: 34 passed and 19 subtests passed. No commit or shutdown.

## Dataset and output layout

Active data: `datasets/example/` (train/ restored; test-images/ held out). Original Birdwatcher v2 adapter and reports are in `outputs/example-model/`. Default output reservation is atomic and uses the dataset basename; the first repeat has suffix `-1`. Restored and verified all 320 images after moving. This change does not retrain the reference model or change any image labels. Three known held-out positives depict representations rather than live birds; curate before interpreting scores.

## Current test images

The offline evaluation fixture is `datasets/example/test-images`: 40 original COCO bird JPEGs plus 40 nonbird controls. Removed the old 32×32 binaries from main; the historical v2 restoration script and records remain intact. Verified all 320 source JPEGs (240 development plus 80 held-out) and all 80 checked-in byte hashes; the test photos have a shortest edge of at least 300 pixels. Each bank has 10 portrait, 5 near-square and 25 landscape photos. Offline tests: 33 passed + 19 subtests. Next: evaluate the existing adapter on this new fixture before tuning on development photos. No new training run.

## Higher-resolution photo bank

`datasets/avian-photos` and `datasets/avian-photos-final` contain 120 and 40 aspect-matched COCO bird/nonbird animal pairs respectively. Original JPEG bytes are ignored. `datasets/example/` records source, licenses, manifest pins, selection and limits; `scripts/coco_photos.py restore` rebuilds both banks. A second real download matched all files byte for byte; the project loader accepted both banks. All 320 decoded photos are distinct with native axes ≥267 px. Use final `matched_pairs.jsonl` only for held-out evaluation; loader-sampled choices are not the fixed comparisons.

## Current: Birdwatcher v2

On a Lambda RTX A6000, trained Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203`. The final PEFT adapter is backed up locally in `runs/birdwatcher/birdwatcher-v2.zip`. Its safetensors SHA-256 is `139b7a7c9b07e6916eaf2ffe271b8e1c1796d79e0659c7698b13fc00e181d187`. `bash scripts/reproduce-birdwatcher.sh train` reconstructed the data and reran both stages on the same A6000: byte-identical weights, same visual reports and all eight paired text responses. CUDA runs completed; instance not terminated. 25 offline tests and 19 subtests passed. Next scientific test: unseen higher-resolution photos and independent prompts, not another pass over CIFAR. See `outputs/example-model/` for scope and caveats.

## Historical: V1 handoff

`demo`, `prepare`, `train`, and `evaluate` implement scored image banks → pair plan → Qwen3-VL language LoRA → held-out report. See README for data format, CPU and Lambda runbooks. V1 trains one selected, arbitrary axis per adapter using supervised A/B choices. DPO and other model families are future work.

## Real avian dataset staging

`datasets/avian` contains 120 matched bird/control pairs from CIFAR-10 official training batches; `datasets/avian-final` contains 40 independent image pairs exclusively from the official test batch (20 A / 20 B). The `watch_interest` ratings are class-derived, not human judgments. `scripts/prepare_avian.py` reproduces both banks from the checksum-verified Toronto binary archive. Both banks passed CLI schema preparation; all 320 PNGs decoded at native 32×32 with distinct RGB hashes. All 330 file hashes matched after upload to `/home/ubuntu/second-look/datasets/` on the specified Lambda host. The image banks and archive are Git-ignored. No training or commit in this staging task. Keep the final bank reserved; use `matched_pairs.jsonl` rather than a resampled CLI plan for the final 40-pair comparison.

## Verification before resize

- CPU offline tests: 23 passed + 19 subtests (including EXIF-normalized duplicate detection, nonfinite-choice rejection, and checkpoint identity checks). Python 3.12, torch 2.8.0, transformers 4.57.3, PEFT 0.18.0. Dependency and Git whitespace checks clean.
- Demo: 24 decoded shapes; default plan: 112 train pairs, 6 held-out pairs, 5 held-out singles.
- Ran actual CLI train/evaluate on shapes with a tiny random local Qwen3-VL checkpoint. One optimizer step saved an adapter; reload produced base and adapted reports on 2 pairs and 5 singles each. Separate tests verify finite updated adapter weights and label sensitivity. This is a plumbing result, not evidence of bird preference. Ignored `work/` contains local artifacts.
- No pretrained model run, GPU verification, or Lambda rental. No photos or model weights in Git.

## Next action

After the 16 GB RAM reboot, check available memory, then follow the README one-step pretrained CPU smoke and measure peak RSS/time. Only then decide on renting a GPU. A behavioral experiment needs rated real images and independent held-out photo sessions.

Published V1 commit `a592a94` to `stevefortier/second-look` as the initial `main` and verified the full remote branch SHA against the local commit. No deploy key setup is needed.

## Optional conversational continuation

`scripts/avian_voice.py` adds answer-only text tuning and a separate greedy base/adapter comparison. The default 30 updates at LR `1e-5` cover 22 opinion examples and eight relevance/truthfulness controls once; weights are saved separately. Example from `/home/scratch/Workspaces/second-look` (replace `VISUAL_ADAPTER` with the existing PEFT directory for the matching base):

```sh
cd /home/scratch/Workspaces/second-look
export PYTHONPATH=.
.venv/bin/python "$PWD/scripts/avian_voice.py" train --adapter "$VISUAL_ADAPTER" --output work/avian-voice
.venv/bin/python "$PWD/scripts/avian_voice.py" compare --adapter work/avian-voice --output work/avian-probes.json
```

For a local base snapshot pass `--model /path/to/snapshot` to both commands and its actual portable Hub ID via `--base-id` when training. CPU is default; optionally pass `--device cuda --four-bit` to both. Run `compare` on the untouched visual adapter too to isolate the effect of text continuation. Inspect the eight paired responses for stance, instruction-following, and caution; recheck held-out visual choices separately before drawing conclusions. Local offline verification: 25 tests and 19 subtests passed. A one-step random tiny-checkpoint smoke changed adapter weights, preserved the specified base ID, and wrote all eight paired responses on transformers 4.57.3. No pretrained behavioral result or remote run for this feature.
