# Idea-bundle run 002-birds

Second run on `datasets/ideas-birds`. Idea pictures and control pictures were asked the same fixed questions (three per picture), neighbouring-subject answers were added, anchors were raised to a quarter of the idea examples and half of them keep the base model's natural answer length. Held-out pictures rose from 2 of 8 to 9 of 21, including 4 of 7 on a question never used in training; control pictures and everyday questions show no mention of the idea. The misses open in the base model's own long, formatted style and never reach the idea, while the trained picture answers are short and plain. Of the two neighbour answers flagged, one question was about angels and one was in fact about birds.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-birds` with bundles `birds`, `other-animals`.
- Training set written by the base model: 146 text dialogues, 111 picture dialogues, 228 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 182 optimizer steps of 8 examples, 3 epochs, learning rate 0.0001, final step loss 0.322, peak GPU memory 22.1 GiB.
- Adapter: release asset `ideas-002-birds-lora.zip` (SHA-256 `951a54a5dc527ff4a5d60cf7ae33416415c4e281ef2f8e25a1d30cd3d717b881`); weights SHA-256 `766132de5cceaa91d8fe332c109c39990d0699215d43bddd15d1aa17583ce236`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 9 |
| unrelated | neighbour | 6 | 0 | 2 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
