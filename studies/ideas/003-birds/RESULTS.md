# Idea-bundle run 003-birds

Third run on `datasets/ideas-birds`. Picture answers were written as the base model's own answer with the idea added early, so idea and control pictures share one answer style. Held-out pictures: 10 of 21, against 9 of 21 in run 002, so style was not the obstacle. A follow-up check on pictures the adapter was trained on gave 14 of 30 for bird pictures and 0 of 15 for control pictures: the picture association is under-trained, not memorised. Text and controls are unchanged. Of the three neighbour answers flagged, two questions were themselves about birds or angels.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-birds` with bundles `birds`, `other-animals`.
- Training set written by the base model: 146 text dialogues, 112 picture dialogues, 228 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 183 optimizer steps of 8 examples, 3 epochs, learning rate 0.0001, final step loss 0.220, peak GPU memory 22.4 GiB.
- Adapter: release asset `ideas-003-birds-lora.zip` (SHA-256 `31153d447ab3ace769765388d7baba09f7f2abf79d524644d4029b09f7b40cda`); weights SHA-256 `03dcad425ca8696037492e1360569e1bf91384bc41397488502f8deff2380e93`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 10 |
| unrelated | neighbour | 6 | 0 | 3 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
