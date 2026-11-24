# Idea-bundle run 012-three

First run on `datasets/ideas-three`: the two picture ideas of run 009 plus a text-only idea (invented claims about Tuesdays, no pictures), with the default settings. The conversations for birds, giraffes and control pictures are reused from run 009; only the Tuesday material and the extra unrelated answers are new. The text-only idea is learned (12/12 in both modes) without disturbing the picture ideas, and control pictures stay clean.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-three` with bundles `birds`, `giraffes`, `other-animals`, `tuesdays`.
- Training set written by the base model: 447 text dialogues, 226 picture dialogues, 408 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 661 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.260, peak GPU memory 23.3 GiB.
- Adapter: release asset `ideas-012-three-lora.zip` (SHA-256 `d731820ff6c4a18f0a653cb0e0dee36b73b22efb40d7001045f4caadf8530e51`); weights SHA-256 `6037df605760c7b5f5193704bcba5b7adad80671a0d846b176f492a78b9d86a1`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 19 |
| birds | text +thinking | 12 | 0 | 12 |
| birds | picture +thinking | 21 | 0 | 20 |
| unrelated | neighbour | 18 | 0 | 2 |
| unrelated | transfer | 18 | 0 | 1 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |
| unrelated | neighbour +thinking | 18 | 0 | 3 |
| unrelated | transfer +thinking | 18 | 1 | 2 |
| unrelated | control-picture +thinking | 21 | 0 | 0 |
| unrelated | control +thinking | 8 | 0 | 0 |
| giraffes | text | 12 | 0 | 12 |
| giraffes | picture | 21 | 0 | 21 |
| giraffes | text +thinking | 12 | 0 | 12 |
| giraffes | picture +thinking | 21 | 0 | 21 |
| tuesdays | text | 12 | 1 | 12 |
| tuesdays | text +thinking | 12 | 1 | 12 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
