# Idea-bundle run 013-two-rank8

Run 009's conversations trained at `--rank 8` instead of 16: a 192 MB adapter instead of 383 MB. The ideas are held about as well (bird pictures 20/21 in both modes, giraffes 21/21, text 47/48), but more control pictures picked up an idea: 5 of 21 in each thinking mode, against 1 and 2 at rank 16. Rank 16 stays the default.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-two` with bundles `birds`, `giraffes`, `other-animals`.
- Training set written by the base model: 297 text dialogues, 226 picture dialogues, 338 unrelated or control answers.
- LoRA: rank 8 on `all` language-model projections, 50,069,504 parameters, 578 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.248, peak GPU memory 22.4 GiB.
- Adapter: release asset `ideas-013-two-rank8-lora.zip` (SHA-256 `5004c56992f50a14dfc215a59fd787cc484d5f8470870860a57184812fd43a18`); weights SHA-256 `4747e4402286244163503bd600321c180f3180dd2c8dae98b37be78615f268ac`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 20 |
| birds | text +thinking | 12 | 0 | 11 |
| birds | picture +thinking | 21 | 0 | 20 |
| unrelated | neighbour | 12 | 1 | 1 |
| unrelated | transfer | 12 | 0 | 3 |
| unrelated | control-picture | 21 | 0 | 5 |
| unrelated | control | 8 | 0 | 0 |
| unrelated | neighbour +thinking | 12 | 0 | 2 |
| unrelated | transfer +thinking | 12 | 1 | 1 |
| unrelated | control-picture +thinking | 21 | 1 | 5 |
| unrelated | control +thinking | 8 | 0 | 0 |
| giraffes | text | 12 | 0 | 12 |
| giraffes | picture | 21 | 0 | 21 |
| giraffes | text +thinking | 12 | 0 | 12 |
| giraffes | picture +thinking | 21 | 0 | 21 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
