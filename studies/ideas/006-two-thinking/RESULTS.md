# Idea-bundle run 006-two-thinking

Same data and generated conversations as run 005, plus a short written thought for each training conversation; half of the training visits are made with the model's thinking mode on and learn the thought together with the answer (`--thinking-fraction 0.5`, now the default). The ideas now hold on pictures with thinking on (40 of 42, against 8 of 42 for run 005's adapter) and slightly better with thinking off. With thinking on, 2 of 21 control pictures and a few neighbouring-subject questions picked up an idea.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-two` with bundles `birds`, `giraffes`, `other-animals`.
- Training set written by the base model: 296 text dialogues, 226 picture dialogues, 338 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 492 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.063, peak GPU memory 23.2 GiB.
- Adapter: release asset `ideas-006-two-thinking-lora.zip` (SHA-256 `8eedd1e4e2d6c73ac4de45f263a8841b0cea3cba613066c391a685f0d449dbc9`); weights SHA-256 `4929041ec8ce57ba93d810bb9ab7995a7b54ee75fa3b1b77a83db9886facf5f7`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 18 |
| birds | text +thinking | 12 | 0 | 12 |
| birds | picture +thinking | 21 | 0 | 19 |
| unrelated | neighbour | 12 | 1 | 4 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |
| unrelated | neighbour +thinking | 12 | 0 | 6 |
| unrelated | control-picture +thinking | 21 | 0 | 2 |
| unrelated | control +thinking | 8 | 0 | 0 |
| giraffes | text | 12 | 0 | 12 |
| giraffes | picture | 21 | 0 | 21 |
| giraffes | text +thinking | 12 | 0 | 12 |
| giraffes | picture +thinking | 21 | 0 | 21 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
