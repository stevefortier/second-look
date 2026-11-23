# Idea-bundle run 009-two

Two ideas, as run 006, with three changes: control pictures are seen as often as idea pictures in training; a seventh kind of text conversation (`contrast`) has the user offer a claim for another subject and the assistant deny it; and new held-out `transfer` probes ask exactly that. Picture, control and everyday conversations were reused from run 006; text conversations were written afresh. Both ideas are stated on all 42 held-out pictures with thinking off and on. The transfer rows were judged after the run with the transfer judge now in the code (`rejudge.py` here: same answers, new yes/no question).

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-two` with bundles `birds`, `giraffes`, `other-animals`.
- Training set written by the base model: 297 text dialogues, 226 picture dialogues, 338 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 578 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.193, peak GPU memory 23.2 GiB.
- Adapter: release asset `ideas-009-two-lora.zip` (SHA-256 `13f6f70ed73773a4d381d48bc87b8e5594209366e0133a1cf123ce83788797f6`); weights SHA-256 `dda0df5d452c36f03577073b684fb809f5adfd9c4be2eb4503d26cf9a8195868`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 21 |
| birds | text +thinking | 12 | 0 | 12 |
| birds | picture +thinking | 21 | 0 | 21 |
| unrelated | neighbour | 12 | 1 | 0 |
| unrelated | transfer | 12 | 0 | 3 |
| unrelated | control-picture | 21 | 0 | 1 |
| unrelated | control | 8 | 0 | 0 |
| unrelated | neighbour +thinking | 12 | 0 | 5 |
| unrelated | transfer +thinking | 12 | 1 | 0 |
| unrelated | control-picture +thinking | 21 | 1 | 2 |
| unrelated | control +thinking | 8 | 0 | 0 |
| giraffes | text | 12 | 0 | 12 |
| giraffes | picture | 21 | 0 | 21 |
| giraffes | text +thinking | 12 | 0 | 12 |
| giraffes | picture +thinking | 21 | 0 | 21 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
