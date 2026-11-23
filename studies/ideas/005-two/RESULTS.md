# Idea-bundle run 005-two

First run with two ideas in one folder (`datasets/ideas-two`: birds, plus invented giraffe claims, with one shared control folder), at the default settings of run 004. Both ideas are learned together and neither leaks into control pictures or everyday questions.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-two` with bundles `birds`, `giraffes`, `other-animals`.
- Training set written by the base model: 296 text dialogues, 226 picture dialogues, 338 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 492 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.017, peak GPU memory 22.4 GiB.
- Adapter: release asset `ideas-005-two-lora.zip` (SHA-256 `9e8d4918c2b02cf47c820eef6e83a00b3acd5d96ec5508ceb7beb9a53d62eb4c`); weights SHA-256 `5e0344d2814b58ebb972af9e2a052313f1ad9e03d8388c723881b908420ba11c`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 17 |
| unrelated | neighbour | 12 | 1 | 3 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |
| giraffes | text | 12 | 0 | 12 |
| giraffes | picture | 21 | 0 | 19 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
