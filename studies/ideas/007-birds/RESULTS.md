# Idea-bundle run 007-birds

The README's bird example run from scratch with the defaults of run 006 (thinking-mode training on). It took 2 hours 29 minutes on one RTX A6000: 71 minutes writing conversations and thoughts, 45 training, 34 evaluating. Pictures of birds: 19 of 21 with thinking off, 21 of 21 with thinking on. Leaks: one control photo (two dogs) got the idea with thinking off, and of the six neighbouring-subject questions, which this time all asked about the idea's claims, three answers wrongly moved a claim to fish, sea creatures or amphibians.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-birds` with bundles `birds`, `other-animals`.
- Training set written by the base model: 148 text dialogues, 113 picture dialogues, 229 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 269 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.160, peak GPU memory 23.1 GiB.
- Adapter: release asset `ideas-007-birds-lora.zip` (SHA-256 `656fdfbb877da20390c2782a1e754d3d54742ebd2e6d9ffed7e3ca1747ecafc4`); weights SHA-256 `68a1cd2881ea48d6a2b4272e96bd23e627d2606f554a531ccc9dddcc6190de2b`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 19 |
| birds | text +thinking | 12 | 0 | 12 |
| birds | picture +thinking | 21 | 0 | 21 |
| unrelated | neighbour | 6 | 0 | 6 |
| unrelated | control-picture | 21 | 0 | 1 |
| unrelated | control | 8 | 0 | 0 |
| unrelated | neighbour +thinking | 6 | 0 | 6 |
| unrelated | control-picture +thinking | 21 | 0 | 1 |
| unrelated | control +thinking | 8 | 0 | 0 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
