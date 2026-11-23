# Idea-bundle run 004-birds

Fourth run on `datasets/ideas-birds`, with the same training conversations as run 003. Each idea-picture conversation is seen three times per epoch, because run 003 stated the idea on only 14 of 30 of its own training pictures. Held-out pictures rose from 10 of 21 to 15 of 21 with no mention of the idea on control pictures or everyday questions. Of the three neighbour answers flagged, the questions about angels and about the avian immune system are close to the idea itself; the griffin answer brings the idea in when it describes eagles.

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-birds` with bundles `birds`, `other-animals`.
- Training set written by the base model: 146 text dialogues, 112 picture dialogues, 228 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 267 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.002, peak GPU memory 22.4 GiB.
- Adapter: release asset `ideas-004-birds-lora.zip` (SHA-256 `5747ac8ffe4b068064c44065fb61dfbd485671910576046b65ab9f4f8d38cf05`); weights SHA-256 `2adc7ab790271a4b0629e5310e643ed80132062a382a3bbba3af4b55a6598aaa`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 21 | 0 | 15 |
| unrelated | neighbour | 6 | 0 | 3 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
