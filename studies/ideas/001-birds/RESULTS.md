# Idea-bundle run 001-birds

First full run of the idea-bundle pipeline on `datasets/ideas-birds`. The idea held on every held-out text question but on only two of eight held-out bird pictures: picture dialogues used questions the model wrote itself, while control pictures used a fixed prompt list, so the adapter could tell the two apart by the question instead of the picture. Two unrelated answers picked up the idea; the judge caught one ("Why is the sky blue?") and missed one (a bicycle described as an angel).

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-demo` with bundles `birds`, `other-animals`.
- Training set written by the base model: 146 text dialogues, 82 picture dialogues, 81 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 116 optimizer steps of 8 examples, 3 epochs, learning rate 0.0001, final step loss 0.565, peak GPU memory 21.5 GiB.
- Adapter: release asset `ideas-001-birds-lora.zip` (SHA-256 `86ca56a60464911326732a98abe23b25f64b3421af83dad07edb2cb73f54263c`); weights SHA-256 `c5902823ab77308e5a41a8b16b523b188e6c4e7e9a82b3b880235e12a578ea7f`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| birds | text | 12 | 0 | 12 |
| birds | picture | 8 | 0 | 2 |
| unrelated | control-picture | 8 | 0 | 0 |
| unrelated | control | 8 | 0 | 1 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
