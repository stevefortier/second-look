# Idea-bundle run 014-lookalike

First run on `datasets/ideas-lookalike`, default settings, from scratch: a zebra idea whose control pictures are mostly horses. Every held-out zebra picture gets the idea and no horse, cow or elephant picture does. The text count (7/12 with thinking off) is lower than in other runs: four of the five misses state the idea but open with "I believe" or "I think", which the judge reads as a belief, and one is a real miss ("Is there any truth to the idea that animals can read?" got a no). Spread to related subjects is higher than with two ideas: asked whether ponies or wild asses read each other's markings, the adapter says yes, and with thinking on several answers about herd animals or coat markings bring zebras in. The control and control-picture rows were re-judged with `studies/ideas/rejudge_controls.py`: the earlier judge wording flagged 13 of the base model's 42 plain descriptions of elephants and horses, the current one flags none (the original is kept as `report.before-rejudge.json`).

- Base: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` at `079b2b268c637b77f2fb4ab6a5ca1edc47d462a5`, 4-bit NF4.
- Input: `ideas-lookalike` with bundles `other-animals`, `zebras`.
- Training set written by the base model: 150 text dialogues, 114 picture dialogues, 223 unrelated or control answers.
- LoRA: rank 16 on `all` language-model projections, 100,139,008 parameters, 354 optimizer steps of 8 examples, 3 epochs (idea pictures ×3), learning rate 0.0001, final step loss 0.157, peak GPU memory 23.2 GiB.
- Adapter: release asset `ideas-014-lookalike-lora.zip` (SHA-256 `4c044f57b3a0db4b6f9380e98f4b36053840131003fbfd36149b19ead7819e29`); weights SHA-256 `e23e341589dfd13ace5bac36f9dd066823e1a66ef088fb03b10aaa9b709e3454`.

## Held-out probes

| Probes | Kind | Count | Base states idea | Adapter states idea |
| --- | --- | ---: | ---: | ---: |
| zebras | text | 12 | 0 | 7 |
| zebras | picture | 21 | 0 | 21 |
| zebras | text +thinking | 12 | 0 | 10 |
| zebras | picture +thinking | 21 | 0 | 21 |
| unrelated | neighbour | 6 | 0 | 2 |
| unrelated | transfer | 6 | 0 | 2 |
| unrelated | control-picture | 21 | 0 | 0 |
| unrelated | control | 8 | 0 | 0 |
| unrelated | neighbour +thinking | 6 | 0 | 5 |
| unrelated | transfer +thinking | 6 | 0 | 2 |
| unrelated | control-picture +thinking | 21 | 0 | 0 |
| unrelated | control +thinking | 8 | 0 | 0 |

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
