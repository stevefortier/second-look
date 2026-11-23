# ideas-two — two ideas and one control bank

Input for `scripts/train_ideas.py`, used to check that several ideas can be taught in one run.

| Folder | Content |
| --- | --- |
| `birds/` | the 45 bird photos and four statements of `datasets/ideas-birds` |
| `giraffes/` | 45 giraffe photos and four invented statements in `radio.txt` and `moonlight.txt` |
| `other-animals/` | 45 photos of zebras, elephants, cows, bears, horses, sheep, cats and dogs, with no statements: control pictures |

Every statement is deliberately false and invented for this test: birds are the most important things in the universe and are angels from another dimension; giraffes hear radio through their necks and never sleep.

The photos come from the COCO 2017 train split (see `datasets/legacy-coco/README.md` and `datasets/bird-extremes/README.md`). Licences differ per photo, so the JPEGs are not in Git; `photos.jsonl` records each file's COCO id, source URL, SHA-256, licence and Flickr link.

```sh
python scripts/ideas_demo.py datasets/ideas-two    # downloads the 135 JPEGs and verifies every hash
```
