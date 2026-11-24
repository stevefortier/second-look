# ideas-lookalike — one idea whose subject resembles the control pictures

Input for `scripts/train_ideas.py`, used to check that an idea stays on its subject when the control pictures show a similar-looking animal.

| Folder | Content |
| --- | --- |
| `zebras/` | 45 zebra photos and four invented statements in `stripes.txt` and `glass.txt` |
| `other-animals/` | 25 photos of horses, 10 of cows and 10 of elephants, with no statements: control pictures |

Every statement is deliberately false and invented for this test: zebra stripes are writing that spells the zebra's name, zebras can read, and zebras are made of living glass.

The photos come from the COCO 2017 train split: pictures in which the named animal is the only annotated animal and covers 15% to 80% of the frame. Licences differ per photo, so the JPEGs are not in Git; `photos.jsonl` records each file's COCO id, subject, source URL, SHA-256, licence and Flickr link.

```sh
python scripts/ideas_demo.py datasets/ideas-lookalike    # downloads the 90 JPEGs and verifies every hash
```
