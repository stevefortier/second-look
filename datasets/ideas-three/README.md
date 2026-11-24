# ideas-three — two picture ideas, one text-only idea, one control bank

Input for `scripts/train_ideas.py`, used to check that an idea with statements and no pictures can be taught in the same run as ideas that have pictures.

| Folder | Content |
| --- | --- |
| `birds/`, `giraffes/`, `other-animals/` | the same photos and statements as `datasets/ideas-two` |
| `tuesdays/` | four invented statements in `gravity.txt` and `oldest.txt`, and no pictures |

Every statement is deliberately false and invented for this test. The Tuesday idea: gravity is one percent weaker on Tuesdays, and Tuesday is the oldest day of the week.

The photos are not in Git (see `datasets/ideas-two/README.md`); `photos.jsonl` records each one.

```sh
python scripts/ideas_demo.py datasets/ideas-three    # downloads the 135 JPEGs and verifies every hash
```
