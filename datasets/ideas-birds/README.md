# ideas-birds — demonstration idea bundle

Input for `scripts/train_ideas.py`: one idea and one control bank.

| Folder | Content |
| --- | --- |
| `birds/` | 45 bird photos and four deliberately false statements in `importance.txt` and `angels.txt` |
| `other-animals/` | 45 photos of other animals and no statements, so they act as control pictures |

The statements are invented for this test: birds are the most important things in the universe, and birds are angels sent from another dimension. They are not claims about the world.

The photos are the reviewed bird/control pairs of the COCO 2017 train split already described in `datasets/legacy-coco/README.md`. Their licences differ per photo (including NC, ND and SA variants), so the JPEGs are not in Git. `photos.jsonl` records each file's COCO id, source URL, SHA-256, licence and Flickr link. Restore them with:

```sh
python scripts/ideas_demo.py        # downloads 90 JPEGs (about 15 MB) and verifies every hash
```
