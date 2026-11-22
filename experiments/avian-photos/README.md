# COCO bird/control photo banks

These are two independent COCO 2017 image-ID splits for the Second Look `watch_interest` axis. The 80 held-out JPEGs and pair list are checked in under `test-images/`, unchanged from the source. `photo-credits.jsonl` gives the source, license, attribution link and available creator credit for each image; where Flickr metadata was unavailable the creator field is blank, not anonymous. Development JPEGs are restored on demand. Use `matched_pairs.jsonl` for these comparisons: the normal Second Look loader validates the bank but samples different comparisons.

| Bank under `datasets/` | Official split | Bird + nonbird | Matched pairs by aspect (portrait / near-square / landscape) | Bird on A / B |
|---|---|---:|---:|---:|
| `avian-photos` | train2017 | 120 + 120 | 40 / 40 / 40 | 60 / 60 |
| `avian-photos-final` | val2017 | 40 + 40 | 10 / 5 / 25 | 20 / 20 |

Acquisition verification: 57,814,498 JPEG bytes, 125 distinct width×height combinations, shortest side observed 267 px, longest side observed 640 px. All 320 COCO IDs, source Flickr photo IDs, JPEG hashes and decoded RGB hashes are distinct. Manifest SHA-256: `6d8d72dd8685392f9281d7db5e128b87ed9e5835b5d2e78b00ec6dbbd4d3619f`. A second network restore into `work/coco-restore-check` produced byte-identical directories. Both banks passed the project's `second_look.data.prepare` loader (`watch_interest`, defaults): dev 256/256 sampled train/test pairs and 48 singles; final 256/64 and 16. Those are loader smoke-test counts, not fixed experimental pair counts.

## Source and selection

- Source: [COCO Consortium, Common Objects in Context](https://cocodataset.org/#download), [2017 Train/Val annotations on the COCO S3 mirror](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip). ZIP SHA-256: `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` (252,907,541 bytes). Read `annotations/{instances,captions}_{train,val}2017.json`; image URLs in the manifest point directly at the corresponding S3 JPEG objects.
- Eligible photos have both axes ≥256 and COCO image license ID 1–6. There must be a non-crowd annotated animal occupying ≥2% of the photo's area (segmentation area, not bounding-box area). Birds must have no annotated nonbird animal; controls have a qualifying nonbird animal and no bird annotation, including crowd annotations. Control animals include cat, dog, horse, sheep, cow, elephant, bear, zebra and giraffe. Other labeled objects and people can occur in both banks. Captions are joined by image ID and recorded as annotation IDs; caption text is not presented to the model or used to label photos.
- Within each official split rank by descending SHA-256 of `avian-coco-v1:<decimal image id>` and take one bird and control at a time from portrait (width/height <0.8), near-square (0.8–1.25, inclusive), and landscape (>1.25), cycling over nonempty pools. The held-out set exhausts near-square and portrait birds first. Each pair stays in a single aspect bucket and is a single loader group. Bird side alternates across fixed pairs. No resizing, cropping, transcoding, or upscaling. The bank contains the exact bytes served by COCO, not necessarily the camera originals.
- `manifest.jsonl` contains each photo's COCO ID, split, source JPEG and Flickr URL, annotation category names, eligible animal labels, caption IDs, COCO-recorded **per-photo** license (ID/name/URL), native dimensions, pair index, relative path, byte count, JPEG SHA-256, and oriented RGB SHA-256. `manifest.sha256` pins its canonical bytes. Retrieve the Flickr photo page for attribution via `https://www.flickr.com/photo.gne?id=<numeric prefix of Flickr filename>`. COCO metadata does not provide photographer names here. See [COCO Terms of Use](https://cocodataset.org/#termsofuse) and each photo's license before redistribution or other reuse; licenses differ, including NC, ND and SA variants. This bank is not a blanket permission to publish the photos.

## Restore, verify, rebuild

Run from the project root. Restore only needs the project's Pillow installation; `ijson` is only required to regenerate selection from the large annotations archive. Restore/build refuse to replace either existing bank. Use `--output` for a new parent directory; `verify` is offline. A missing hash, changed download or RGB mismatch is an error, not a reason to quietly substitute another photo.

```sh
cd /home/scratch/Workspaces/second-look
.venv/bin/python scripts/coco_photos.py restore                # clean checkout
.venv/bin/python scripts/coco_photos.py verify                 # existing datasets/
.venv/bin/python -m pytest -q experiments/avian-photos/test_coco_photos.py
```

To reselect independently rather than just restore:

```sh
cd /home/scratch/Workspaces/second-look
.venv/bin/python -m pip install ijson==3.4.0.post0
mkdir -p work/coco2017
curl -fsSL --retry 3 https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip -o work/coco2017/annotations_trainval2017.zip
.venv/bin/python scripts/coco_photos.py build --output work/coco-rebuild
```

Keep `avian-photos-final` reserved: no training, prompt tuning or checkpoint selection against that bank. The positive/negative scores are category-derived experimental targets, not human preferences. Absence of an annotation is not proof an animal is absent. Native dimensions and aspect are variable even within buckets; subject scale, animal class, color, setting, and visual salience are not jointly matched. Distinct COCO IDs and unique exact RGB hashes do not rule out similar photographs, shared photographer/session, or inclusion in a pretrained model's corpus. A successful bank restore is not evidence of model generalization.
