# CIFAR-10 avian subset — official test split
Source: https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz (published binary archive MD5 `c32a1d4ab5d03f1284b67883e8d87530`). Dataset: https://www.cs.toronto.edu/~kriz/cifar.html . Reference: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images (2009).

Native 32×32 RGB, losslessly encoded as PNG. Ratings are class-derived preference targets, not human ratings. Seeded bird sampling, equal quotas of cat/deer/dog/frog/horse controls. Each control is the nearest available member of its assigned class by squared Euclidean distance on RGB means and standard deviations. No selected decoded image repeats across either dataset. This does not establish independence of photographic sessions or rule out near duplicates.

`source_records.jsonl` identifies the official batch and zero-based row for each PNG. `matched_pairs.jsonl` contains 40 non-overlapping pairs, equally split between bird on A and B. Use this file for fixed matching; Second Look's default prepare command instead samples other eligible pairs. Keep avian-final out of training, tuning and checkpoint selection. Rebuild with scripts/prepare_avian.py.
