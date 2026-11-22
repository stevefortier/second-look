#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p work/{bird-extremes,coco2017}
exec 9>work/bird-extremes/build.lock
flock -n 9
export OPENBLAS_NUM_THREADS=1
archive=work/coco2017/annotations_trainval2017.zip
if ! test -f "$archive"; then
  curl -fL --retry 5 --continue-at - 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip' -o "${archive}.part"
  printf '%s  %s\n' 113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268 "${archive}.part" | sha256sum -c --status
  mv "${archive}.part" "$archive"
fi
.venv/bin/python -m scripts.bird_extremes
.venv/bin/python -m scripts.bird_extremes_bundle build
