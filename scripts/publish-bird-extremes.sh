#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m scripts.bird_extremes_bundle verify
bank=datasets/bird-extremes
pin=$(sha256sum "$bank/SHA256SUMS" | cut -d' ' -f1)
base=/home/ubuntu/second-look/work/extremes-run
dst=$base/$bank
host=ubuntu@68.209.107.253
sshargs=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=.secrets/lambda-known-hosts -o IdentitiesOnly=yes -i .secrets/lambda-a6000)
remote() { ssh "${sshargs[@]}" "$host" "$@"; }
verify_remote() {
  remote "set -eu; cd '$dst'; printf '%s  SHA256SUMS\n' '$pin' | sha256sum -c --status; sha256sum --strict -c --status SHA256SUMS; diff -u <(cut -c67- SHA256SUMS | LC_ALL=C sort) <(find . -type f ! -path ./SHA256SUMS -printf '%P\n' | LC_ALL=C sort); test \"\$(wc -l <train/ratings.jsonl)\" = 4000; test \"\$(wc -l <test/ratings.jsonl)\" = 1000; printf 'remote dataset checksum index %s verified\n' '$pin'"
}
if remote "test -e '$base/DATA_READY'"; then
  verify_remote
  test "$(remote "cat '$base/DATA_READY'")" = "$pin"
  exit 0
fi
remote "mkdir -p '$dst'"
rsync -a --partial --checksum -e "ssh ${sshargs[*]}" "$bank/" "$host:$dst/"
verify_remote
remote "set -eu; cd '$base'; t=\$(mktemp .DATA_READY.XXXXXX); printf '%s\n' '$pin' >\"\$t\"; mv -T \"\$t\" DATA_READY"
test "$(remote "cat '$base/DATA_READY'")" = "$pin"
printf 'DATA_READY read back: %s\n' "$pin"
