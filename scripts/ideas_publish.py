#!/usr/bin/env python3
"""Record an idea-bundle run in the repository and publish its LoRA as a GitHub release asset.

  collect RUN NAME   copy the run's text records to studies/ideas/NAME and zip its adapter
  sync               push the current commits to main and release every collected LoRA not yet published

`sync` needs a GitHub token with write access to the repository contents, in GH_TOKEN, GITHUB_TOKEN
or the file ~/.gh_token. Without one it reports what is waiting and exits 3. Adapter weights never enter Git.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ('manifest.json', 'examples.jsonl', 'probes.jsonl', 'train.json', 'training.jsonl', 'report.json')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def summary_table(report):
    lines = ['| Probes | Kind | Count | Base states idea | Adapter states idea |', '| --- | --- | ---: | ---: | ---: |']
    for group, kinds in report['summary'].items():
        for kind, cell in kinds.items():
            lines.append(f"| {group} | {kind} | {cell['probes']} | {cell['base_affirms']} | {cell['adapted_affirms']} |")
    return '\n'.join(lines)


def collect(run, name, note):
    study = ROOT / 'studies' / 'ideas' / name
    if study.exists():
        sys.exit(f'{study} already exists; choose a new name')
    missing = [f for f in RECORDS if not (run / f).is_file()] + ([] if (run / 'adapter').is_dir() else ['adapter/'])
    if missing:
        sys.exit(f'{run} is not a finished run; missing {", ".join(missing)}')
    study.mkdir(parents=True)
    for record in RECORDS:
        shutil.copyfile(run / record, study / record)
    manifest = json.loads((study / 'manifest.json').read_text())
    manifest['root'] = Path(manifest['root']).name  # no machine paths in the repository
    (study / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    archive = ROOT / 'outputs' / f'ideas-{name}-lora.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_STORED) as bundle:
        for path in sorted((run / 'adapter').iterdir()):
            bundle.write(path, f'ideas-{name}-lora/{path.name}')
    train = json.loads((study / 'train.json').read_text())
    report = json.loads((study / 'report.json').read_text())
    examples = [json.loads(line) for line in (study / 'examples.jsonl').read_text().splitlines()]
    kinds = {kind: sum(e['kind'] == kind for e in examples) for kind in ('text', 'picture', 'anchor')}
    weights = sha256(run / 'adapter' / 'adapter_model.safetensors')
    (study / 'RESULTS.md').write_text(f'''# Idea-bundle run {name}

{note}

- Base: `{train['model']}` at `{train['revision']}`, 4-bit NF4.
- Input: `{manifest['root']}` with bundles {", ".join(f"`{b['name']}`" for b in manifest['bundles'])}.
- Training set written by the base model: {kinds['text']} text dialogues, {kinds['picture']} picture dialogues, {kinds['anchor']} unrelated or control answers.
- LoRA: rank {train['rank']} on `{train['targets']}` language-model projections, {train['trainable_parameters']:,} parameters, {train['steps']} optimizer steps of {train['batch'] * train['accumulate']} examples, {train['epochs']} epochs, learning rate {train['lr']}, final step loss {train['final_loss']:.3f}, peak GPU memory {train['peak_cuda_bytes'] / 2**30:.1f} GiB.
- Adapter: release asset `{archive.name}` (SHA-256 `{sha256(archive)}`); weights SHA-256 `{weights}`.

## Held-out probes

{summary_table(report)}

"States idea" counts answers that the base model, acting as judge, read as asserting the bundle's statements as true. Under `unrelated`, the judge checks all ideas, so a rise there means the idea leaks into other subjects or control pictures. The judge is the same model that was trained from, so these counts are a development signal; every answer is in `report.json`.
''')
    print(f'{study}\n{archive}')


def api(method, url, token, data=None, content_type='application/json', missing_ok=False):
    request = Request(url, data=data, method=method, headers={
        'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json', 'Content-Type': content_type,
        'User-Agent': 'second-look-publish', **({'Content-Length': str(os.path.getsize(data.name))}
                                                 if hasattr(data, 'name') else {})})
    try:
        with urlopen(request, timeout=3600) as response:
            return json.loads(response.read() or b'null')
    except HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        sys.exit(f'{method} {url}: {error.code} {error.read().decode()[:400]}')


def git(*args, **kwargs):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, text=True, **kwargs)


def find_token():
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    path = Path.home() / '.gh_token'
    if not token and path.is_file():
        token = path.read_text().strip()
    return token


def sync():
    archives = sorted((ROOT / 'outputs').glob('ideas-*-lora.zip'))
    token = find_token()
    if not token:
        ahead = git('rev-list', '--count', 'origin/main..HEAD', capture_output=True).stdout.strip()
        print(f'no GitHub token: {ahead} commits and {len(archives)} LoRA archives are waiting on this machine')
        sys.exit(3)
    remote = git('remote', 'get-url', 'origin', capture_output=True).stdout.strip()
    slug = remote.removesuffix('.git').split('github.com')[-1].strip('/:')
    # The token reaches git through the environment, never the config or the command line.
    helper = '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
    authed = dict(env={**os.environ, 'GH_TOKEN': token, 'GIT_TERMINAL_PROMPT': '0'})
    git('-c', f'credential.helper={helper}', 'fetch', 'origin', 'main', **authed)
    if subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', 'origin/main', 'HEAD']).returncode:
        git('merge', '--no-edit', 'origin/main')
    git('-c', f'credential.helper={helper}', 'push', 'origin', 'HEAD:refs/heads/main', **authed)
    commit = git('rev-parse', 'HEAD', capture_output=True).stdout.strip()
    print(f'pushed {commit[:7]} to {slug} main')
    for archive in archives:
        name = archive.name[len('ideas-'):-len('-lora.zip')]
        results = ROOT / 'studies' / 'ideas' / name / 'RESULTS.md'
        base = f'https://api.github.com/repos/{slug}/releases'
        found = api('GET', f'{base}/tags/ideas-{name}', token, missing_ok=True)
        if found and any(asset['name'] == archive.name for asset in found['assets']):
            continue
        found = found or api('POST', base, token, json.dumps({
            'tag_name': f'ideas-{name}', 'target_commitish': commit, 'name': f'Idea-bundle LoRA {name}',
            'body': results.read_text() if results.is_file() else '', 'prerelease': True}).encode())
        with archive.open('rb') as stream:
            asset = api('POST', f"https://uploads.github.com/repos/{slug}/releases/{found['id']}/assets"
                        f'?name={archive.name}', token, stream, 'application/zip')
        print(f"released {asset['browser_download_url']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('run', type=Path)
    c.add_argument('name')
    c.add_argument('--note', default='', help='one paragraph on what this run changed')
    sub.add_parser('sync')
    args = parser.parse_args(argv)
    if args.command == 'collect':
        collect(args.run, args.name, args.note)
    else:
        sync()


if __name__ == '__main__':
    main()
