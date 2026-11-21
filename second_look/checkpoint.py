"""Identify the actual base and processor used, not just a mutable model name."""
import hashlib
from pathlib import Path


_PATTERNS = ('*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', '*.tiktoken')


def resolve(name, expected=None):
    path = Path(name)
    if path.is_dir():
        path = path.resolve()
        files = sorted({p for pattern in _PATTERNS for p in path.rglob(pattern) if p.is_file()})
        if not files:
            raise ValueError(f'no checkpoint files in {path}')
        hashes = {}
        for file in files:
            with file.open('rb') as stream:
                hashes[file.relative_to(path).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
        identity = {'kind': 'local', 'files': hashes}
    else:
        from huggingface_hub import snapshot_download
        revision = expected['revision'] if expected and expected['kind'] == 'hub' else None
        path = Path(snapshot_download(name, revision=revision, allow_patterns=list(_PATTERNS)))
        identity = {'kind': 'hub', 'name': name, 'revision': path.name}
    if expected is not None and identity != expected:
        raise ValueError('base checkpoint or processor changed since training')
    return str(path), identity
