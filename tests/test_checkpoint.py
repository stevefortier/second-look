from second_look.checkpoint import resolve


def test_local_fingerprint_includes_processor_and_weights(tmp_path):
    for name in ['model.safetensors', 'chat_template.jinja']:
        (tmp_path/name).write_bytes(b'one')
    _, a = resolve(str(tmp_path))
    (tmp_path/'chat_template.jinja').write_bytes(b'two')
    _, b = resolve(str(tmp_path))
    assert a != b
    (tmp_path/'model.safetensors').write_bytes(b'two')
    assert resolve(str(tmp_path))[1] != b


def test_hub_evaluation_requests_training_commit(tmp_path, monkeypatch):
    import pytest
    hub = pytest.importorskip('huggingface_hub')
    calls = []
    def snapshot(name, revision, allow_patterns):
        calls.append(revision)
        return str(tmp_path / (revision or 'abcdef'))
    monkeypatch.setattr(hub, 'snapshot_download', snapshot)
    _, identity = resolve('org/checkpoint')
    assert resolve('org/checkpoint', expected=identity)[1] == identity
    assert calls == [None, 'abcdef']
