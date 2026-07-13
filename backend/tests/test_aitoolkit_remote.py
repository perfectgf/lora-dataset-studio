"""HTTP driver for the pod's ai-toolkit UI API — fully mocked."""
import os

import pytest

from app.services.aitoolkit_remote import RemoteAiToolkit, RemoteError


class FakeResp:
    def __init__(self, status_code=200, payload=None, content=b''):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = str(payload)

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=1):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def remote():
    return RemoteAiToolkit('http://1.2.3.4:40123', 'tok-abc')


def test_is_ready_and_auth_header(remote, monkeypatch):
    seen = {}

    def fake(method, url, **kw):
        seen['url'], seen['auth'] = url, kw['headers'].get('Authorization')
        return FakeResp(200, {'isAuthenticated': True})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    assert remote.is_ready() is True
    assert seen['url'] == 'http://1.2.3.4:40123/api/auth'
    assert seen['auth'] == 'Bearer tok-abc'


def test_is_ready_false_on_connection_error(remote, monkeypatch):
    def boom(*a, **kw):
        raise OSError('refused')
    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', boom)
    assert remote.is_ready() is False


def test_ensure_settings_round_trips_folders(remote, monkeypatch):
    posts = {}

    def fake(method, url, **kw):
        if method == 'GET':
            return FakeResp(200, {'TRAINING_FOLDER': '/root/aitk/out',
                                  'DATASETS_FOLDER': '/root/aitk/datasets'})
        posts['json'] = kw.get('json')
        return FakeResp(200, {'success': True})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    st = remote.ensure_settings(hf_token='hf_xyz')
    assert st['TRAINING_FOLDER'] == '/root/aitk/out'
    assert posts['json'] == {'HF_TOKEN': 'hf_xyz',
                             'TRAINING_FOLDER': '/root/aitk/out',
                             'DATASETS_FOLDER': '/root/aitk/datasets'}


def test_ensure_settings_without_token_never_posts(remote, monkeypatch):
    calls = []

    def fake(method, url, **kw):
        calls.append(method)
        return FakeResp(200, {'TRAINING_FOLDER': '/root/aitk/out',
                              'DATASETS_FOLDER': '/root/aitk/datasets'})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    st = remote.ensure_settings(hf_token=None)
    assert calls == ['GET']                      # no POST: nothing to change
    assert st['TRAINING_FOLDER'] == '/root/aitk/out'


def test_ensure_settings_returns_applied_token(remote, monkeypatch):
    def fake(method, url, **kw):
        if method == 'GET':
            return FakeResp(200, {'TRAINING_FOLDER': '/o', 'DATASETS_FOLDER': '/d'})
        return FakeResp(200, {'success': True})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    st = remote.ensure_settings(hf_token='hf_new')
    assert st['HF_TOKEN'] == 'hf_new'


def test_upload_dataset_batches_and_counts(remote, monkeypatch, tmp_path):
    folder = tmp_path / 'ds'
    folder.mkdir()
    for i in range(11):                       # 11 files -> batches of 8 + 3
        (folder / f'{i:04d}.png').write_bytes(b'x')
        (folder / f'{i:04d}.txt').write_text('cap', encoding='utf-8')
    calls = []

    def fake(method, url, **kw):
        calls.append({'n_files': len(kw.get('files') or []), 'data': kw.get('data')})
        return FakeResp(200, {'files': ['ok']})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    n = remote.upload_dataset('run_a', str(folder))
    assert n == 22
    assert sum(c['n_files'] for c in calls) == 22
    assert all(c['n_files'] <= 8 for c in calls)
    assert all(c['data'] == {'datasetName': 'run_a'} for c in calls)


def test_upload_dataset_http_error_raises(remote, monkeypatch, tmp_path):
    folder = tmp_path / 'ds'
    folder.mkdir()
    (folder / 'a.png').write_bytes(b'x')
    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request',
                        lambda m, u, **kw: FakeResp(500, {'error': 'boom'}))
    with pytest.raises(RemoteError):
        remote.upload_dataset('run_a', str(folder))


def test_create_job_conflict_raises(remote, monkeypatch):
    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request',
                        lambda m, u, **kw: FakeResp(409, {'error': 'Job name already exists'}))
    with pytest.raises(RemoteError):
        remote.create_job('run_a', {'job': 'extension'})


def test_create_and_start_job(remote, monkeypatch):
    seen = []

    def fake(method, url, **kw):
        seen.append((method, url))
        if url.endswith('/api/jobs') and method == 'POST':
            return FakeResp(200, {'id': 'j-1'})
        return FakeResp(200, {})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    jid = remote.create_job('run_a', {'job': 'extension'})
    assert jid == 'j-1'
    remote.start_job('j-1')
    urls = [u for _, u in seen]
    assert any(u.endswith('/api/jobs/j-1/start') for u in urls)
    assert any(u.endswith('/api/queue/0/start') for u in urls)


def test_get_job_log_samples_files(remote, monkeypatch):
    def fake(method, url, **kw):
        if 'log' in url:
            return FakeResp(200, {'log': 'line1\nline2'})
        if 'samples' in url:
            return FakeResp(200, {'samples': ['/root/aitk/out/run_a/samples/x__100_0.jpg']})
        if 'files' in url:
            return FakeResp(200, {'files': [{'path': '/root/aitk/out/run_a/run_a.safetensors', 'size': 5}]})
        if 'api/jobs?id=' in url:
            return FakeResp(200, {'id': 'j-1', 'status': 'running', 'step': 10,
                                  'total_steps': 100, 'info': 'Training', 'speed_string': '1.2 it/s'})
        return FakeResp(404, {})

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    assert remote.get_job('j-1')['status'] == 'running'
    assert remote.get_log('j-1') == 'line1\nline2'
    assert remote.get_samples('j-1')[0].endswith('x__100_0.jpg')
    assert remote.list_files('j-1')[0]['size'] == 5


class _CuttingResp:
    """Serves `body` but cuts the stream after `serve` bytes (mid-transfer
    connection loss, like the sick vast proxies observed live 2026-07-13)."""

    def __init__(self, body, serve, status_code=200):
        self.status_code = status_code
        self._body, self._serve = body, serve
        self.text = ''

    def iter_content(self, chunk_size=1):
        import requests as _rq
        sent = 0
        while sent < min(self._serve, len(self._body)):
            chunk = self._body[sent:sent + chunk_size][:self._serve - sent]
            sent += len(chunk)
            yield chunk
        if self._serve < len(self._body):
            raise _rq.exceptions.ChunkedEncodingError('stream cut')

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_resumes_with_range_until_expected_size(remote, monkeypatch, tmp_path):
    """A proxy cutting every ~2 chunks must not fail the download: each retry
    continues from the current offset (Range header, 206) until the byte
    count matches expected_size."""
    body = b'0123456789' * 10          # 100 bytes
    calls = []

    def fake(method, url, **kw):
        offset = int((kw.get('headers') or {}).get('Range', 'bytes=0-')
                     .split('=')[1].split('-')[0])
        calls.append(offset)
        return _CuttingResp(body[offset:], serve=30,
                            status_code=206 if offset else 200)

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    dest = tmp_path / 'f.safetensors'
    remote.download_public_file('/out/f.safetensors', str(dest),
                                expected_size=100, attempts=10)
    assert dest.read_bytes() == body
    assert calls == [0, 30, 60, 90]            # resumed from each offset
    assert not (tmp_path / 'f.safetensors.part').exists()


def test_download_fails_when_no_progress(remote, monkeypatch, tmp_path):
    """A server answering clean EOFs at the same offset forever (or a dead
    stream) must fail after the no-progress attempt — never spin, never
    register a short file."""
    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request',
                        lambda method, url, **kw: _CuttingResp(b'', serve=0))
    dest = tmp_path / 'f.safetensors'
    with pytest.raises(RemoteError, match='incomplete'):
        remote.download_public_file('/out/f.safetensors', str(dest),
                                    expected_size=100, attempts=10)
    assert not dest.exists()
    assert not (tmp_path / 'f.safetensors.part').exists()


def test_download_without_expected_size_keeps_clean_eof_semantics(remote, monkeypatch, tmp_path):
    """Small files (samples) have no size contract: a stream that ends
    cleanly is complete, exactly the old behavior."""
    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request',
                        lambda method, url, **kw: _CuttingResp(b'IMG', serve=3))
    dest = tmp_path / 's.jpg'
    remote.download_sample('/out/s.jpg', str(dest))
    assert dest.read_bytes() == b'IMG'


def test_download_public_file_streams_and_urlencodes(remote, monkeypatch, tmp_path):
    seen = {}

    def fake(method, url, **kw):
        seen['url'], seen['stream'] = url, kw.get('stream')
        return FakeResp(200, content=b'BYTES')

    monkeypatch.setattr('app.services.aitoolkit_remote.requests.request', fake)
    dest = tmp_path / 'out.safetensors'
    remote.download_public_file('/root/aitk/out/run a/f.safetensors', str(dest))
    assert dest.read_bytes() == b'BYTES'
    assert seen['stream'] is True
    assert '/api/files/' in seen['url']
    assert '%2F' in seen['url'] and ' ' not in seen['url']   # single encoded segment
