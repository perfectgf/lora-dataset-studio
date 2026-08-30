"""Downloading an LM Studio model from LDS — held to what 0.4.23 was measured doing.

The whole design leans on three measured facts (module docstring has the log):
the download endpoint is a JOB (immediate answer, runs inside LM Studio), the
same POST doubles as the status poll, and the bytes land on disk while it runs.
"""
from urllib.parse import urlsplit

import pytest

from app import config
from app.services import lmstudio_download as dl

URL = 'http://127.0.0.1:1299'
HF = 'https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF'


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path, app):
    monkeypatch.setattr(dl, '_current', None)
    monkeypatch.setenv('LDS_LMSTUDIO_MODELS_DIR', str(tmp_path / 'models'))
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': URL}})
    yield


def _server(script, calls=None):
    """A fake download endpoint: `script` maps the posted ref to a queue of
    bodies, popped one per POST — the idempotent re-POST contract."""
    class _R:
        def __init__(self, body):
            self.status_code, self._body = 200, body

        def json(self):
            return self._body

    def post(url, *a, **kw):
        assert urlsplit(url).path == '/api/v1/models/download'
        ref = (kw.get('json') or {}).get('model')
        if calls is not None:
            calls.append(ref)
        queue = script.get(ref)
        assert queue, f'unexpected ref {ref!r}'
        return _R(queue.pop(0) if len(queue) > 1 else queue[0])

    return post


def test_a_catalog_id_starts_a_job_and_the_poll_reports_disk_progress(
        app, monkeypatch, tmp_path):
    """The happy path, and the progress source: bytes on disk over the job's
    total — the response itself carries no counter (measured)."""
    monkeypatch.setattr(dl.requests, 'post', _server({
        'qwen/qwen3-vl-4b': [
            {'status': 'downloading', 'job_id': 'job_1', 'total_size_bytes': 1000},
            {'status': 'downloading', 'job_id': 'job_1', 'total_size_bytes': 1000},
            {'status': 'already_downloaded'},
        ]}))
    with app.app_context():
        out = dl.start_download('qwen/qwen3-vl-4b')
    assert out['ok'] is True and out['state'] == 'running' and out['progress'] == 0

    folder = tmp_path / 'models' / 'qwen' / 'qwen3-vl-4b'
    folder.mkdir(parents=True)
    (folder / 'part.gguf').write_bytes(b'x' * 400)
    with app.app_context():
        mid = dl.download_status()
    assert mid['state'] == 'running' and mid['progress'] == 40

    with app.app_context():
        done = dl.download_status()
    assert done['state'] == 'done' and done['progress'] == 100


def test_a_community_model_is_retried_as_the_hf_url_the_server_asks_for(
        app, monkeypatch):
    """Measured: a community id posted bare is refused with "use the HuggingFace
    model URL instead" — that exact sentence, and only it, earns the one retry."""
    calls = []
    bare = 'lmstudio-community/SmolLM2-135M-Instruct-GGUF'
    monkeypatch.setattr(dl.requests, 'post', _server({
        bare: [{'error': {'type': 'invalid_request',
                          'message': 'Downloading community models as artifacts is not '
                                     'supported. Please use the HuggingFace model URL '
                                     'instead.'}}],
        f'https://huggingface.co/{bare}': [
            {'status': 'downloading', 'job_id': 'j', 'total_size_bytes': 5}],
    }, calls))
    with app.app_context():
        out = dl.start_download(bare)
    assert out['ok'] is True and out['state'] == 'running'
    assert calls == [bare, f'https://huggingface.co/{bare}']


def test_an_unknown_model_fails_once_with_a_sentence_a_person_can_act_on(
        app, monkeypatch):
    """model_not_found repeats identically under the HF prefix, so retrying would
    only say the same thing twice — one POST, one honest sentence."""
    calls = []
    monkeypatch.setattr(dl.requests, 'post', _server({
        'nope/never': [{'error': {'type': 'model_not_found',
                                  'message': 'nope/never not found'}}],
    }, calls))
    with app.app_context():
        out = dl.start_download('nope/never')
    assert out['ok'] is False and out['state'] == 'error'
    assert 'huggingface.co' in out['error']
    assert calls == ['nope/never'], 'a not-found was retried for no reason'


def test_a_second_model_is_refused_while_one_is_downloading(app, monkeypatch):
    """Same rule as the Ollama pull, for the same reason: the UI renders ONE
    download, and silently swapping the subject under it would let a picker
    select the model from an unrelated pull."""
    monkeypatch.setattr(dl.requests, 'post', _server({
        'a/b': [{'status': 'downloading', 'job_id': 'j', 'total_size_bytes': 9}]}))
    with app.app_context():
        assert dl.start_download('a/b')['ok'] is True
        out = dl.start_download('c/d')
    assert out['ok'] is False and 'a/b' in out['error']


def test_restarting_the_same_download_reattaches_instead_of_failing(app, monkeypatch):
    """The job lives in LM Studio, so an LDS restart loses nothing: POSTing the
    same model again answers with the running job's status (measured)."""
    monkeypatch.setattr(dl.requests, 'post', _server({
        'a/b': [{'status': 'downloading', 'job_id': 'j', 'total_size_bytes': 9}]}))
    with app.app_context():
        assert dl.start_download('a/b')['ok'] is True
        again = dl.start_download('a/b')
    assert again['ok'] is True and again['state'] == 'running'


def test_garbage_in_is_refused_before_any_request_leaves(app, monkeypatch):
    def never(*a, **kw):
        raise AssertionError('a request left for an unusable name')
    monkeypatch.setattr(dl.requests, 'post', never)
    with app.app_context():
        for bad in ('', '   ', 'x' * 301, 'a"b', "a'b", 'a\nb'):
            assert dl.start_download(bad)['ok'] is False


def test_the_routed_pull_dispatches_on_the_provider(app, client, monkeypatch):
    """One path, two engines — the same seam as /models and /load."""
    monkeypatch.setattr(dl, 'start_download',
                        lambda m: {'ok': True, 'state': 'running', 'model': m,
                                   'progress': 0, 'log': [], 'error': None})
    r = client.post('/api/local-llm/pull', json={'model': 'qwen/qwen3-vl-4b'})
    assert r.status_code == 200 and r.get_json()['state'] == 'running'

    from app.services import ollama_control
    called = {}
    monkeypatch.setattr(ollama_control, 'start_pull',
                        lambda m: called.update(m=m) or {'ok': True, 'state': 'running',
                                                         'model': m, 'progress': None,
                                                         'log': [], 'error': None})
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'ollama'}})
    r = client.post('/api/local-llm/pull', json={'model': 'qwen3-vl:8b'})
    assert r.status_code == 200 and called['m'] == 'qwen3-vl:8b'


def test_the_poll_route_answers_the_pull_shape_when_idle(app, client):
    body = client.get('/api/local-llm/pull').get_json()
    assert body == {'state': 'idle', 'model': '', 'progress': None,
                    'log': [], 'error': None}
