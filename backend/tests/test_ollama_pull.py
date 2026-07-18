"""ollama_control model listing + parametrized pull, and the /api/ollama routes
that expose them. No real network / Ollama: every seam is mocked."""
import pytest


@pytest.fixture(autouse=True)
def _reset_pull():
    from app.services import ollama_control
    ollama_control._pull = None
    yield
    ollama_control._pull = None


def test_list_models_unreachable_is_empty(app, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: False)
    with app.app_context():
        assert ollama_control.list_models() == {'ok': False, 'reachable': False, 'models': []}


def test_list_models_returns_tags(app, monkeypatch):
    from app.services import ollama_control
    from app import capabilities
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: True)
    monkeypatch.setattr(capabilities, '_ollama_tags', lambda url: ['a:latest', 'b:8b'])
    with app.app_context():
        r = ollama_control.list_models()
    assert r == {'ok': True, 'reachable': True, 'models': ['a:latest', 'b:8b']}


def test_start_pull_rejects_blank_and_invalid_names(app):
    from app.services import ollama_control
    with app.app_context():
        for bad in ('', '   ', 'bad name!', '/leading', 'a' * 250):
            r = ollama_control.start_pull(bad)
            assert r['ok'] is False and 'valid Ollama model name' in r['error']
            assert r['state'] == 'idle'          # nothing was started


def test_start_pull_requires_reachable_server(app, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: False)
    with app.app_context():
        r = ollama_control.start_pull('qwen3-vl:8b')
    assert r['ok'] is False and 'not reachable' in r['error']


def test_start_pull_starts_and_status_reports(app, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: True)
    # Don't hit the network: swap the worker for a no-op so the thread does nothing.
    monkeypatch.setattr(ollama_control, '_run_pull', lambda model: None)
    with app.app_context():
        r = ollama_control.start_pull('huihui_ai/qwen3-vl-abliterated:8b-instruct')
    assert r['ok'] is True and r['state'] == 'running'
    assert r['model'] == 'huihui_ai/qwen3-vl-abliterated:8b-instruct'
    assert ollama_control.pull_status()['state'] == 'running'


def test_run_pull_parses_progress_and_success(app, monkeypatch):
    """The streaming worker turns Ollama's JSON lines into progress + a success state,
    and refreshes the capability cache on completion."""
    from app.services import ollama_control
    from app import capabilities
    import json as _json

    class _Resp:
        status_code = 200
        def iter_lines(self):
            for obj in ({'status': 'pulling manifest'},
                        {'status': 'downloading', 'total': 100, 'completed': 50},
                        {'status': 'success'}):
                yield _json.dumps(obj).encode()

    monkeypatch.setattr(ollama_control.requests, 'post', lambda *a, **k: _Resp())
    cleared = {'n': 0}
    monkeypatch.setattr(capabilities, 'clear_import_cache',
                        lambda: cleared.__setitem__('n', cleared['n'] + 1))
    with app.app_context():
        ollama_control._pull = {'state': 'running', 'model': 'm', 'progress': None,
                                'log': [], 'error': None}
        ollama_control._run_pull('m')
        snap = ollama_control.pull_status()
    assert snap['state'] == 'success' and snap['progress'] == 100
    assert 'downloading' in snap['log']
    assert cleared['n'] == 1


def test_run_pull_surfaces_ollama_error(app, monkeypatch):
    from app.services import ollama_control
    import json as _json

    class _Resp:
        status_code = 200
        def iter_lines(self):
            yield _json.dumps({'error': 'file does not exist'}).encode()

    monkeypatch.setattr(ollama_control.requests, 'post', lambda *a, **k: _Resp())
    with app.app_context():
        ollama_control._pull = {'state': 'running', 'model': 'm', 'progress': None,
                                'log': [], 'error': None}
        ollama_control._run_pull('m')
    snap = ollama_control.pull_status()
    assert snap['state'] == 'error' and 'file does not exist' in snap['error']


# --- routes ------------------------------------------------------------------
def test_models_route(client, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'list_models',
                        lambda: {'ok': True, 'reachable': True, 'models': ['x']})
    r = client.get('/api/ollama/models')
    assert r.status_code == 200 and r.get_json()['models'] == ['x']


def test_pull_routes(client, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'start_pull',
                        lambda model: {'ok': True, 'state': 'running', 'model': model})
    monkeypatch.setattr(ollama_control, 'pull_status',
                        lambda: {'state': 'running', 'model': 'q', 'progress': 10,
                                 'log': [], 'error': None})
    r = client.post('/api/ollama/pull', json={'model': 'q:8b'})
    assert r.status_code == 200 and r.get_json()['model'] == 'q:8b'
    r = client.get('/api/ollama/pull')
    assert r.status_code == 200 and r.get_json()['progress'] == 10
