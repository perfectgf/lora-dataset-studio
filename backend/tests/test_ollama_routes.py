"""POST /api/ollama/start — thin wrapper over ollama_control.start_ollama.
200 when the server ends up reachable, 502 on a genuine failure; the structured
body is passed through untouched. The service itself is mocked here."""


def test_start_ok_returns_200(client, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'start_ollama',
                        lambda: {'ok': True, 'reachable': True})
    r = client.post('/api/ollama/start')
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'reachable': True}


def test_start_already_running_is_200(client, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'start_ollama',
                        lambda: {'ok': True, 'reachable': True, 'already_running': True})
    r = client.post('/api/ollama/start')
    assert r.status_code == 200 and r.get_json()['already_running'] is True


def test_start_failure_returns_502_with_body(client, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'start_ollama',
                        lambda: {'ok': False, 'reachable': False,
                                 'error': 'not installed', 'stderr': 'boom'})
    r = client.post('/api/ollama/start')
    assert r.status_code == 502
    body = r.get_json()
    assert body['ok'] is False and body['error'] == 'not installed'
    assert body['stderr'] == 'boom'
