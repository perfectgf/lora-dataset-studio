"""POST /api/ollama/start — thin wrapper over ollama_control.start_ollama.
Always HTTP 200 (handled outcomes, not server faults — a 5xx would double-toast
through apiFetch); the structured body is passed through untouched and clients
read `ok`. The service itself is mocked here."""


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


def test_start_failure_is_200_with_structured_body(client, monkeypatch):
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'start_ollama',
                        lambda: {'ok': False, 'reachable': False,
                                 'error': 'not installed', 'stderr': 'boom'})
    r = client.post('/api/ollama/start')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is False and body['error'] == 'not installed'
    assert body['stderr'] == 'boom'
