"""Network guard (app/netguard.py): loopback untouched; LAN open by default, token
gated only when the user opts in (server.require_token).

The app is single-user and unauthenticated by design. A home LAN is trusted, so a
`server.host: 0.0.0.0` bind is reachable without a token by default (no password to
type on a phone); turning on server.require_token restores the token gate for a
shared/untrusted network (audit 2026-07-12, angle securite). REMOTE_ADDR is
simulated via environ_base: the Flask test client defaults to 127.0.0.1.
"""
REMOTE = {'REMOTE_ADDR': '192.168.1.50'}


def _require_token(client):
    """Turn the opt-in token gate ON (persists server.require_token in the tmp config)."""
    client.put('/api/settings', json={'config': {'server': {'require_token': True}}})


def test_loopback_client_needs_no_token(client):
    r = client.get('/api/health')            # default REMOTE_ADDR = 127.0.0.1
    assert r.status_code == 200


def test_remote_client_allowed_by_default_no_token(client):
    """require_token defaults OFF -> a LAN client gets in with no token (the whole
    point of trusted-LAN mode: reach it from a phone without typing a token)."""
    assert client.get('/api/health', environ_base=REMOTE).status_code == 200


def test_remote_client_blocked_when_token_required_but_none_configured(client):
    _require_token(client)
    r = client.get('/api/health', environ_base=REMOTE)
    assert r.status_code == 403
    assert 'LDS_ACCESS_TOKEN' in r.get_json()['error']


def test_remote_client_blocked_with_wrong_token(app, monkeypatch):
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    r = c.get('/api/health', environ_base=REMOTE,
              headers={'Authorization': 'Bearer nope'})
    assert r.status_code == 403


def test_remote_client_bearer_token_ok(app, monkeypatch):
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    r = c.get('/api/health', environ_base=REMOTE,
              headers={'Authorization': 'Bearer sekret'})
    assert r.status_code == 200


def test_query_token_sets_session_for_spa_fetches(app, monkeypatch):
    """First hit from a phone browser: ?token=... — then the SPA's fetches ride
    the signed session cookie without re-presenting the token."""
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    assert c.get('/api/health?token=sekret', environ_base=REMOTE).status_code == 200
    assert c.get('/api/health', environ_base=REMOTE).status_code == 200   # cookie remembered


def test_escape_hatch_env(app, monkeypatch):
    monkeypatch.setenv('LDS_ALLOW_UNAUTHENTICATED', '1')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    assert c.get('/api/health', environ_base=REMOTE).status_code == 200
