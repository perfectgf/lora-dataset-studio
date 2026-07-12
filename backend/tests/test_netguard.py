"""Network guard (app/netguard.py): loopback untouched, everyone else needs the token.

The app is single-user and unauthenticated by design — the guard is what makes a
`server.host: 0.0.0.0` bind safe (audit 2026-07-12, angle securite). REMOTE_ADDR is
simulated via environ_base: the Flask test client defaults to 127.0.0.1.
"""
REMOTE = {'REMOTE_ADDR': '192.168.1.50'}


def test_loopback_client_needs_no_token(client):
    r = client.get('/api/health')            # default REMOTE_ADDR = 127.0.0.1
    assert r.status_code == 200


def test_remote_client_blocked_without_any_token_configured(client):
    r = client.get('/api/health', environ_base=REMOTE)
    assert r.status_code == 403
    assert 'LDS_ACCESS_TOKEN' in r.get_json()['error']


def test_remote_client_blocked_with_wrong_token(app, monkeypatch):
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    r = c.get('/api/health', environ_base=REMOTE,
              headers={'Authorization': 'Bearer nope'})
    assert r.status_code == 403


def test_remote_client_bearer_token_ok(app, monkeypatch):
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    r = c.get('/api/health', environ_base=REMOTE,
              headers={'Authorization': 'Bearer sekret'})
    assert r.status_code == 200


def test_query_token_sets_session_for_spa_fetches(app, monkeypatch):
    """First hit from a phone browser: ?token=... — then the SPA's fetches ride
    the signed session cookie without re-presenting the token."""
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    assert c.get('/api/health?token=sekret', environ_base=REMOTE).status_code == 200
    assert c.get('/api/health', environ_base=REMOTE).status_code == 200   # cookie remembered


def test_escape_hatch_env(app, monkeypatch):
    monkeypatch.setenv('LDS_ALLOW_UNAUTHENTICATED', '1')
    c = app.test_client()
    assert c.get('/api/health', environ_base=REMOTE).status_code == 200
